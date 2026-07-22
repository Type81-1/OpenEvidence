from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )

# ============================================================
# 必须在导入 torch、transformers、sentence_transformers 之前设置
# 隐藏 NVIDIA 显卡，避免在没有 NVIDIA 显卡的机器上误选 CUDA
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import html
import re

import chromadb
import torch
from sentence_transformers import SentenceTransformer


MetadataValue = str | int | float | bool
Metadata = dict[str, MetadataValue]


# ============================================================
# 基础配置
# ============================================================

# Chroma 数据库保存位置：
# E:\OpenEvidence\chroma_db
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "medical_evidence"

# 多语言模型：
# 可以处理“中文问题 + 英文 PubMed 文献”的跨语言检索
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

EMBEDDING_DEVICE = os.getenv(
    "EMBEDDING_DEVICE",
    "cpu",
).strip().lower()


def resolve_embedding_device(device_name: str) -> str:
    """
    解析本地向量模型使用的设备。

    可选值：
    - cpu：最稳，默认值
    - mps：Apple Silicon GPU
    - auto：Mac 上优先使用 mps，否则退回 cpu
    """

    if device_name == "auto":
        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    if device_name == "mps":
        if torch.backends.mps.is_available():
            return "mps"

        print(
            "提示：当前 PyTorch 不支持 mps，"
            "已自动退回 CPU。"
        )

        return "cpu"

    return "cpu"


DEVICE = resolve_embedding_device(
    EMBEDDING_DEVICE
)


# ============================================================
# 加载向量模型
# ============================================================

print(f"正在加载向量模型：{MODEL_NAME}")

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE,
)

print(f"向量模型设备：{model.device}")


# ============================================================
# 初始化 Chroma
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DB_DIR),
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
)


# ============================================================
# 文本清洗
# ============================================================

def clean_text(text: str) -> str:
    """
    清洗普通文本或 PubMed 返回的 XML。

    主要处理：
    1. 删除 XML/HTML 标签
    2. 转换 &amp; 等 HTML 实体
    3. 合并多余空格和空行
    """

    if not text:
        return ""

    # 去除 XML / HTML 标签
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # 将 &amp;、&lt; 等实体转换为普通字符
    text = html.unescape(text)

    # 合并连续空白字符
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# 长文本切分
# ============================================================

def split_text(
    text: str,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[str]:
    """
    将长文本切成多个文本块。

    参数：
    max_chars：
        每个文本块最多大约多少字符。

    overlap：
        相邻文本块之间保留多少重叠字符，
        避免重要句子刚好被切断。
    """

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return []

    if len(cleaned_text) <= max_chars:
        return [cleaned_text]

    chunks: list[str] = []

    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(
            start + max_chars,
            text_length,
        )

        # 如果还没有到文本末尾，尝试在句号附近切分
        if end < text_length:
            search_start = start + max_chars // 2

            possible_breaks = [
                cleaned_text.rfind(". ", search_start, end),
                cleaned_text.rfind("。", search_start, end),
                cleaned_text.rfind("; ", search_start, end),
                cleaned_text.rfind("；", search_start, end),
            ]

            best_break = max(possible_breaks)

            if best_break > start:
                end = best_break + 1

        chunk = cleaned_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        # 保留一定重叠区域
        next_start = end - overlap

        # 防止意外出现死循环
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================
# 删除某个来源的旧文本块
# ============================================================

def delete_existing_document(doc_id: str) -> None:
    """
    删除同一 doc_id 以前写入的文本块。

    这样重复运行 main.py 时，旧内容不会残留。
    """

    try:
        existing = collection.get(
            where={
                "source_id": doc_id,
            }
        )

        existing_ids = existing.get(
            "ids",
            [],
        )

        if existing_ids:
            collection.delete(
                ids=existing_ids,
            )

    except Exception as exc:
        # 删除失败不会阻止后续写入
        print(
            f"提示：清理旧文档时出现问题，"
            f"将继续执行：{exc}"
        )


def normalize_metadata(
    metadata: dict | None,
) -> Metadata:
    """
    Chroma metadata 只支持简单标量；这里统一清洗。
    """

    if not metadata:
        return {}

    normalized: Metadata = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, bool):
            normalized[key] = value
        elif isinstance(value, int | float):
            normalized[key] = value
        elif isinstance(value, str):
            normalized[key] = value
        else:
            normalized[key] = str(value)

    return normalized


# ============================================================
# 文档入库
# ============================================================

def add_document(
    text: str,
    doc_id: str,
    metadata: dict | None = None,
) -> int:
    """
    将一篇文档切分、向量化并写入 Chroma。

    返回值：
        实际写入的文本块数量。
    """

    if not doc_id or not doc_id.strip():
        raise ValueError(
            "doc_id 不能为空。"
        )

    chunks = split_text(text)

    if not chunks:
        raise ValueError(
            f"文档 {doc_id} 清洗后没有有效文本。"
        )

    print(
        f"文档 {doc_id} 已切分为 "
        f"{len(chunks)} 个文本块。"
    )

    # 删除这个来源以前写入的旧文本块
    delete_existing_document(doc_id)

    embeddings = model.encode(
        chunks,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    chunk_ids = [
        f"{doc_id}_chunk_{index:04d}"
        for index in range(len(chunks))
    ]

    base_metadata = normalize_metadata(metadata)

    metadata_list: list[Metadata] = [
        {
            **base_metadata,
            "source_id": doc_id,
            "chunk_index": index,
            "total_chunks": len(chunks),
        }
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata_list,
    )

    print(
        f"文档 {doc_id} 已成功写入 Chroma。"
    )

    return len(chunks)


# ============================================================
# 向量检索
# ============================================================

def search(
    query: str,
    topk: int = 3,
) -> list[str]:
    """
    根据问题检索最相关的文本块。

    参数：
    query：
        用户问题。

    topk：
        返回多少条最相关证据。
    """

    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    document_count = collection.count()

    if document_count == 0:
        print(
            "向量数据库当前为空，"
            "请先调用 add_document() 写入文档。"
        )
        return []

    # 防止请求数量超过数据库中的文档数量
    result_count = min(
        max(topk, 1),
        document_count,
    )

    query_embedding = model.encode(
        cleaned_query,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    result = collection.query(
        query_embeddings=[
            query_embedding,
        ],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result.get(
        "documents",
        [],
    )

    if not documents:
        return []

    if not documents[0]:
        return []

    return [
        document
        for document in documents[0]
        if document
    ]


# ============================================================
# 带详细信息的检索
# ============================================================

def search_with_details(
    query: str,
    topk: int = 3,
) -> list[dict]:
    """
    检索并返回文档、来源、块编号和距离。

    这个函数不是 main.py 必需的，
    但调试和后续展示引用来源时很有用。
    """

    cleaned_query = clean_text(query)

    if not cleaned_query:
        return []

    document_count = collection.count()

    if document_count == 0:
        return []

    result_count = min(
        max(topk, 1),
        document_count,
    )

    query_embedding = model.encode(
        cleaned_query,
        device=DEVICE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    result = collection.query(
        query_embeddings=[
            query_embedding,
        ],
        n_results=result_count,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = result.get(
        "documents",
        [[]],
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]],
    )[0]

    distances = result.get(
        "distances",
        [[]],
    )[0]

    detailed_results = []

    for index, document in enumerate(documents):
        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        detailed_results.append(
            {
                "document": document,
                "metadata": metadata,
                "source_id": metadata.get("source_id", "unknown"),
                "source_type": metadata.get("source_type", "unknown"),
                "pmid": metadata.get("pmid", ""),
                "nct_id": metadata.get("nct_id", ""),
                "guideline_id": metadata.get("guideline_id", ""),
                "title": metadata.get("title", ""),
                "organization": metadata.get("organization", ""),
                "journal": metadata.get("journal", ""),
                "year": metadata.get("year", ""),
                "topic": metadata.get("topic", ""),
                "status": metadata.get("status", ""),
                "conditions": metadata.get("conditions", ""),
                "interventions": metadata.get("interventions", ""),
                "url": metadata.get("url", ""),
                "chunk_index": metadata.get("chunk_index", -1),
                "distance": distance,
            }
        )

    return detailed_results


# ============================================================
# 单独运行 vector_db.py 时执行的测试
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始测试向量数据库")
    print("=" * 60)

    test_documents = [
        {
            "id": "esc_guideline_2018",
            "text": """
2018 ESC/ESH hypertension guideline:
Hypertension is usually a chronic condition.
Many patients need long-term blood pressure treatment
to reduce the risk of stroke, myocardial infarction,
heart failure, and kidney disease.
""",
        },
        {
            "id": "aha_guideline",
            "text": """
ACC/AHA hypertension guideline:
Patients should maintain continuous blood pressure control.
Long-term antihypertensive treatment can reduce
cardiovascular events in patients with persistent hypertension.
""",
        },
        {
            "id": "sprint_trial",
            "text": """
SPRINT clinical trial:
Intensive blood pressure treatment reduced the risk
of major cardiovascular events and all-cause mortality
in selected adults with elevated cardiovascular risk.
""",
        },
        {
            "id": "lifestyle_intervention",
            "text": """
Lifestyle interventions for hypertension include
reducing sodium intake, maintaining a healthy body weight,
regular physical activity, limiting alcohol intake,
and following a healthy dietary pattern.
""",
        },
        {
            "id": "lipid_treatment",
            "text": """
Clinical guidelines for high cholesterol recommend
lifestyle intervention as the foundation of treatment.
Statin therapy may be recommended according to
LDL cholesterol level and overall cardiovascular risk.
""",
        },
    ]

    for test_document in test_documents:
        add_document(
            text=test_document["text"],
            doc_id=test_document["id"],
        )

    print(
        f"\n数据库中当前共有 "
        f"{collection.count()} 个文本块。"
    )

    test_query = (
        "体检发现血脂偏高，"
        "生活方式干预和药物治疗有哪些证据？"
    )

    print("\n测试问题：")
    print(test_query)

    results = search_with_details(
        query=test_query,
        topk=3,
    )

    print("\n检索结果：")

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n--- 结果 {index} ---"
        )
        print(
            f"来源：{result['source_id']}"
        )
        print(
            f"文本块：{result['chunk_index']}"
        )
        print(
            f"距离：{result['distance']}"
        )
        print(
            f"内容：\n{result['document']}"
        )

    print("\n" + "=" * 60)
    print("向量数据库测试完成")
    print("=" * 60)
