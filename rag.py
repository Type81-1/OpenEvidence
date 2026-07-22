from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from vector_db import search_with_details


# ============================================================
# 读取项目目录下的 .env
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    # override=True：
    # 优先使用当前项目 .env 中的配置，
    # 避免 Windows 环境变量里残留的旧 Key 覆盖它
    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )
else:
    print(f"提示：没有找到 .env 文件：{ENV_PATH}")


# ============================================================
# 配置
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini",
).strip()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "",
).strip()

LLM_THINKING = os.getenv(
    "LLM_THINKING",
    "",
).strip().lower()

# 可选模式：
#
# auto：
#   尝试调用 OpenAI，失败后自动显示证据摘录
#
# openai：
#   必须调用 OpenAI，失败后显示错误
#
# extractive：
#   完全不调用 OpenAI，只显示检索证据
LLM_MODE = os.getenv(
    "LLM_MODE",
    "auto",
).strip().lower()


# ============================================================
# 构造证据上下文
# ============================================================

def get_metadata(
    evidence: dict,
) -> dict:
    """
    兼容 search_with_details 返回的顶层字段和 metadata 字段。
    """

    metadata = evidence.get(
        "metadata",
        {},
    )

    if not isinstance(metadata, dict):
        metadata = {}

    return metadata


def build_evidence_header(
    index: int,
    evidence: dict,
) -> str:
    """
    生成证据标题，让 [1] 能对应到 PMID / URL。
    """

    metadata = get_metadata(evidence)
    source_type = metadata.get(
        "source_type",
        evidence.get("source_type", "unknown"),
    )

    if source_type == "pubmed":
        pmid = metadata.get(
            "pmid",
            evidence.get("pmid", ""),
        )

        return f"[{index}] PubMed PMID: {pmid}"

    if source_type == "clinical_trial":
        nct_id = metadata.get(
            "nct_id",
            evidence.get("nct_id", ""),
        )

        return f"[{index}] ClinicalTrials.gov NCT ID: {nct_id}"

    if source_type == "guideline":
        guideline_id = metadata.get(
            "guideline_id",
            evidence.get("guideline_id", ""),
        )

        return f"[{index}] Guideline ID: {guideline_id}"

    return f"[{index}] Source: {source_type}"


def build_evidence_metadata_lines(
    evidence: dict,
) -> list[str]:
    """
    将可追溯字段写入 prompt。
    """

    metadata = get_metadata(evidence)

    field_map = [
        ("title", "Title"),
        ("journal", "Journal"),
        ("year", "Year"),
        ("doi", "DOI"),
        ("nct_id", "NCT ID"),
        ("guideline_id", "Guideline ID"),
        ("organization", "Organization"),
        ("topic", "Topic"),
        ("status", "Status"),
        ("study_type", "Study type"),
        ("phase", "Phase"),
        ("conditions", "Conditions"),
        ("interventions", "Interventions"),
        ("enrollment", "Enrollment"),
        ("start_date", "Start date"),
        ("completion_date", "Completion date"),
        ("sponsor", "Sponsor"),
        ("location", "Location"),
        ("url", "URL"),
        ("publication_types", "Publication types"),
    ]

    lines: list[str] = []

    for field_name, label in field_map:
        value = metadata.get(
            field_name,
            evidence.get(field_name, ""),
        )

        if value:
            lines.append(f"{label}: {value}")

    return lines


def build_evidence_context(
    evidence_list: list[dict],
) -> str:
    """
    给检索到的证据添加编号和可追溯 metadata。

    例如：

    [1] PubMed PMID: 12345678
    第一条证据……

    [2] PubMed PMID: 23456789
    第二条证据……
    """

    numbered_evidence: list[str] = []

    for index, evidence in enumerate(
        evidence_list,
        start=1,
    ):
        document = evidence.get(
            "document",
            "",
        )

        cleaned_evidence = " ".join(
            str(document).split()
        )

        lines = [
            build_evidence_header(
                index=index,
                evidence=evidence,
            ),
            *build_evidence_metadata_lines(evidence),
            "Evidence text:",
            cleaned_evidence,
        ]

        numbered_evidence.append(
            "\n".join(lines)
        )

    return "\n\n".join(numbered_evidence)


def build_reference_summary(
    evidence_list: list[dict],
) -> str:
    """
    生成固定参考证据列表，避免最终答案只有 [1] 但没有 PMID。
    """

    references: list[str] = []

    for index, evidence in enumerate(
        evidence_list,
        start=1,
    ):
        metadata = get_metadata(evidence)
        source_type = metadata.get(
            "source_type",
            evidence.get("source_type", "unknown"),
        )

        if source_type == "pubmed":
            pmid = metadata.get("pmid", "")
            title = metadata.get("title", "")
            journal = metadata.get("journal", "")
            year = metadata.get("year", "")
            url = metadata.get("url", "")

            references.append(
                " | ".join(
                    part
                    for part in [
                        f"[{index}] PMID: {pmid}",
                        title,
                        journal,
                        year,
                        url,
                    ]
                    if part
                )
            )
        elif source_type == "clinical_trial":
            nct_id = metadata.get("nct_id", "")
            title = metadata.get("title", "")
            status = metadata.get("status", "")
            conditions = metadata.get("conditions", "")
            interventions = metadata.get("interventions", "")
            url = metadata.get("url", "")

            references.append(
                " | ".join(
                    part
                    for part in [
                        f"[{index}] NCT ID: {nct_id}",
                        title,
                        status,
                        conditions,
                        interventions,
                        url,
                    ]
                    if part
                )
            )
        elif source_type == "guideline":
            guideline_id = metadata.get("guideline_id", "")
            title = metadata.get("title", "")
            organization = metadata.get("organization", "")
            year = metadata.get("year", "")
            topic = metadata.get("topic", "")
            url = metadata.get("url", "")

            references.append(
                " | ".join(
                    part
                    for part in [
                        f"[{index}] Guideline ID: {guideline_id}",
                        title,
                        organization,
                        year,
                        topic,
                        url,
                    ]
                    if part
                )
            )
        else:
            references.append(
                f"[{index}] Source: {source_type}"
            )

    return "\n".join(references)


# ============================================================
# 构造提示词
# ============================================================

def build_prompt(
    question: str,
    evidence_list: list[dict],
) -> str:
    """
    构造发送给大模型的完整提示词。
    """

    context = build_evidence_context(
        evidence_list
    )

    reference_summary = build_reference_summary(
        evidence_list
    )

    prompt = f"""
你是一名谨慎的临床证据助手。

请只根据下面提供的医学证据回答问题。

用户问题：

{question}

检索到的医学证据：

{context}

证据编号与来源：

{reference_summary}

必须遵守以下要求：

1. 只能使用上面提供的证据，不允许编造医学事实。
2. 不允许编造作者、论文、期刊、指南、PMID、NCT ID、指南 ID、统计数字或治疗阈值。
3. 每个关键结论都必须标注对应证据编号，例如 [1]、[2]。
4. 必须区分证据类型：PubMed 文献、ClinicalTrials.gov 临床试验登记、指南/共识。
5. 指南/共识可以说明推荐方向，但不得把它当作针对个人的诊断或处方。
6. 如果证据不足、证据与问题不完全相关，必须明确说明，不要强行得出结论。
7. 区分“生活方式干预”和“药物治疗”。
8. 不提供针对个人的诊断、处方剂量、用药选择、停药建议或复诊间隔。
9. 使用中文回答。
10. 提醒用户，实际治疗需要由医生结合具体情况判断。
11. 如果引用 PubMed 证据，“参考证据”部分必须列出 PMID、标题、期刊、年份和 URL。
12. 如果引用 ClinicalTrials.gov 证据，“参考证据”部分必须列出 NCT ID、标题、状态和 URL。
13. 如果引用指南/共识证据，“参考证据”部分必须列出指南 ID、标题、机构、年份和 URL。

请严格按照下面的格式输出：

## 简短回答

用一到两段概括主要结论。

## 生活方式干预证据

根据证据说明饮食、运动、体重管理等干预。

## 药物治疗证据

根据证据说明药物治疗及其适用条件。

## 临床试验证据

根据 ClinicalTrials.gov 证据说明相关临床试验的状态、研究对象、干预措施和局限；如果没有相关临床试验证据，明确说明。

## 指南/共识证据

根据指南/共识证据说明推荐方向、适用范围和注意事项；如果没有相关指南/共识证据，明确说明。

## 证据局限

说明当前证据有哪些不足。

## 临床意义

说明这些证据对临床决策意味着什么，但不得替代医生判断。

## 参考证据

列出本回答实际引用的证据编号。PubMed 证据列出 PMID、标题、期刊、年份和 URL；ClinicalTrials.gov 证据列出 NCT ID、标题、状态和 URL；指南/共识证据列出指南 ID、标题、机构、年份和 URL。
""".strip()

    return prompt


# ============================================================
# API 不可用时的备用回答
# ============================================================

def build_fallback_answer(
    question: str,
    evidence_list: list[dict],
    reason: str,
) -> str:
    """
    OpenAI API 暂时不可用时，
    直接展示向量数据库检索出的证据。

    该模式不进行大模型总结。
    """

    output: list[str] = [
        "## 当前运行模式",
        "",
        "大模型当前不可用，系统已自动切换到“证据摘录模式”。",
        "",
        f"原因：{reason}",
        "",
        "## 用户问题",
        "",
        question,
        "",
        "## 检索到的证据",
        "",
    ]

    if not evidence_list:
        output.extend(
            [
                "当前没有检索到相关证据。",
                "",
                "因此无法根据现有知识库回答该问题。",
            ]
        )

        return "\n".join(output)

    for index, evidence in enumerate(
        evidence_list,
        start=1,
    ):
        document = evidence.get(
            "document",
            "",
        )

        cleaned_evidence = " ".join(
            str(document).split()
        )

        # 避免一条 PubMed XML 文本过长，
        # 导致整个终端都被占满
        max_length = 1800

        if len(cleaned_evidence) > max_length:
            cleaned_evidence = (
                cleaned_evidence[:max_length]
                + "……"
            )

        output.extend(
            [
                f"### {build_evidence_header(index, evidence)}",
                "",
                *build_evidence_metadata_lines(evidence),
                "",
                cleaned_evidence,
                "",
            ]
        )

    output.extend(
        [
            "## 说明",
            "",
            "以上是向量数据库检索到的原始证据摘录，"
            "尚未经过大模型归纳和总结。",
            "",
            "本系统仅用于课程项目和技术演示，"
            "不能替代医生诊断或治疗建议。",
        ]
    )

    return "\n".join(output)


# ============================================================
# 调用 OpenAI
# ============================================================

def call_openai(
    prompt: str,
) -> str:
    """
    调用大模型 API。

    如果配置了 OPENAI_BASE_URL，则按 OpenAI-compatible
    Chat Completions 接口调用，适配学校或第三方大模型平台。
    否则使用 OpenAI 官方 Responses API。
    """

    if not OPENAI_API_KEY:
        raise RuntimeError(
            ".env 中没有读取到 OPENAI_API_KEY。"
        )

    client_kwargs = {
        "api_key": OPENAI_API_KEY,
    }

    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL

    client = OpenAI(**client_kwargs)

    print(
        f"正在调用大模型：{OPENAI_MODEL}"
    )

    if OPENAI_BASE_URL:
        print(
            f"正在使用 OpenAI-compatible 接口：{OPENAI_BASE_URL}"
        )

        response = client.chat.completions.create(
            **{
                "model": OPENAI_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                **(
                    {
                        "extra_body": {
                            "thinking": {
                                "type": LLM_THINKING,
                            }
                        }
                    }
                    if LLM_THINKING
                    in {
                        "enabled",
                        "disabled",
                    }
                    else {}
                ),
            }
        )

        answer = response.choices[0].message.content
    else:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )

        answer = response.output_text

    if not answer:
        raise RuntimeError(
            "OpenAI 返回了空回答。"
        )

    return answer


# ============================================================
# RAG 主函数
# ============================================================

def answer_question(
    question: str,
    topk: int = 5,
) -> str:
    """
    完整流程：

    用户问题
        ↓
    Chroma 向量检索
        ↓
    构造证据提示词
        ↓
    OpenAI 生成答案
        ↓
    如果 API 失败，则显示证据摘录
    """

    question = question.strip()

    if not question:
        return "问题不能为空。"

    print("正在从向量数据库检索证据...")

    evidence_list = search_with_details(
        query=question,
        topk=topk,
    )

    if not evidence_list:
        return build_fallback_answer(
            question=question,
            evidence_list=[],
            reason="向量数据库没有检索到相关证据。",
        )

    print(
        f"已从向量数据库检索到 "
        f"{len(evidence_list)} 条证据。"
    )

    # 完全不调用大模型
    if LLM_MODE == "extractive":
        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason="当前 LLM_MODE 设置为 extractive。",
        )

    prompt = build_prompt(
        question=question,
        evidence_list=evidence_list,
    )

    try:
        return call_openai(prompt)

    except AuthenticationError as exc:
        reason = (
            "OpenAI API 身份认证失败。"
            "请检查 .env 中的 OPENAI_API_KEY 是否有效。"
        )

        if LLM_MODE == "openai":
            return f"调用失败：{reason}\n\n{exc}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )

    except RateLimitError as exc:
        error_text = str(exc)

        if "insufficient_quota" in error_text:
            reason = (
                "OpenAI API 项目没有可用额度"
                "（insufficient_quota）。"
            )
        else:
            reason = (
                "OpenAI API 当前达到调用频率限制。"
            )

        if LLM_MODE == "openai":
            return f"调用失败：{reason}\n\n{exc}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )

    except APIConnectionError as exc:
        reason = (
            "无法连接 OpenAI API，"
            "请检查网络、代理或防火墙。"
        )

        if LLM_MODE == "openai":
            return f"调用失败：{reason}\n\n{exc}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )

    except BadRequestError as exc:
        reason = (
            "OpenAI 请求参数错误，"
            "或者当前项目不能使用指定模型。"
        )

        if LLM_MODE == "openai":
            return f"调用失败：{reason}\n\n{exc}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )

    except RuntimeError as exc:
        reason = str(exc)

        if LLM_MODE == "openai":
            return f"调用失败：{reason}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )

    except Exception as exc:
        reason = (
            f"调用大模型时发生未知错误："
            f"{type(exc).__name__}: {exc}"
        )

        if LLM_MODE == "openai":
            return f"调用失败：{reason}"

        return build_fallback_answer(
            question=question,
            evidence_list=evidence_list,
            reason=reason,
        )


# ============================================================
# 单独运行 rag.py 时的测试
# ============================================================

if __name__ == "__main__":
    test_question = (
        "高血压患者为什么需要长期吃药？"
        "有哪些指南或研究依据？"
    )

    print("=" * 60)
    print("RAG 模块测试")
    print("=" * 60)

    print(f"当前模型：{OPENAI_MODEL}")
    print(f"当前模式：{LLM_MODE}")
    print(f".env 路径：{ENV_PATH}")
    print(
        "是否读取到 API Key："
        f"{bool(OPENAI_API_KEY)}"
    )

    result = answer_question(
        question=test_question,
        topk=5,
    )

    print("\n" + "=" * 60)
    print("回答结果")
    print("=" * 60)
    print(result)
