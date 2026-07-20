from pubmed import search_pubmed, fetch_pubmed_detail
from vector_db import add_document
from rag import answer_question


def evidence_agent(question: str, pubmed_query: str) -> str | None:
    """
    question:
        用户原始问题，用于最终生成中文答案。

    pubmed_query:
        英文 PubMed 检索词，用于查找医学文献。
    """

    print("=" * 60)
    print("用户问题：")
    print(question)

    print("\n正在检索 PubMed...")
    print(f"英文检索词：{pubmed_query}")

    # 第一步：用英文关键词检索 PubMed
    ids = search_pubmed(
        keyword=pubmed_query,
        max_results=5,
    )

    print(f"PubMed 返回的 PMID：{ids}")

    # 必须先判断是否为空，不能直接使用 ids[0]
    if not ids:
        print("\nPubMed 没有返回文献。")
        print("请尝试更短、更通用的英文关键词。")
        return None

    print(f"\n成功找到 {len(ids)} 篇文献。")

    # 第二步：获取文献详情
    print("正在获取论文标题和摘要...")

    papers_xml = fetch_pubmed_detail(ids)

    if not papers_xml.strip():
        print("获取到的 PubMed 文献内容为空。")
        return None

    # 第三步：存入 Chroma
    print("正在把证据写入向量数据库...")

    document_id = f"pubmed_{ids[0]}"

    add_document(
        text=papers_xml,
        doc_id=document_id,
    )

    print(f"证据已写入数据库，文档 ID：{document_id}")

    # 第四步：RAG 检索并调用大模型回答
    print("\n正在生成带证据的回答...")

    result = answer_question(question)

    print("\n" + "=" * 60)
    print("最终回答：")
    print("=" * 60)
    print(result)

    return result


if __name__ == "__main__":
    # 给最终大模型看的中文问题
    question_cn = """
体检发现血脂偏高，
生活方式干预和药物治疗分别有哪些证据？
""".strip()

    # 给 PubMed 使用的英文检索词
    pubmed_query_en = (
        "dyslipidemia lifestyle intervention statin therapy "
        "cardiovascular risk"
    )

    evidence_agent(
        question=question_cn,
        pubmed_query=pubmed_query_en,
    )