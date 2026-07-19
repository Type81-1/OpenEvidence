from pubmed import search_pubmed,fetch_pubmed_detail

from vector_db import add_document

from rag import answer_question



def evidence_agent(question):


    # Step 1:
    # 调用PubMed工具

    ids=search_pubmed(
        question
    )


    papers=fetch_pubmed_detail(ids)


    # Step 2:
    # 加入知识库

    add_document(
        papers,
        "pubmed_"+ids[0]
    )


    # Step3:
    # RAG回答

    result=answer_question(
        question
    )


    return result



if __name__=="__main__":


    q="""

体检发现血脂偏高，
生活方式干预和药物治疗分别有哪些证据？

"""


    print(
        evidence_agent(q)
    )