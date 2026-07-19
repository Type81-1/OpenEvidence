import chromadb
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "BAAI/bge-small-en"
)


client = chromadb.PersistentClient(
    path="./chroma_db"
)


collection = client.get_or_create_collection(
    name="medical_evidence"
)



def add_document(
        text,
        doc_id
):

    embedding = model.encode(text).tolist()


    collection.upsert(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )



def search(query, topk=3):

    emb=model.encode(query).tolist()


    result=collection.query(
        query_embeddings=[emb],
        n_results=topk
    )


    return result["documents"][0]



if __name__ == "__main__":

    # 这些是用于测试的模拟医学证据
    test_documents = [
        {
            "id": "esc_guideline_2018",
            "text": """
2018 ESC/ESH hypertension guideline:
Hypertension is usually a chronic condition.
Many patients need long-term blood pressure treatment
to reduce the risk of stroke, myocardial infarction,
heart failure, and kidney disease.
"""
        },
        {
            "id": "aha_guideline",
            "text": """
ACC/AHA hypertension guideline:
Patients should maintain continuous blood pressure control.
Long-term antihypertensive treatment can reduce
cardiovascular events in patients with persistent hypertension.
"""
        },
        {
            "id": "sprint_trial",
            "text": """
SPRINT clinical trial:
Intensive blood pressure treatment reduced the risk
of major cardiovascular events and all-cause mortality
in selected adults with elevated cardiovascular risk.
"""
        },
        {
            "id": "lifestyle_intervention",
            "text": """
Lifestyle interventions for hypertension include
reducing sodium intake, maintaining a healthy body weight,
regular physical activity, limiting alcohol intake,
and following a healthy dietary pattern.
"""
        },
        {
            "id": "lipid_treatment",
            "text": """
Clinical guidelines for high cholesterol recommend
lifestyle intervention as the foundation of treatment.
Statin therapy may be recommended according to
LDL cholesterol level and overall cardiovascular risk.
"""
        }
    ]

    # 把测试文档逐个存入向量数据库
    for document in test_documents:
        add_document(
            text=document["text"],
            doc_id=document["id"]
        )

    print(f"已写入 {len(test_documents)} 篇测试文档。")

    # 测试检索
    query = "What evidence supports lifestyle and drug treatment for high cholesterol?"

    print("\n测试问题：")
    print(query)

    results = search(query, topk=3)

    print("\n检索结果：")

    for index, result in enumerate(results, start=1):
        print(f"\n--- 结果 {index} ---")
        print(result)