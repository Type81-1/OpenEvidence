import os
from dotenv import load_dotenv
from openai import OpenAI

from vector_db import search

load_dotenv()

client=OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)



def answer_question(question):


    evidence = search(question)


    context="\n\n".join(
        evidence
    )


    prompt=f"""

你是一名临床证据助手。

请根据下面医学证据回答问题。

要求：

1. 不允许编造医学事实
2. 如果证据不足，请明确说明
3. 每个关键结论必须引用来源编号

问题：

{question}


证据：

{context}


输出格式：

## 简短回答

xxx


## 证据依据

- [1] xxx
- [2] xxx


## 临床意义

xxx

"""


    response=client.chat.completions.create(

        model="gpt-5-mini",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]

    )


    return response.choices[0].message.content



if __name__=="__main__":

    print(
        answer_question(
            "高血压患者为什么需要长期吃药？"
        )
    )