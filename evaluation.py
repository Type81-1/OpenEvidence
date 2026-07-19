def evaluate(answer):

    metrics={

        "has_reference":
            "[1]" in answer,

        "has_uncertainty":
            "不足" in answer
            or "无法确定" in answer,

        "length":
            len(answer)

    }


    return metrics



if __name__=="__main__":

    result=evaluate(
        "依据[1]研究显示..."
    )

    print(result)