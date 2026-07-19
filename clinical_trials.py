import requests


def search_trials(keyword):


    url="https://clinicaltrials.gov/api/v2/studies"


    params={
        "query.term":keyword,
        "pageSize":5
    }


    r=requests.get(
        url,
        params=params
    )


    data=r.json()


    results=[]


    for item in data["studies"]:

        title=(
            item["protocolSection"]
            ["identificationModule"]
            ["briefTitle"]
        )

        results.append(title)


    return results



if __name__=="__main__":

    print(
        search_trials(
            "hypertension drug"
        )
    )