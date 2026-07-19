import requests


def search_pubmed(keyword, max_results=5):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": keyword,
        "retmode": "json",
        "retmax": max_results
    }

    r = requests.get(url, params=params)

    ids = r.json()["esearchresult"]["idlist"]

    return ids



def fetch_pubmed_detail(ids):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db":"pubmed",
        "id":",".join(ids),
        "retmode":"xml"
    }

    r=requests.get(url,params=params)

    return r.text



if __name__=="__main__":

    ids=search_pubmed(
        "hypertension long term medication guideline"
    )

    print(ids)

    print(fetch_pubmed_detail(ids))