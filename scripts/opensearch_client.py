import os

import boto3
from opensearchpy import (
    AWSV4SignerAuth,
    OpenSearch,
    RequestsHttpConnection,
)

try:
    from .bedrock_client import embed_texts
except ImportError:
    from bedrock_client import embed_texts


AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
OPENSEARCH_HOST = os.environ.get(
    "OPENSEARCH_HOST",
    "58ci3d6wn7150so4ppv9.aoss.ap-south-1.on.aws",
)
OPENSEARCH_INDEX = os.environ.get(
    "OPENSEARCH_INDEX",
    "gita-verses",
)


def get_opensearch_client():
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, AWS_REGION, "aoss")

    return OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


def search_similar_verses(query_text, k=10):
    client = get_opensearch_client()
    query_vector = embed_texts([query_text])[0]

    response = client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": k,
            "_source": [
                "verse_id",
                "chapter",
                "verse_number",
                "english",
                "sanskrit",
                "transliteration",
            ],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            },
        },
    )

    hits = []

    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        hits.append(
            {
                "id": source["verse_id"],
                "document": source["english"],
                "metadata": {
                    "reference": source["verse_id"],
                    "chapter": source["chapter"],
                    "verse": source["verse_number"],
                    "sanskrit": source["sanskrit"],
                    "transliteration": source["transliteration"],
                },
                "score": hit["_score"],
            }
        )

    return hits