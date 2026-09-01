import json
from pathlib import Path

from opensearch_client import OPENSEARCH_INDEX, get_opensearch_client


VECTOR_CACHE_PATH = Path("data/vector_cache/gita_vectors.json")


def fetch_vectors():
    client = get_opensearch_client()

    response = client.search(
        index=OPENSEARCH_INDEX,
        body={
            "size": 1000,
            "_source": [
                "verse_id",
                "chapter",
                "verse_number",
                "english",
                "sanskrit",
                "transliteration",
                "themes",
                "embedding",
            ],
            "query": {
                "match_all": {},
            },
        },
        request_timeout=120,
    )

    records = []

    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        records.append(
            {
                "id": source["verse_id"],
                "chapter": source["chapter"],
                "verse_number": source["verse_number"],
                "english": source["english"],
                "sanskrit": source["sanskrit"],
                "transliteration": source["transliteration"],
                "themes": source.get("themes", []),
                "embedding": source["embedding"],
            }
        )

    records.sort(
        key=lambda record: (
            record["chapter"],
            record["verse_number"],
        )
    )

    return records


def main():
    records = fetch_vectors()

    if len(records) != 700:
        raise ValueError(f"Expected 700 vectors, found {len(records)}")

    VECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_CACHE_PATH.write_text(
        json.dumps(records, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Exported {len(records)} vectors to {VECTOR_CACHE_PATH}")


if __name__ == "__main__":
    main()
