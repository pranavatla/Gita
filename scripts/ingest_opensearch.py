import json
from pathlib import Path

from opensearchpy.helpers import bulk

from bedrock_client import embed_texts
from opensearch_client import get_opensearch_client, OPENSEARCH_INDEX


SOURCE_DIR = Path("data/sources/bhagavad-gita/slok")
ENRICHED_DIR = Path("data/enriched")
BATCH_SIZE = 32


def load_records():
    records = []

    for path in SOURCE_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        english = data.get("prabhu", {}).get("et", "").strip()

        if not english or "did not comment" in english:
            continue

        enrichment_path = ENRICHED_DIR / f"{data['_id']}.json"

        if not enrichment_path.exists():
            raise FileNotFoundError(
                f"Missing enrichment: {enrichment_path}"
            )

        enrichment = json.loads(
            enrichment_path.read_text(encoding="utf-8")
        )

        records.append(
            {
                "id": data["_id"],
                "chapter": data["chapter"],
                "verse_number": data["verse"],
                "sanskrit": data["slok"].strip(),
                "transliteration": data["transliteration"].strip(),
                "english": english,
                "themes": enrichment["themes"],
            }
        )

    records.sort(key=lambda item: (item["chapter"], item["verse_number"]))

    if len(records) != 700:
        raise ValueError(
            f"Expected 700 verse chunks, found {len(records)}"
        )

    return records


def build_embedding_inputs(batch):
    return [
        (
            f"search_document: Bhagavad Gita {item['id']}. "
            f"Translation: {item['english']} "
            f"Themes: {', '.join(item['themes'])}"
        )
        for item in batch
    ]


def build_actions(batch, embeddings):
    actions = []

    for item, embedding in zip(batch, embeddings):
        actions.append(
            {
                "_op_type": "index",
                "_index": OPENSEARCH_INDEX,
                "_id": item["id"],
                "_source": {
                    "verse_id": item["id"],
                    "chapter": item["chapter"],
                    "verse_number": item["verse_number"],
                    "english": item["english"],
                    "sanskrit": item["sanskrit"],
                    "transliteration": item["transliteration"],
                    "themes": item["themes"],
                    "embedding": embedding,
                },
            }
        )

    return actions


def main():
    client = get_opensearch_client()
    records = load_records()

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        embedding_inputs = build_embedding_inputs(batch)
        embeddings = embed_texts(embedding_inputs)
        actions = build_actions(batch, embeddings)

        success_count, _ = bulk(
            client,
            actions,
            request_timeout=120,
        )

        indexed = min(start + BATCH_SIZE, len(records))
        print(
            f"Indexed {indexed}/700 "
            f"(wrote {success_count} docs in this batch)"
        )

    print("Finished indexing all verses into OpenSearch.")


if __name__ == "__main__":
    main()