import json
from pathlib import Path

import chromadb
from bedrock_client import embed_texts


SOURCE_DIR = Path("data/sources/bhagavad-gita/slok")
ENRICHED_DIR = Path("data/enriched")
CHROMA_DIR = "data/chroma"

COLLECTION_NAME = "gita_verses_v3_bedrock"
BATCH_SIZE = 32


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

    themes = enrichment["themes"]

    records.append(
        {
            "id": data["_id"],
            "chapter": data["chapter"],
            "verse": data["verse"],
            "sanskrit": data["slok"].strip(),
            "transliteration": data["transliteration"].strip(),
            "english": english,
            "themes": themes,
        }
    )


records.sort(key=lambda item: (item["chapter"], item["verse"]))

assert len(records) == 700, (
    f"Expected 700 verse chunks, found {len(records)}"
)


client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


for start in range(0, len(records), BATCH_SIZE):
    batch = records[start : start + BATCH_SIZE]

    embedding_inputs = [
        (
            f"search_document: Bhagavad Gita {item['id']}. "
            f"Translation: {item['english']} "
            f"Themes: {', '.join(item['themes'])}"
        )
        for item in batch
    ]

    embeddings = embed_texts(embedding_inputs)

    collection.upsert(
        ids=[item["id"] for item in batch],
        documents=[item["english"] for item in batch],
        embeddings=embeddings,
        metadatas=[
            {
                "reference": item["id"],
                "chapter": item["chapter"],
                "verse": item["verse"],
                "sanskrit": item["sanskrit"],
                "transliteration": item["transliteration"],
                "translator": (
                    "A.C. Bhaktivedanta Swami Prabhupada"
                ),
                "themes": ", ".join(item["themes"]),
            }
            for item in batch
        ],
    )

    indexed = min(start + BATCH_SIZE, len(records))
    print(f"Indexed {indexed}/700")


print(
    f"Finished. Collection contains "
    f"{collection.count()} chunks."
)
