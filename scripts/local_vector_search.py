import json
import math
import time
from functools import lru_cache
from pathlib import Path

try:
    from .bedrock_client import embed_texts
except ImportError:
    from bedrock_client import embed_texts


VECTOR_CACHE_PATH = Path("data/vector_cache/gita_vectors.json")


@lru_cache(maxsize=1)
def load_vector_cache():
    records = json.loads(VECTOR_CACHE_PATH.read_text(encoding="utf-8"))

    if len(records) != 700:
        raise ValueError(f"Expected 700 vectors, found {len(records)}")

    return records


def dot_product(left, right):
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left, right)
    )


def search_similar_verses(query_text, k=10, include_timings=False):
    records = load_vector_cache()

    embedding_started = time.perf_counter()
    query_vector = embed_texts([query_text])[0]
    embedding_seconds = time.perf_counter() - embedding_started

    search_started = time.perf_counter()
    scored_records = sorted(
        (
            (
                dot_product(query_vector, record["embedding"]),
                record,
            )
            for record in records
        ),
        key=lambda item: item[0],
        reverse=True,
    )[:k]
    local_search_seconds = time.perf_counter() - search_started

    hits = []

    for score, record in scored_records:
        hits.append(
            {
                "id": record["id"],
                "document": record["english"],
                "metadata": {
                    "reference": record["id"],
                    "chapter": record["chapter"],
                    "verse": record["verse_number"],
                    "sanskrit": record["sanskrit"],
                    "transliteration": record["transliteration"],
                    "themes": record.get("themes", []),
                },
                "score": score,
            }
        )

    if include_timings:
        return {
            "hits": hits,
            "timings": {
                "embedding": embedding_seconds,
                "local_search": local_search_seconds,
            },
        }

    return hits
