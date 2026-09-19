import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

try:
    from .bedrock_client import embed_texts
except ImportError:
    from bedrock_client import embed_texts


VECTOR_CACHE_PATH = Path("data/vector_cache/gita_vectors.json")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=1)
def load_vector_cache():
    records = json.loads(VECTOR_CACHE_PATH.read_text(encoding="utf-8"))

    if len(records) != 700:
        raise ValueError(f"Expected 700 vectors, found {len(records)}")

    return records


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


@lru_cache(maxsize=1)
def load_lexical_index():
    records = load_vector_cache()
    tokenized_corpus = []

    for record in records:
        themes = record.get("themes", [])
        if isinstance(themes, str):
            themes = [themes]

        searchable_text = " ".join(
            [
                record["english"],
                *themes,
            ]
        )
        tokenized_corpus.append(tokenize(searchable_text))

    return BM25Okapi(tokenized_corpus)


def record_to_hit(record, score):
    return {
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

    hits = [
        record_to_hit(record, score)
        for score, record in scored_records
    ]

    if include_timings:
        return {
            "hits": hits,
            "timings": {
                "embedding": embedding_seconds,
                "local_search": local_search_seconds,
            },
        }

    return hits


def search_lexical_verses(query_text, k=10):
    records = load_vector_cache()
    bm25 = load_lexical_index()
    query_tokens = tokenize(query_text)

    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    ranked_indexes = sorted(
        range(len(records)),
        key=lambda index: scores[index],
        reverse=True,
    )[:k]

    return [
        record_to_hit(records[index], float(scores[index]))
        for index in ranked_indexes
        if scores[index] > 0
    ]
