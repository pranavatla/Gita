import json
import sys

import chromadb
import ollama


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "gita_verses_v2"
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.1:8b"
RESULTS_PER_QUERY = 3
MAX_CANDIDATES = 8

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 4,
        }
    },
    "required": ["queries"],
}


def decompose_question(question):
    system_prompt = (
        "Decompose the user's scenario into 3 to 4 independent "
        "semantic retrieval phrases. Cover: (1) every explicit "
        "action and object, (2) the directly supported ethical motive "
        "or principle, and (3) the resulting harm or consequence. Do "
        "not combine all meanings into one phrase. Preserve money or "
        "property. When someone obtains another person's money through "
        "concealed facts, include wrongful gain or greed as its own "
        "retrieval concept. Use short concept phrases, not questions. "
        "Do not mention scripture or verses."
    )

    for attempt in range(1, 4):
        response = ollama.chat(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            format=QUERY_SCHEMA,
            options={"temperature": 0},
        )

        try:
            result = json.loads(response.message.content)
        except json.JSONDecodeError:
            result = {}

        queries = result.get("queries", [])

        if isinstance(queries, list):
            queries = [
                query.strip()
                for query in queries
                if isinstance(query, str) and query.strip()
            ]
            queries = list(dict.fromkeys(queries))

        if 3 <= len(queries) <= 4:
            return queries

        print(
            f"Invalid query decomposition, retrying "
            f"({attempt}/3)"
        )

    raise RuntimeError("Could not decompose the question")


def retrieve(question, concept_queries):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Keep the complete scenario and also search each smaller concept.
    search_queries = [question, *concept_queries]
    search_queries = list(dict.fromkeys(search_queries))

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[
            f"search_query: {query}"
            for query in search_queries
        ],
    )

    query_results = []
    candidates = {}

    for query, embedding in zip(
        search_queries,
        response.embeddings,
    ):
        results = collection.query(
            query_embeddings=[embedding],
            n_results=RESULTS_PER_QUERY,
        )

        rows = []

        for rank, (
            verse_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            row = {
                "id": verse_id,
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "query": query,
                "query_rank": rank,
            }
            rows.append(row)

            if verse_id not in candidates:
                candidates[verse_id] = {
                    "document": document,
                    "metadata": metadata,
                    "matches": [],
                }

            candidates[verse_id]["matches"].append(
                {
                    "query": query,
                    "rank": rank,
                    "distance": distance,
                }
            )

        query_results.append(rows)

    # Interleave ranks so every part of the scenario gets representation.
    ordered_ids = []

    for rank_index in range(RESULTS_PER_QUERY):
        for rows in query_results:
            if rank_index >= len(rows):
                continue

            verse_id = rows[rank_index]["id"]

            if verse_id not in ordered_ids:
                ordered_ids.append(verse_id)

            if len(ordered_ids) == MAX_CANDIDATES:
                return search_queries, ordered_ids, candidates

    return search_queries, ordered_ids, candidates


def print_results(question, search_queries, ordered_ids, candidates):
    print(f"\nOriginal question: {question}\n")
    print("Searches used:")

    for index, query in enumerate(search_queries, start=1):
        print(f"  {index}. {query}")

    for rank, verse_id in enumerate(ordered_ids, start=1):
        candidate = candidates[verse_id]
        metadata = candidate["metadata"]

        print("\n" + "=" * 70)
        print(f"Combined rank: {rank}")
        print(f"Reference: {verse_id}")

        print("Matched searches:")
        for match in candidate["matches"]:
            print(
                f"  - {match['query']} "
                f"(query rank {match['rank']}, "
                f"distance {match['distance']:.4f})"
            )

        print("\nOriginal Sanskrit:")
        print(metadata["sanskrit"])

        print("\nTransliteration:")
        print(metadata["transliteration"])

        print("\nEnglish:")
        print(candidate["document"])

        print("\nRetrieval themes:")
        print(metadata.get("themes", ""))

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python scripts/retrieve_multi.py "your question"'
        )

    question = " ".join(sys.argv[1:]).strip()
    concept_queries = decompose_question(question)

    search_queries, ordered_ids, candidates = retrieve(
        question,
        concept_queries,
    )

    print_results(
        question,
        search_queries,
        ordered_ids,
        candidates,
    )


if __name__ == "__main__":
    main()
