import sys

import chromadb
import ollama


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "gita_verses_v1"
EMBEDDING_MODEL = "nomic-embed-text"
RESULT_COUNT = 5


def retrieve(question):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[f"search_query: {question}"],
    )

    return collection.query(
        query_embeddings=[response.embeddings[0]],
        n_results=RESULT_COUNT,
        include=["documents", "metadatas", "distances"],
    )


def print_results(results):
    for rank, (verse_id, english, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print("=" * 70)
        print(f"Rank: {rank}")
        print(f"Reference: {verse_id}")
        print(f"Cosine distance: {distance:.4f}")

        print("\nOriginal Sanskrit:")
        print(metadata["sanskrit"])

        print("\nTransliteration:")
        print(metadata["transliteration"])

        print("\nEnglish:")
        print(english)

    print("=" * 70)


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = input("Describe your situation: ").strip()

    if not question:
        raise SystemExit("Question cannot be empty.")

    print(f"\nSearching for: {question}\n")
    print_results(retrieve(question))


if __name__ == "__main__":
    main()
