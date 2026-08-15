import sys

import chromadb
import ollama


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "gita_verses_v1"
EMBEDDING_MODEL = "nomic-embed-text"
REWRITING_MODEL = "llama3.1:8b"
RESULT_COUNT = 8


def rewrite_question(question):
    response = ollama.chat(
        model=REWRITING_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Convert a real-world situation into a concise semantic "
                    "search query for retrieving relevant Bhagavad Gita verses. "
                    "Identify the underlying action, motive, ethical problem, "
                    "mental state, and likely consequence. "
                    "Do not answer the situation. "
                    "Do not cite or guess any verse number. "
                    "Return only one English sentence of 15 to 40 words."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response.message.content.strip()


def retrieve(search_query):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=[f"search_query: {search_query}"],
    )

    return collection.query(
        query_embeddings=[response.embeddings[0]],
        n_results=RESULT_COUNT,
        include=["documents", "metadatas", "distances"],
    )


def print_results(results):
    rows = zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )

    for rank, (verse_id, english, metadata, distance) in enumerate(
        rows,
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

    rewritten_query = rewrite_question(question)

    print(f"\nOriginal question:\n{question}")
    print(f"\nRewritten retrieval query:\n{rewritten_query}\n")

    results = retrieve(rewritten_query)
    print_results(results)


if __name__ == "__main__":
    main()
