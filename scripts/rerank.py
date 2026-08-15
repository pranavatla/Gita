import json
import re
import sys
from functools import lru_cache

import chromadb
import torch
from bedrock_client import converse_text, embed_texts, parse_json_text
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "gita_verses_v3_bedrock"
GENERATION_MODEL = "llama3.1:8b"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

RESULTS_PER_QUERY = 10
MAX_RERANK_CANDIDATES = 50
FINAL_RESULTS = 4
RRF_CONSTANT = 60

DOMAIN_EXPANSIONS = {
    "desire": [
        "lust compels harmful action",
        "lust covers knowledge and judgment",
        "greed and lust cause degradation",
        "sense control over harmful desire",
    ],
    "anger": [
        "anger causes clouded judgment",
        "anger leads to bewilderment",
    ],
    "greed": [
        "greed causes wrongful action",
        "greed and material desire cause degradation",
    ],
    "responsibility": [
        "own duty and prescribed duty",
        "work born of one's nature",
        "abandoning prescribed responsibility",
    ],
    "duty": [
        "own duty and prescribed duty",
        "work born of one's nature",
    ],
    "grief": [
        "grief mourning and lamentation",
        "mourning the living and dead",
    ],
    "death": [
        "death grief and mourning",
        "death and the eternal soul",
    ],
    "anxiety": [
        "anxiety and fear",
        "attachment to results",
        "equanimity in success and failure",
    ],
    "success": [
        "equanimity in success and failure",
        "happiness and distress loss and gain victory and defeat",
        "duty without attachment to results",
    ],
    "succeed": [
        "equanimity in success and failure",
        "happiness and distress loss and gain victory and defeat",
        "duty without attachment to results",
    ],
    "failure": [
        "equanimity in success and failure",
        "happiness and distress loss and gain victory and defeat",
        "duty without attachment to results",
    ],
    "fail": [
        "equanimity in success and failure",
        "happiness and distress loss and gain victory and defeat",
        "duty without attachment to results",
    ],
    "sleep": [
        "regulated eating sleeping work and recreation",
        "neither too much nor too little sleep",
        "balanced habits relieve material suffering",
    ],
}

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "literal_action": {"type": "string"},
        "source_concepts": {"type": "string"},
        "ethical_character": {"type": "string"},
        "consequence_or_remedy": {"type": "string"},
    },
    "required": [
        "literal_action",
        "source_concepts",
        "ethical_character",
        "consequence_or_remedy",
    ],
}

def generate_search_queries(question):
    system_prompt = (
        "Convert the user's scenario into exactly four distinct semantic "
        "retrieval phrases. literal_action preserves what actually "
        "happened. source_concepts translates modern wording into 3 to 6 "
        "concise philosophical or scriptural concepts and useful synonyms, "
        "such as lust, craving, attachment, anger, greed, prescribed duty, "
        "detachment, or sense control when directly supported. "
        "ethical_character describes only the conduct or principle directly "
        "supported by the scenario; do not invent character judgments such "
        "as laziness, arrogance, or bad character. consequence_or_remedy "
        "contains concise moral consequences and corrective principles "
        "supported by the scenario; do not invent legal, social, financial, "
        "or supernatural outcomes. Keep all four values short, distinct, "
        "and useful for retrieval. Do not mention scriptures, verse numbers, "
        "people, or facts absent from the scenario."
    )

    for attempt in range(1, 4):
        raw_content = converse_text(
            messages=[
                {
                    "role": "user",
                    "content": [{"text": question}],
                }
            ],
            system_prompt=(
                f"{system_prompt} Return only valid JSON matching this "
                f"schema: {json.dumps(QUERY_SCHEMA)}"
            ),
            max_tokens=400,
            temperature=0,
        )

        try:
            result = parse_json_text(raw_content)
        except json.JSONDecodeError:
            result = {}

        fields = [
            "literal_action",
            "source_concepts",
            "ethical_character",
            "consequence_or_remedy",
        ]
        queries = {
            field: result.get(field, "").strip()
            for field in fields
            if isinstance(result.get(field), str)
        }

        if (
            len(queries) == 4
            and all(queries.values())
            and len(set(queries.values())) == 4
        ):
            return queries

        print(
            f"Invalid search-query generation, retrying "
            f"({attempt}/3)"
        )

    raise RuntimeError("Could not generate valid search queries")


def expand_domain_concepts(question):
    words = set(
        re.findall(r"[a-z]+", question.lower())
    )

    # Convert simple plurals to singular forms.
    normalized_words = set(words)

    for word in words:
        if word.endswith("ies") and len(word) > 3:
            normalized_words.add(word[:-3] + "y")
        elif word.endswith("s") and len(word) > 3:
            normalized_words.add(word[:-1])

    expansions = []

    for term, term_queries in DOMAIN_EXPANSIONS.items():
        if term in normalized_words:
            expansions.extend(term_queries)

    return list(dict.fromkeys(expansions))


def retrieve_candidates(question, generated_queries):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    domain_queries = expand_domain_concepts(question)

    search_queries = [
        question,
        *generated_queries,
        *domain_queries,
    ]
    search_queries = list(dict.fromkeys(search_queries))

    embeddings = embed_texts(
        [
            f"search_query: {query}"
            for query in search_queries
        ],
    )

    candidates = {}

    for query, embedding in zip(
        search_queries,
        embeddings,
    ):
        results = collection.query(
            query_embeddings=[embedding],
            n_results=RESULTS_PER_QUERY,
        )

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
            candidate = candidates.setdefault(
                verse_id,
                {
                    "id": verse_id,
                    "document": document,
                    "metadata": metadata,
                    "rrf_score": 0.0,
                    "matches": [],
                },
            )

            candidate["rrf_score"] += 1 / (RRF_CONSTANT + rank)
            candidate["matches"].append(
                {
                    "query": query,
                    "rank": rank,
                    "distance": distance,
                }
            )

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate["rrf_score"],
            -len(candidate["matches"]),
            min(
                match["distance"]
                for match in candidate["matches"]
            ),
        ),
    )

    return search_queries, ordered_candidates[:MAX_RERANK_CANDIDATES]


@lru_cache(maxsize=1)
def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        local_files_only=True,
    )

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def rerank_globally(search_queries, candidates):
    tokenizer, model, device = load_reranker()

    candidate_texts = [
        (
            f"Translation: {candidate['document']} "
            "Themes: "
            f"{candidate['metadata'].get('themes', '')}"
        )
        for candidate in candidates
    ]

    pairs = [
        (query, candidate_text)
        for query in search_queries
        for candidate_text in candidate_texts
    ]

    inputs = tokenizer(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        scores = model(
            **inputs,
            return_dict=True,
        ).logits.view(
            len(search_queries),
            len(candidates),
        ).float().cpu()

    original_scores = scores[0]
    auxiliary_scores, auxiliary_indexes = scores[1:].max(
        dim=0,
    )
    combined_scores = (
        original_scores + auxiliary_scores
    ) / 2

    candidate_indexes = torch.argsort(
        combined_scores,
        descending=True,
    )[:FINAL_RESULTS].tolist()

    selected = []

    for candidate_index in candidate_indexes:
        best_query_index = (
            auxiliary_indexes[candidate_index].item() + 1
        )
        combined_score = combined_scores[
            candidate_index
        ].item()

        selected.append(
            {
                "candidate": candidates[candidate_index],
                "best_query": search_queries[best_query_index],
                "original_score": original_scores[
                    candidate_index
                ].item(),
                "auxiliary_score": auxiliary_scores[
                    candidate_index
                ].item(),
                "combined_score": combined_score,
                "normalized_score": torch.sigmoid(
                    torch.tensor(combined_score)
                ).item(),
            }
        )

    return selected, device


def print_results(
    question,
    search_queries,
    selected,
    device,
):
    print(f"\nOriginal question: {question}\n")
    print("Searches used:")
    for index, query in enumerate(search_queries, start=1):
        print(f"  {index}. {query}")

    print(f"\nReranker: {RERANKER_MODEL} on {device}")
    print(
        "These are related candidate principles. "
        "The scores rank passages; they are not confidence values "
        "and do not prove an exact verse match."
    )

    for rank, result in enumerate(selected, start=1):
        candidate = result["candidate"]
        metadata = candidate["metadata"]

        print("\n" + "=" * 70)
        print(f"Reranked position: {rank}")
        print(f"Reference: {candidate['id']}")
        print(f"Best auxiliary query: {result['best_query']}")
        print(
            "Original-question BGE score: "
            f"{result['original_score']:.4f}"
        )
        print(
            "Best auxiliary BGE score: "
            f"{result['auxiliary_score']:.4f}"
        )
        print(
            "Combined BGE score: "
            f"{result['combined_score']:.4f}"
        )

        print("\nOriginal Sanskrit:")
        print(metadata["sanskrit"])

        print("\nTransliteration:")
        print(metadata["transliteration"])

        print("\nEnglish:")
        print(candidate["document"])

    print("\n" + "=" * 70)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python scripts/rerank.py "your question"'
        )

    question = " ".join(sys.argv[1:]).strip()
    facet_queries = generate_search_queries(question)
    search_queries, candidates = retrieve_candidates(
        question,
        list(facet_queries.values()),
    )
    selected, device = rerank_globally(
        search_queries,
        candidates,
    )

    print_results(
        question,
        search_queries,
        selected,
        device,
    )


if __name__ == "__main__":
    main()
