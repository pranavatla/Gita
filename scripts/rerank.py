import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    from .local_vector_search import (
        search_lexical_verses,
        search_similar_verses,
    )
except ImportError:
    from local_vector_search import (
        search_lexical_verses,
        search_similar_verses,
    )

try:
    from .bedrock_client import converse_text, parse_json_text
except ImportError:
    from bedrock_client import converse_text, parse_json_text

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

RESULTS_PER_QUERY = 10
MAX_RERANK_CANDIDATES = 20
FINAL_RESULTS = 6
RRF_CONSTANT = 60
NUM_FACET_QUERIES = 4

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
    "karma": [
        "fruits and moral consequences of action",
        "results of righteous and unrighteous action after death",
    ],
    "rebirth": [
        "future birth shaped by conduct and consciousness",
        "repeated adverse or demoniac births",
        "degradation into lower conditions of existence",
    ],
    "wrong": [
        "destructive demoniac conduct and its consequences",
        "cruel envious harmful action leading to degradation",
    ],
    "wrongdoing": [
        "destructive demoniac conduct and its consequences",
        "cruel envious harmful action leading to degradation",
    ],
    "punishment": [
        "adverse consequences and lower future births",
        "degradation caused by destructive conduct",
    ],
    "punished": [
        "adverse consequences and lower future births",
        "degradation caused by destructive conduct",
    ],
}

FACET_FIELDS = [
    "literal_action",
    "source_concepts",
    "consequence_or_remedy",
    "ideal_evidence",
]

QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "literal_action": {"type": "string"},
        "source_concepts": {"type": "string"},
        "consequence_or_remedy": {"type": "string"},
        "ideal_evidence": {"type": "string"},
    },
    "required": FACET_FIELDS,
}

def generate_search_queries(question):
    system_prompt = (
        "Transform the user's natural question into exactly four distinct "
        "retrieval phrases for a Bhagavad Gita corpus. literal_action "
        "preserves what the user is asking or describing. source_concepts "
        "translates modern wording into 3 to 6 relevant Gita concepts and "
        "synonyms. consequence_or_remedy describes the doctrinal consequence "
        "or corrective principle being sought. For theological questions, "
        "include relevant concepts such as karma, fruits of action, after "
        "death, rebirth, adverse birth, degradation, demoniac conduct, or "
        "liberation when supported by the question. ideal_evidence is a short "
        "HyDE-style description of what an ideal answering passage would say, "
        "using likely corpus vocabulary without inventing a quotation, verse "
        "number, or doctrinal conclusion. Keep the four values distinct and "
        "optimized for retrieval. Do not mention specific verses."
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

        queries = {
            field: result.get(field, "").strip()
            for field in FACET_FIELDS
            if isinstance(result.get(field), str)
        }

        if (
            len(queries) == NUM_FACET_QUERIES
            and all(queries.values())
            and len(set(queries.values())) == NUM_FACET_QUERIES
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
        elif word.endswith("ing") and len(word) > 5:
            normalized_words.add(word[:-3])
        elif word.endswith("ed") and len(word) > 4:
            normalized_words.add(word[:-2])
        elif word.endswith("s") and len(word) > 3:
            normalized_words.add(word[:-1])

    expansions = []

    for term, term_queries in DOMAIN_EXPANSIONS.items():
        if term in normalized_words:
            expansions.extend(term_queries)

    return list(dict.fromkeys(expansions))


def retrieve_candidates(question, generated_queries):
    domain_queries = expand_domain_concepts(question)

    search_queries = [
        question,
        *generated_queries,
        *domain_queries,
    ]
    search_queries = list(dict.fromkeys(search_queries))

    candidates = {}

    search_requests = [
        (query, retrieval_type)
        for query in search_queries
        for retrieval_type in ("semantic", "lexical")
    ]

    def run_search(request):
        query, retrieval_type = request

        if retrieval_type == "semantic":
            results = search_similar_verses(
                query,
                k=RESULTS_PER_QUERY,
            )
        else:
            results = search_lexical_verses(
                query,
                k=RESULTS_PER_QUERY,
            )

        return query, retrieval_type, results

    with ThreadPoolExecutor(max_workers=8) as executor:
        search_results = list(
            executor.map(run_search, search_requests)
        )

    for query, retrieval_type, results in search_results:
        for result_index, item in enumerate(results, start=1):
            verse_id = item["id"]

            candidate = candidates.setdefault(
                verse_id,
                {
                    "id": verse_id,
                    "document": item["document"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "matches": [],
                },
            )

            candidate["rrf_score"] += 1 / (
                RRF_CONSTANT + result_index
            )
            candidate["matches"].append(
                {
                    "query": query,
                    "retrieval_type": retrieval_type,
                    "rank": result_index,
                    "distance": -item["score"],
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

    retained_candidates = ordered_candidates[:MAX_RERANK_CANDIDATES]

    print(
        "retrieval trace:",
        json.dumps(
            {
                "queries": search_queries,
                "fused_candidates": [
                    {
                        "id": candidate["id"],
                        "rrf_score": round(candidate["rrf_score"], 6),
                        "sources": sorted(
                            {
                                match["retrieval_type"]
                                for match in candidate["matches"]
                            }
                        ),
                        "match_count": len(candidate["matches"]),
                    }
                    for candidate in retained_candidates
                ],
            },
            ensure_ascii=False,
        ),
    )

    return search_queries, retained_candidates


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
    original_query = search_queries[0]
    intent_facets = search_queries[1 : 1 + NUM_FACET_QUERIES]
    rerank_query = (
        f"{original_query} "
        f"Key intent facets: {'; '.join(intent_facets)}"
    )

    candidate_texts = [
        (
            f"Translation: {candidate['document']} "
            "Themes: "
            f"{candidate['metadata'].get('themes', '')}"
        )
        for candidate in candidates
    ]

    inputs = tokenizer(
        [rerank_query] * len(candidate_texts),
        candidate_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        scores = model(
            **inputs,
            return_dict=True,
        ).logits.view(-1).float().cpu()

    candidate_indexes = torch.argsort(
        scores,
        descending=True,
    )[:FINAL_RESULTS].tolist()

    selected = []

    for candidate_index in candidate_indexes:
        score = scores[candidate_index].item()

        selected.append(
            {
                "candidate": candidates[candidate_index],
                "best_query": original_query,
                "original_score": score,
                "auxiliary_score": score,
                "combined_score": score,
                "normalized_score": torch.sigmoid(
                    torch.tensor(score)
                ).item(),
            }
        )

    print(
        "reranking trace:",
        json.dumps(
            [
                {
                    "id": result["candidate"]["id"],
                    "score": round(result["combined_score"], 6),
                }
                for result in selected
            ]
        ),
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
