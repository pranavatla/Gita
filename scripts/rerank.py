import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

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
    "guna": [
        "three modes goodness passion ignorance",
        "mode of goodness illuminating happiness knowledge",
        "mode of passion desires longings material actions",
        "mode of ignorance delusion indolence sleep",
    ],
    "spiritual": [
        "whatever you do eat give perform as an offering to God",
        "work as sacrifice for the Supreme remain free from bondage",
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
    # Generated facets often translate natural language into Sanskrit terms
    # such as sattva, rajas, tamas, or gunas. Apply deterministic corpus
    # vocabulary bridges to both the original question and those facets so
    # the matching English translations can enter the candidate set.
    domain_queries = expand_domain_concepts(
        " ".join([question, *generated_queries])
    )

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



def print_results(question, search_queries, candidates):
    print(f"\nOriginal question: {question}\n")
    print("Searches used:")

    for index, query in enumerate(search_queries, start=1):
        print(f"  {index}. {query}")

    print("\nRRF fused candidates:")

    for rank, candidate in enumerate(candidates, start=1):
        print(
            f"  {rank}. {candidate['id']} "
            f"(RRF {candidate['rrf_score']:.6f})"
        )


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
    print_results(question, search_queries, candidates)


if __name__ == "__main__":
    main()
