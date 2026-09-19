import json
import sys
import time

from bedrock_client import converse_text, parse_json_text

from rerank import (
    generate_search_queries,
    retrieve_candidates,
)


RERANK_LIMIT = 6

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_verse_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": RERANK_LIMIT,
            "maxItems": RERANK_LIMIT,
        },
        "evidence_strength": {
            "type": "string",
            "enum": ["strong", "partial", "none"],
        },
        "evidence_reason": {
            "type": "string",
        },
    },
    "required": [
        "ranked_verse_ids",
        "evidence_strength",
        "evidence_reason",
    ],
}

MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "direct_answer": {
            "type": "string",
        },
        "explanation": {
            "type": "string",
        },
    },
    "required": [
        "direct_answer",
        "explanation",
    ],
}


def build_candidate_context(candidates):
    passages = []

    for candidate in candidates:
        themes = candidate["metadata"].get("themes", "")
        passages.append(
            "\n".join(
                [
                    f"ID: {candidate['id']}",
                    f"English translation: {candidate['document']}",
                    f"Themes: {themes}",
                ]
            )
        )

    return "\n\n".join(passages)


def build_context(selected):
    return build_candidate_context(
        [result["candidate"] for result in selected]
    )


def normalize_verse_id(value):
    if isinstance(value, dict):
        value = (
            value.get("id")
            or value.get("verse_id")
            or value.get("verse")
            or value.get("reference")
        )

    if not isinstance(value, str):
        return ""

    normalized = value.strip().upper().replace(" ", "")
    normalized = normalized.replace("BG.", "BG")

    match = re.search(
        r"^(?:BG)?(\\d{1,2})[.:-](\\d{1,2})$",
        normalized,
    )

    if match:
        chapter, verse = match.groups()
        return f"BG{int(chapter)}.{int(verse)}"

    return normalized


def coerce_ranking_result(result, candidate_order):
    if isinstance(result, list):
        result = {"ranked_verse_ids": result}

    if not isinstance(result, dict):
        return {}

    ranked_ids = (
        result.get("ranked_verse_ids")
        or result.get("ranked_ids")
        or result.get("ranked_verses")
        or result.get("verse_ids")
        or result.get("selected_verse_ids")
        or result.get("ranking")
    )

    if isinstance(ranked_ids, dict):
        ranked_ids = list(ranked_ids.values())

    evidence_strength = (
        result.get("evidence_strength")
        or result.get("strength")
        or result.get("support")
        or "strong"
    )

    if isinstance(evidence_strength, str):
        evidence_strength = evidence_strength.strip().lower()

    evidence_reason = (
        result.get("evidence_reason")
        or result.get("reason")
        or result.get("rationale")
        or result.get("explanation")
        or "Selected by the evidence reranker."
    )

    allowed_ids = set(candidate_order)
    cleaned_ids = []

    if isinstance(ranked_ids, list):
        for verse_id in ranked_ids:
            cleaned_id = normalize_verse_id(verse_id)

            if (
                cleaned_id in allowed_ids
                and cleaned_id not in cleaned_ids
            ):
                cleaned_ids.append(cleaned_id)

    for verse_id in candidate_order:
        if len(cleaned_ids) >= RERANK_LIMIT:
            break

        if verse_id not in cleaned_ids:
            cleaned_ids.append(verse_id)

    return {
        "ranked_verse_ids": cleaned_ids,
        "evidence_strength": evidence_strength,
        "evidence_reason": evidence_reason,
    }


def validate_ranking(result, candidate_order):
    result = coerce_ranking_result(result, candidate_order)
    ranked_ids = result.get("ranked_verse_ids")
    evidence_strength = result.get("evidence_strength")
    evidence_reason = result.get("evidence_reason")

    if not isinstance(ranked_ids, list):
        return False

    if evidence_strength not in {"strong", "partial", "none"}:
        return False

    if not isinstance(evidence_reason, str):
        return False

    evidence_reason = evidence_reason.strip()

    if not evidence_reason:
        return False

    cleaned_ids = [
        verse_id
        for verse_id in ranked_ids
        if isinstance(verse_id, str) and verse_id
    ]

    if len(cleaned_ids) != RERANK_LIMIT:
        return False

    if len(set(cleaned_ids)) != RERANK_LIMIT:
        return False

    if any(verse_id not in candidate_order for verse_id in cleaned_ids):
        return False

    result["ranked_verse_ids"] = cleaned_ids
    result["evidence_reason"] = evidence_reason
    return result


def rerank_and_select(question, candidates):
    context = build_candidate_context(candidates)
    candidate_order = [candidate["id"] for candidate in candidates]

    system_prompt = (
        "You are the evidence reranker in a Bhagavad Gita RAG pipeline. "
        "Rank exactly six supplied passages by how directly their English "
        "translations provide evidence for the user's actual question. "
        "Judge doctrinal entailment and the requested relationship, not mere "
        "keyword overlap. A passage that explicitly connects harmful conduct "
        "to consequences, future birth, degradation, or divine action should "
        "outrank a passage that only mentions karma, action, death, duty, or "
        "rebirth separately. Penalize passages that discuss a nearby topic "
        "without answering the question. Also judge evidence_strength for "
        "the best passage: strong means the top passage can ground a direct "
        "answer, including a nuanced or caveated answer; partial means it is "
        "related but cannot support a grounded answer to the exact question; "
        "none means no supplied passage should be used as evidence. Do not "
        "answer the user and do not invent verses. Return six unique exact "
        "IDs, strongest first. The first ID is the passage selected to ground "
        "the final answer when evidence_strength is strong."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        f"User question:\n{question}\n\n"
                        f"Retrieved candidate passages:\n{context}"
                    )
                }
            ],
        },
    ]

    for attempt in range(1, 4):
        raw_content = converse_text(
            messages=messages,
            system_prompt=(
                f"{system_prompt} Return only valid JSON. Do not include "
                f"markdown or commentary. Example format: "
                f'{{"ranked_verse_ids":["BG16.19","BG16.20",'
                f'"BG14.15","BG16.9","BG9.21","BG1.40"],'
                f'"evidence_strength":"strong","evidence_reason":'
                f'"Short reason."}} Schema: {json.dumps(RERANK_SCHEMA)}'
            ),
            max_tokens=350,
            temperature=0,
        )

        try:
            result = parse_json_text(raw_content)
        except json.JSONDecodeError:
            result = {}

        validated = validate_ranking(result, candidate_order)

        if validated:
            result = validated
            ranked_ids = result["ranked_verse_ids"]
            candidates_by_id = {
                candidate["id"]: candidate
                for candidate in candidates
            }
            selected = [
                {
                    "candidate": candidates_by_id[verse_id],
                    "rank": rank,
                    "evidence_strength": result["evidence_strength"],
                    "evidence_reason": result["evidence_reason"],
                }
                for rank, verse_id in enumerate(ranked_ids, start=1)
            ]
            print(
                "Model evidence reranking trace:",
                json.dumps(
                    [
                        {
                            "id": item["candidate"]["id"],
                            "rank": item["rank"],
                            "evidence_strength": item[
                                "evidence_strength"
                            ],
                        }
                        for item in selected
                    ]
                ),
            )
            return selected

        print(
            f"Invalid Model evidence ranking, retrying "
            f"({attempt}/3)"
        )
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"text": raw_content}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "Return exactly six unique exact IDs from "
                                "the supplied passages, strongest first, "
                                "plus evidence_strength and evidence_reason."
                            )
                        }
                    ],
                },
            ]
        )

    raise RuntimeError("Could not generate a valid evidence ranking")


def validate_message(result):
    direct_answer = result.get("direct_answer")
    explanation = result.get("explanation")

    if not isinstance(direct_answer, str):
        return False

    if not isinstance(explanation, str):
        return False

    direct_answer = direct_answer.strip()
    explanation = explanation.strip()

    if not direct_answer or not explanation:
        return False

    combined = f"{direct_answer} {explanation}".strip()
    lowered = combined.lower()
    forbidden_starts = (
        "the bhagavad gita",
        "bhagavad gita",
        "the gita",
        "verse",
        "bg",
    )

    if lowered.startswith(forbidden_starts):
        return False

    if "bg" in lowered or "verse " in lowered:
        return False

    sentence_count = sum(
        combined.count(marker)
        for marker in (".", "?", "!")
    )

    if sentence_count > 3:
        return False

    if len(combined.split()) > 90:
        return False

    result["direct_answer"] = direct_answer
    result["explanation"] = explanation

    return True


def generate_passage_message(question, used_verse_ids, selected):
    selected_by_id = {
        result["candidate"]["id"]: result
        for result in selected
    }
    chosen_passages = [
        selected_by_id[verse_id]
        for verse_id in used_verse_ids
    ]
    context = build_context(chosen_passages)

    system_prompt = (
        "Write a short, human answer grounded only in the supplied Bhagavad "
        "Gita translation. The full answer formed by direct_answer plus "
        "explanation must be no more than three sentences and no more than "
        "90 words. direct_answer must start with the answer itself, never "
        "with boilerplate such as 'The Bhagavad Gita', 'The Gita', 'Verse', "
        "or a verse ID. Do not mention the verse ID inside the answer because "
        "the UI already shows it. explanation must connect the answer to the "
        "selected translation and separate explicit meaning from reasonable "
        "inference. For certainty, guarantee, proof, punishment, karma, or "
        "rebirth questions, state what the passage supports within the Gita's "
        "worldview; do not claim a mechanical or courtroom-style guarantee "
        "unless the supplied passage explicitly says that. If the passage is "
        "insufficient, say so plainly. Do not add generic advice unless asked. "
        "Do not invent facts, promises, punishments, quotations, or verse IDs."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        f"User question:\n{question}\n\n"
                        f"Selected source passages:\n{context}"
                    )
                }
            ],
        },
    ]

    for attempt in range(1, 4):
        raw_content = converse_text(
            messages=messages,
            system_prompt=(
                f"{system_prompt} Return only valid JSON matching this "
                f"schema: {json.dumps(MESSAGE_SCHEMA)}"
            ),
            max_tokens=300,
            temperature=0,
        )

        try:
            result = parse_json_text(raw_content)
        except json.JSONDecodeError:
            result = {}

        if validate_message(result):
            return result

        print(
            f"Invalid passage message, retrying "
            f"({attempt}/3)"
        )

        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [{"text": raw_content}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "text": (
                                "The previous response was invalid. Return "
                                "a concise direct_answer and explanation: "
                                "three sentences maximum, no verse IDs, no "
                                "boilerplate opening, grounded only in the "
                                "supplied translation."
                            )
                        }
                    ],
                },
            ]
        )

    raise RuntimeError(
        "Could not generate a valid passage message"
    )


def format_passage_message(passage_message):
    if "direct_answer" in passage_message:
        return (
            f"{passage_message['direct_answer']} "
            f"{passage_message['explanation']}"
        )

    # Backward compatibility for older callers and tests.
    return (
        f"{passage_message['principle']} "
        f"Practical application: {passage_message['application']}"
    )


def build_weak_evidence_message(evidence_strength, evidence_reason):
    if evidence_strength == "partial":
        return (
            "I found a related passage, but not enough support to give a "
            "fully grounded answer to this exact question. The closest verse "
            "is shown below for context, but the answer should be treated as "
            f"partial evidence: {evidence_reason}"
        )

    return (
        "I could not find a strong enough supporting passage in the retrieved "
        "evidence to answer this directly. The closest verse is shown below "
        f"only for context: {evidence_reason}"
    )


def print_answer(answer, selected):
    candidates_by_id = {
        result["candidate"]["id"]: result["candidate"]
        for result in selected
    }

    print("\nPractical message:\n")
    print(answer["message"])

    print("\nSupporting passages:")

    for verse_id in answer["used_verse_ids"]:
        candidate = candidates_by_id[verse_id]
        metadata = candidate["metadata"]

        print("\n" + "=" * 70)
        print(f"Reference: {verse_id}")

        print("\nOriginal Sanskrit:")
        print(metadata["sanskrit"])

        print("\nTransliteration:")
        print(metadata["transliteration"])

        print("\nEnglish:")
        print(candidate["document"])

    print("\n" + "=" * 70)


def answer_question(question):
    started_at = time.perf_counter()
    timings = {}

    step_started_at = time.perf_counter()
    facet_queries = generate_search_queries(question)
    elapsed = time.perf_counter() - step_started_at
    timings["query_generation"] = elapsed
    print(f"query generation: {elapsed:.2f}s")

    step_started_at = time.perf_counter()
    search_queries, candidates = retrieve_candidates(
        question,
        list(facet_queries.values()),
    )
    elapsed = time.perf_counter() - step_started_at
    timings["retrieval"] = elapsed
    print(f"retrieval: {elapsed:.2f}s")

    step_started_at = time.perf_counter()
    selected = rerank_and_select(question, candidates)
    elapsed = time.perf_counter() - step_started_at
    timings["reranking"] = elapsed
    timings["verse_selection"] = 0.0
    print(
        f"Model evidence reranking and selection: "
        f"{elapsed:.2f}s"
    )

    used_verse_ids = [selected[0]["candidate"]["id"]]
    evidence_strength = selected[0]["evidence_strength"]
    evidence_reason = selected[0]["evidence_reason"]

    step_started_at = time.perf_counter()
    if evidence_strength == "strong":
        passage_message = generate_passage_message(
            question,
            used_verse_ids,
            selected,
        )
        message = format_passage_message(passage_message)
    else:
        message = build_weak_evidence_message(
            evidence_strength,
            evidence_reason,
        )
    elapsed = time.perf_counter() - step_started_at
    timings["final_answer_generation"] = elapsed
    print(f"final answer generation: {elapsed:.2f}s")

    total_seconds = time.perf_counter() - started_at
    timings["total_pipeline"] = total_seconds

    answer = {
        "message": message,
        "evidence_strength": evidence_strength,
        "evidence_reason": evidence_reason,
        "used_verse_ids": used_verse_ids,
        "timings": timings,
        "total_seconds": total_seconds,
        "trace": {
            "queries": search_queries,
            "fused_candidates": [
                {
                    "id": candidate["id"],
                    "rrf_score": candidate["rrf_score"],
                    "sources": sorted(
                        {
                            match["retrieval_type"]
                            for match in candidate["matches"]
                        }
                    ),
                    "match_count": len(candidate["matches"]),
                }
                for candidate in candidates
            ],
            "reranked_candidates": [
                {
                    "id": result["candidate"]["id"],
                    "rank": result["rank"],
                }
                for result in selected
            ],
            "selected_verse_ids": used_verse_ids,
            "evidence_strength": evidence_strength,
            "evidence_reason": evidence_reason,
        },
    }

    print(f"total pipeline: {total_seconds:.2f}s")

    return answer, selected


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            'Usage: python scripts/answer.py "your question"'
        )

    question = " ".join(sys.argv[1:]).strip()

    answer, selected = answer_question(question)

    print_answer(answer, selected)


if __name__ == "__main__":
    main()
