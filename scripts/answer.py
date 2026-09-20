import json
import re
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

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "direct_answer": {"type": "string"},
        "explanation": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "cited_verse_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "claim_support": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "verse_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                    "support": {"type": "string"},
                },
                "required": ["claim", "verse_ids", "support"],
            },
            "minItems": 1,
            "maxItems": 4,
        },
        "missing_context": {"type": "string"},
    },
    "required": [
        "direct_answer",
        "explanation",
        "confidence",
        "cited_verse_ids",
        "claim_support",
        "missing_context",
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
        "keyword overlap. When a question names multiple states, concepts, "
        "or forces, prefer a passage that explains their shared framework or "
        "interaction over one that explains only a single named element. A "
        "passage that explicitly connects harmful conduct "
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


def selected_passages_by_id(selected):
    return {
        result["candidate"]["id"]: result
        for result in selected
    }


def collapse_words(value, max_words):
    words = str(value).split()
    if len(words) <= max_words:
        return " ".join(words)

    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def normalize_verse_id_list(value):
    if isinstance(value, str) or isinstance(value, dict):
        value = [value]

    if not isinstance(value, list):
        return []

    cleaned = []
    for item in value:
        normalized_id = normalize_verse_id(item)

        if normalized_id and normalized_id not in cleaned:
            cleaned.append(normalized_id)

    return cleaned


def build_default_claim_support(cited_verse_ids, selected):
    selected_by_id = selected_passages_by_id(selected)
    claims = []

    for verse_id in cited_verse_ids:
        candidate = selected_by_id[verse_id]["candidate"]
        claims.append(
            {
                "claim": (
                    "The reflection stays within the selected passage's "
                    "explicit teaching."
                ),
                "verse_ids": [verse_id],
                "support": collapse_words(candidate["document"], 28),
            }
        )

    return claims


def repair_grounded_answer(result, allowed_verse_ids, selected):
    if not isinstance(result, dict):
        result = {}

    allowed_verse_ids = list(allowed_verse_ids)
    allowed_set = set(allowed_verse_ids)

    direct_answer = result.get("direct_answer")
    explanation = result.get("explanation")
    message = {
        "direct_answer": direct_answer,
        "explanation": explanation,
    }

    if validate_message(message):
        direct_answer = message["direct_answer"]
        explanation = message["explanation"]
    else:
        primary = selected_passages_by_id(selected)[allowed_verse_ids[0]]
        translation = collapse_words(primary["candidate"]["document"], 28)
        direct_answer = (
            "Stay close to what the selected passage clearly supports."
        )
        explanation = f"It points to this teaching: {translation}"

    confidence = result.get("confidence")
    if isinstance(confidence, str):
        confidence = confidence.strip().lower()

    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    cited_verse_ids = [
        verse_id
        for verse_id in normalize_verse_id_list(
            result.get("cited_verse_ids")
            or result.get("citations")
            or result.get("verse_ids")
        )
        if verse_id in allowed_set
    ]

    if not cited_verse_ids:
        cited_verse_ids = allowed_verse_ids[:1]

    cited_verse_ids = cited_verse_ids[:3]

    claim_support = result.get("claim_support")
    cleaned_claims = []
    if isinstance(claim_support, list):
        for item in claim_support:
            if not isinstance(item, dict):
                continue

            claim = item.get("claim")
            support = item.get("support")
            verse_ids = [
                verse_id
                for verse_id in normalize_verse_id_list(
                    item.get("verse_ids") or item.get("verse_id")
                )
                if verse_id in allowed_set
            ]

            if (
                isinstance(claim, str)
                and claim.strip()
                and isinstance(support, str)
                and support.strip()
                and verse_ids
            ):
                cleaned_claims.append(
                    {
                        "claim": claim.strip(),
                        "verse_ids": verse_ids[:3],
                        "support": support.strip(),
                    }
                )

    used_in_claims = {
        verse_id
        for item in cleaned_claims
        for verse_id in item["verse_ids"]
    }
    missing_claim_verse_ids = [
        verse_id
        for verse_id in cited_verse_ids
        if verse_id not in used_in_claims
    ]

    if missing_claim_verse_ids:
        cleaned_claims.extend(
            build_default_claim_support(
                missing_claim_verse_ids,
                selected,
            )
        )

    if not cleaned_claims:
        cleaned_claims = build_default_claim_support(
            cited_verse_ids,
            selected,
        )

    missing_context = result.get("missing_context")
    if not isinstance(missing_context, str):
        missing_context = ""

    return {
        "direct_answer": direct_answer,
        "explanation": explanation,
        "confidence": confidence,
        "cited_verse_ids": cited_verse_ids,
        "claim_support": cleaned_claims[:4],
        "missing_context": missing_context.strip(),
    }


def validate_grounded_answer(result, allowed_verse_ids):
    if not isinstance(result, dict):
        return False

    allowed_verse_ids = set(allowed_verse_ids)
    direct_answer = result.get("direct_answer")
    explanation = result.get("explanation")
    confidence = result.get("confidence")
    cited_verse_ids = result.get("cited_verse_ids")
    claim_support = result.get("claim_support")
    missing_context = result.get("missing_context")

    if confidence not in {"high", "medium", "low"}:
        return False

    if not isinstance(missing_context, str):
        return False

    if not validate_message(
        {
            "direct_answer": direct_answer,
            "explanation": explanation,
        }
    ):
        return False

    if not isinstance(cited_verse_ids, list):
        return False

    cleaned_citations = []
    for verse_id in cited_verse_ids:
        normalized_id = normalize_verse_id(verse_id)

        if normalized_id not in allowed_verse_ids:
            return False

        if normalized_id not in cleaned_citations:
            cleaned_citations.append(normalized_id)

    if not cleaned_citations or len(cleaned_citations) > 3:
        return False

    if not isinstance(claim_support, list):
        return False

    cleaned_claims = []
    for item in claim_support:
        if not isinstance(item, dict):
            return False

        claim = item.get("claim")
        support = item.get("support")
        verse_ids = item.get("verse_ids")

        if not isinstance(claim, str) or not claim.strip():
            return False

        if not isinstance(support, str) or not support.strip():
            return False

        if not isinstance(verse_ids, list) or not verse_ids:
            return False

        cleaned_verse_ids = []
        for verse_id in verse_ids:
            normalized_id = normalize_verse_id(verse_id)

            if normalized_id not in allowed_verse_ids:
                return False

            if normalized_id not in cleaned_verse_ids:
                cleaned_verse_ids.append(normalized_id)

        if not cleaned_verse_ids:
            return False

        cleaned_claims.append(
            {
                "claim": claim.strip(),
                "verse_ids": cleaned_verse_ids,
                "support": support.strip(),
            }
        )

    if not cleaned_claims or len(cleaned_claims) > 4:
        return False

    used_in_claims = {
        verse_id
        for item in cleaned_claims
        for verse_id in item["verse_ids"]
    }

    if not set(cleaned_citations).issubset(used_in_claims):
        return False

    result["direct_answer"] = direct_answer.strip()
    result["explanation"] = explanation.strip()
    result["confidence"] = confidence
    result["cited_verse_ids"] = cleaned_citations
    result["claim_support"] = cleaned_claims
    result["missing_context"] = missing_context.strip()
    return True


def build_citations(verse_ids, selected):
    selected_by_id = selected_passages_by_id(selected)
    citations = []

    for verse_id in verse_ids:
        candidate = selected_by_id[verse_id]["candidate"]
        metadata = candidate["metadata"]
        citations.append(
            {
                "id": verse_id,
                "english": candidate["document"],
                "sanskrit": metadata["sanskrit"],
                "transliteration": metadata["transliteration"],
            }
        )

    return citations


def build_grounding_summary(
    evidence_strength,
    evidence_reason,
    cited_verse_ids,
    selected,
    claim_support=None,
    confidence=None,
    missing_context="",
):
    if confidence is None:
        confidence = {
            "strong": "high",
            "partial": "medium",
            "none": "low",
        }.get(evidence_strength, "low")

    return {
        "confidence": confidence,
        "evidence_strength": evidence_strength,
        "evidence_reason": evidence_reason,
        "citations": build_citations(cited_verse_ids, selected),
        "claim_support": claim_support or [],
        "missing_context": missing_context,
    }


def generate_passage_message(question, used_verse_ids, selected):
    selected_by_id = selected_passages_by_id(selected)
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
        "inference. cited_verse_ids must contain only supplied IDs. "
        "claim_support must list each major answer claim, the supplied verse "
        "ID that supports it, and a short paraphrase of the evidence. For "
        "certainty, guarantee, proof, punishment, karma, or "
        "rebirth questions, state what the passage supports within the Gita's "
        "worldview; do not claim a mechanical or courtroom-style guarantee "
        "unless the supplied passage explicitly says that. If the passage is "
        "insufficient, say so plainly and put the limitation in "
        "missing_context. Do not add generic advice unless asked. "
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
                f"schema: {json.dumps(GROUNDING_SCHEMA)}"
            ),
            max_tokens=650,
            temperature=0,
        )

        try:
            result = parse_json_text(raw_content)
        except json.JSONDecodeError:
            result = {}

        if validate_grounded_answer(result, used_verse_ids):
            return result

        repaired_result = repair_grounded_answer(
            result,
            used_verse_ids,
            selected,
        )
        if validate_grounded_answer(repaired_result, used_verse_ids):
            return repaired_result

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
                                "boilerplate opening, cited_verse_ids using "
                                "only supplied IDs, and claim_support "
                                "grounded only in the supplied translation."
                            )
                        }
                    ],
                },
            ]
        )

    fallback_result = repair_grounded_answer(
        {},
        used_verse_ids,
        selected,
    )
    if validate_grounded_answer(fallback_result, used_verse_ids):
        return fallback_result

    raise RuntimeError("Could not generate a valid passage message")


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

    evidence_strength = selected[0]["evidence_strength"]
    evidence_reason = selected[0]["evidence_reason"]
    used_verse_ids = [
        result["candidate"]["id"]
        for result in selected[:2]
    ] if evidence_strength == "strong" else [
        selected[0]["candidate"]["id"]
    ]

    step_started_at = time.perf_counter()
    if evidence_strength == "strong":
        passage_message = generate_passage_message(
            question,
            used_verse_ids,
            selected,
        )
        message = format_passage_message(passage_message)
        grounding = build_grounding_summary(
            evidence_strength=evidence_strength,
            evidence_reason=evidence_reason,
            cited_verse_ids=passage_message["cited_verse_ids"],
            selected=selected,
            claim_support=passage_message["claim_support"],
            confidence=passage_message["confidence"],
            missing_context=passage_message["missing_context"],
        )
    else:
        message = build_weak_evidence_message(
            evidence_strength,
            evidence_reason,
        )
        grounding = build_grounding_summary(
            evidence_strength=evidence_strength,
            evidence_reason=evidence_reason,
            cited_verse_ids=used_verse_ids,
            selected=selected,
            missing_context=evidence_reason,
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
        "grounding": grounding,
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
