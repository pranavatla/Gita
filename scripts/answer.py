import json
import sys
import time

from bedrock_client import converse_text, parse_json_text

from rerank import (
    generate_search_queries,
    rerank_globally,
    retrieve_candidates,
)


SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "used_verse_ids": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "minItems": 1,
            "maxItems": 1,
        },
    },
    "required": [
        "used_verse_ids",
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


def build_context(selected):
    passages = []

    for result in selected:
        candidate = result["candidate"]

        passages.append(
            "\n".join(
                [
                    f"ID: {candidate['id']}",
                    f"English translation: {candidate['document']}",
                ]
            )
        )

    return "\n\n".join(passages)


def validate_selection(result, allowed_ids):
    used_ids = result.get("used_verse_ids")

    if not isinstance(used_ids, list):
        return False

    cleaned_ids = [
        verse_id.strip()
        for verse_id in used_ids
        if isinstance(verse_id, str) and verse_id.strip()
    ]
    cleaned_ids = list(dict.fromkeys(cleaned_ids))

    if len(cleaned_ids) != 1:
        return False

    if any(
        verse_id not in allowed_ids
        for verse_id in cleaned_ids
    ):
        return False

    result["used_verse_ids"] = cleaned_ids

    return True


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

    result["direct_answer"] = direct_answer
    result["explanation"] = explanation

    return True


def select_verse_ids(question, selected):
    context = build_context(selected)

    allowed_ids = {
        result["candidate"]["id"]
        for result in selected
    }

    system_prompt = (
        "Select exactly one supplied Bhagavad Gita passage: the passage "
        "that most directly supports the user's situation. This is passage "
        "selection only; do not answer, explain, judge, or rewrite the "
        "situation. Return exactly one exact ID from the supplied passages "
        "in used_verse_ids."
    )

    user_prompt = (
        f"User situation:\n{question}\n\n"
        f"Candidate passages:\n{context}"
    )

    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}],
        },
    ]

    for attempt in range(1, 4):
        raw_content = converse_text(
            messages=messages,
            system_prompt=(
                f"{system_prompt} Return only valid JSON matching this "
                f"schema: {json.dumps(SELECTION_SCHEMA)}"
            ),
            max_tokens=200,
            temperature=0,
        )

        try:
            result = parse_json_text(raw_content)
        except json.JSONDecodeError:
            result = {}

        if validate_selection(result, allowed_ids):
            return result["used_verse_ids"]

        print(
            f"Invalid verse selection, retrying "
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
                                "The previous selection was invalid. Return "
                                "exactly one exact ID from the supplied "
                                "passages only."
                            )
                        }
                    ],
                },
            ]
        )

    raise RuntimeError(
        "Could not generate a valid verse selection"
    )


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
        "Answer the user's actual question directly and naturally. Use only "
        "the supplied Bhagavad Gita translation as doctrinal evidence. "
        "direct_answer must begin with a clear answer to what the user asked, "
        "not a generic principle. explanation must briefly connect that answer "
        "to the supplied passage. Distinguish what the passage explicitly says "
        "from any reasonable inference. If the passage does not adequately "
        "answer the question, say that the available passage is insufficient; "
        "do not force a match. Do not add generic self-help or practical "
        "application unless the user explicitly asks for advice. Do not invent "
        "facts, promises, punishments, quotations, or verse IDs."
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
                                "a non-empty direct_answer and explanation "
                                "grounded only in the supplied translations."
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


def generate_grounded_answer(question, selected):
    used_verse_ids = select_verse_ids(
        question,
        selected,
    )
    passage_message = generate_passage_message(
        question,
        used_verse_ids,
        selected,
    )

    return {
        "message": format_passage_message(passage_message),
        "used_verse_ids": used_verse_ids,
    }


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
    selected, _ = rerank_globally(
        search_queries,
        candidates,
    )
    elapsed = time.perf_counter() - step_started_at
    timings["reranking"] = elapsed
    print(f"reranking: {elapsed:.2f}s")

    step_started_at = time.perf_counter()
    used_verse_ids = select_verse_ids(
        question,
        selected,
    )
    elapsed = time.perf_counter() - step_started_at
    timings["verse_selection"] = elapsed
    print(f"verse selection: {elapsed:.2f}s")

    step_started_at = time.perf_counter()
    passage_message = generate_passage_message(
        question,
        used_verse_ids,
        selected,
    )
    elapsed = time.perf_counter() - step_started_at
    timings["final_answer_generation"] = elapsed
    print(f"final answer generation: {elapsed:.2f}s")

    total_seconds = time.perf_counter() - started_at
    timings["total_pipeline"] = total_seconds

    answer = {
        "message": (
            format_passage_message(passage_message)
        ),
        "used_verse_ids": used_verse_ids,
        "timings": timings,
        "total_seconds": total_seconds,
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
