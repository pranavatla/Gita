#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from answer import answer_question


DEFAULT_CASES = (
    Path(__file__).resolve().parent.parent
    / "eval"
    / "rag_eval_cases.json"
)


def load_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_grounding(answer, acceptable_ids):
    grounding = answer.get("grounding")

    if not isinstance(grounding, dict):
        return False, "missing grounding object"

    citations = grounding.get("citations")
    claim_support = grounding.get("claim_support")
    confidence = grounding.get("confidence")

    if confidence not in {"high", "medium", "low"}:
        return False, "invalid confidence"

    if not isinstance(citations, list) or not citations:
        return False, "missing citations"

    citation_ids = {
        citation.get("id")
        for citation in citations
        if isinstance(citation, dict)
    }

    if not citation_ids:
        return False, "citation IDs missing"

    if answer["evidence_strength"] == "strong":
        if not citation_ids.intersection(acceptable_ids):
            return False, "citations do not include an acceptable verse"

        if not isinstance(claim_support, list) or not claim_support:
            return False, "claim support missing"

        for item in claim_support:
            if not isinstance(item, dict):
                return False, "invalid claim support item"

            if not item.get("claim") or not item.get("support"):
                return False, "claim support lacks claim or support"

            verse_ids = item.get("verse_ids")
            if not isinstance(verse_ids, list) or not verse_ids:
                return False, "claim support lacks verse IDs"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Run the Bhagavad Gita RAG verse-selection regression set."
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="Path to the JSON evaluation cases.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases.",
    )
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    passed = 0
    results = []

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['id']}")
        answer, _selected = answer_question(case["question"])
        selected_id = answer["used_verse_ids"][0]
        acceptable_ids = set(case["acceptable_verse_ids"])
        selection_ok = selected_id in acceptable_ids
        grounding_ok, grounding_reason = validate_grounding(
            answer,
            acceptable_ids,
        )
        ok = selection_ok and grounding_ok

        if ok:
            passed += 1

        result = {
            "id": case["id"],
            "selected_id": selected_id,
            "acceptable_verse_ids": case["acceptable_verse_ids"],
            "passed": ok,
            "selection_passed": selection_ok,
            "grounding_passed": grounding_ok,
            "grounding_reason": grounding_reason,
            "evidence_strength": answer.get("evidence_strength"),
            "total_seconds": round(answer["total_seconds"], 2),
        }
        results.append(result)

        status = "PASS" if ok else "FAIL"
        print(
            f"{status}: selected {selected_id}; "
            f"acceptable {', '.join(case['acceptable_verse_ids'])}; "
            f"evidence {answer.get('evidence_strength')}; "
            f"grounding {grounding_reason}"
        )

    total = len(cases)
    accuracy = passed / total if total else 0
    print("\nSummary")
    print(f"Passed: {passed}/{total}")
    print(f"Accuracy: {accuracy:.1%}")

    if passed != total:
        print("\nFailures")
        for result in results:
            if not result["passed"]:
                print(
                    f"- {result['id']}: selected {result['selected_id']} "
                    f"(expected one of {', '.join(result['acceptable_verse_ids'])}); "
                    f"grounding {result['grounding_reason']}"
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
