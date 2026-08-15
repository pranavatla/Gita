import argparse
import json
from pathlib import Path

from rerank import (
    generate_search_queries,
    retrieve_candidates,
    rerank_globally,
)


DEFAULT_EVAL_FILE = "data/eval_holdout.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Bhagavad Gita retrieval."
    )
    parser.add_argument(
        "eval_file",
        nargs="?",
        default=DEFAULT_EVAL_FILE,
        help="Evaluation JSON file to run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_file = Path(args.eval_file)
    cases = json.loads(eval_file.read_text())

    print(f"Evaluation file: {eval_file}")
    passed = 0

    for case in cases:
        print("\n" + "=" * 70)
        print(f"Test: {case['id']}")
        print(f"Query: {case['query']}")

        facets = generate_search_queries(case["query"])

        searches, candidates = retrieve_candidates(
            case["query"],
            list(facets.values()),
        )

        selected, _ = rerank_globally(
            searches,
            candidates,
        )

        returned_ids = [
            result["candidate"]["id"]
            for result in selected
        ]

        expected_ids = case["expected_any"]
        hits = [
            verse_id
            for verse_id in returned_ids
            if verse_id in expected_ids
        ]

        success = bool(hits)

        if success:
            passed += 1

        print("Expected any:", ", ".join(expected_ids))
        print("Returned:", ", ".join(returned_ids))
        print("Hits:", ", ".join(hits) if hits else "None")
        print("Result:", "PASS" if success else "FAIL")

    total = len(cases)

    print("\n" + "=" * 70)
    print(f"Holdout Hit@4: {passed}/{total}")
    print(f"Hit rate: {(passed / total) * 100:.1f}%")


if __name__ == "__main__":
    main()
