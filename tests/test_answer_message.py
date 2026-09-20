import sys
import unittest
import types
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

bedrock_client = types.ModuleType("bedrock_client")
bedrock_client.converse_text = lambda *args, **kwargs: ""
bedrock_client.parse_json_text = lambda value: value
sys.modules["bedrock_client"] = bedrock_client

rerank = types.ModuleType("rerank")
rerank.generate_search_queries = lambda question: {}
rerank.retrieve_candidates = lambda question, generated_queries: ([], [])
sys.modules["rerank"] = rerank

from answer import (
    build_grounding_summary,
    format_passage_message,
    repair_grounded_answer,
    validate_grounded_answer,
)

sys.modules.pop("bedrock_client", None)
sys.modules.pop("rerank", None)


class FormatPassageMessageTests(unittest.TestCase):
    def test_formats_answer_without_repeated_introduction(self):
        message = format_passage_message(
            {
                "principle": "Bring the wandering mind back under control.",
                "application": "Return attention gently whenever it wanders.",
            }
        )

        self.assertEqual(
            message,
            (
                "Bring the wandering mind back under control. "
                "Practical application: Return attention gently whenever "
                "it wanders."
            ),
        )
        self.assertNotIn(
            "A related principle from the selected passage is:",
            message,
        )

    def test_validates_grounded_answer_with_allowed_citations(self):
        result = {
            "direct_answer": "Consequences are shown as part of the moral order.",
            "explanation": "The supplied passage connects harmful conduct with future degradation.",
            "confidence": "high",
            "cited_verse_ids": ["BG16.19"],
            "claim_support": [
                {
                    "claim": "Harmful conduct has consequences.",
                    "verse_ids": ["BG16.19"],
                    "support": "The passage describes degraded future births for cruel conduct.",
                }
            ],
            "missing_context": "",
        }

        self.assertTrue(
            validate_grounded_answer(
                result,
                ["BG16.19", "BG16.20"],
            )
        )
        self.assertEqual(result["cited_verse_ids"], ["BG16.19"])

    def test_rejects_grounded_answer_with_unsupplied_citation(self):
        result = {
            "direct_answer": "Consequences are shown as part of the moral order.",
            "explanation": "The supplied passage connects harmful conduct with future degradation.",
            "confidence": "high",
            "cited_verse_ids": ["BG4.7"],
            "claim_support": [
                {
                    "claim": "Harmful conduct has consequences.",
                    "verse_ids": ["BG4.7"],
                    "support": "Unsupported citation.",
                }
            ],
            "missing_context": "",
        }

        self.assertFalse(
            validate_grounded_answer(
                result,
                ["BG16.19", "BG16.20"],
            )
        )

    def test_builds_grounding_summary_with_citation_payload(self):
        selected = [
            {
                "candidate": {
                    "id": "BG2.47",
                    "document": "You have a right to perform your duty, but not to the fruits.",
                    "metadata": {
                        "sanskrit": "karmany evadhikaras te",
                        "transliteration": "karmany evadhikaras te",
                    },
                }
            }
        ]

        grounding = build_grounding_summary(
            evidence_strength="strong",
            evidence_reason="Directly addresses work without attachment.",
            cited_verse_ids=["BG2.47"],
            selected=selected,
            claim_support=[
                {
                    "claim": "Focus on duty rather than results.",
                    "verse_ids": ["BG2.47"],
                    "support": "The verse separates duty from fruits.",
                }
            ],
        )

        self.assertEqual(grounding["confidence"], "high")
        self.assertEqual(grounding["citations"][0]["id"], "BG2.47")

    def test_repairs_malformed_grounding_support(self):
        selected = [
            {
                "candidate": {
                    "id": "BG6.35",
                    "document": (
                        "It is undoubtedly very difficult to curb the "
                        "restless mind, but it is possible by suitable "
                        "practice and by detachment."
                    ),
                    "metadata": {
                        "sanskrit": "asamsayam mahabaho",
                        "transliteration": "asamsayam mahabaho",
                    },
                }
            },
            {
                "candidate": {
                    "id": "BG6.26",
                    "document": (
                        "From whatever and wherever the mind wanders, "
                        "one must bring it back under the control of the self."
                    ),
                    "metadata": {
                        "sanskrit": "yato yato niscalati",
                        "transliteration": "yato yato niscalati",
                    },
                }
            },
        ]
        malformed = {
            "direct_answer": "A wandering mind can be trained.",
            "explanation": "The selected passage points to practice and detachment.",
            "confidence": "HIGH",
            "cited_verse_ids": "6.35",
            "claim_support": [
                {
                    "claim": "Practice helps steady the mind.",
                    "verse_id": "BG6.35",
                    "support": "The passage names practice and detachment.",
                }
            ],
        }

        repaired = repair_grounded_answer(
            malformed,
            ["BG6.35", "BG6.26"],
            selected,
        )

        self.assertTrue(
            validate_grounded_answer(
                repaired,
                ["BG6.35", "BG6.26"],
            )
        )
        self.assertEqual(repaired["confidence"], "high")
        self.assertEqual(repaired["cited_verse_ids"], ["BG6.35"])


if __name__ == "__main__":
    unittest.main()
