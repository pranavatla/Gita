import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rerank import expand_domain_concepts


class DomainExpansionTests(unittest.TestCase):
    def test_bridges_guna_terms_to_english_corpus_vocabulary(self):
        expansions = expand_domain_concepts(
            "gunas sattva rajas tamas clarity restlessness laziness"
        )

        self.assertEqual(
            expansions,
            [
                "three modes goodness passion ignorance",
                "mode of goodness illuminating happiness knowledge",
                "mode of passion desires longings material actions",
                "mode of ignorance delusion indolence sleep",
            ],
        )

    def test_bridges_spiritual_work_to_offering_passages(self):
        expansions = expand_domain_concepts(
            "turn ordinary work into something spiritual"
        )

        self.assertEqual(
            expansions,
            [
                "whatever you do eat give perform as an offering to God",
                "work as sacrifice for the Supreme remain free from bondage",
            ],
        )


if __name__ == "__main__":
    unittest.main()
