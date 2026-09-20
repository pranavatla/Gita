import sys
import unittest
import types
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

bedrock_client = types.ModuleType("bedrock_client")
bedrock_client.converse_text = lambda *args, **kwargs: ""
bedrock_client.embed_texts = lambda texts: []
bedrock_client.parse_json_text = lambda value: value
sys.modules["bedrock_client"] = bedrock_client

rank_bm25 = types.ModuleType("rank_bm25")
rank_bm25.BM25Okapi = object
sys.modules["rank_bm25"] = rank_bm25

from rerank import expand_domain_concepts

sys.modules.pop("bedrock_client", None)
sys.modules.pop("rank_bm25", None)


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
