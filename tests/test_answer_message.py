import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from answer import format_passage_message


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


if __name__ == "__main__":
    unittest.main()
