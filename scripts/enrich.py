import argparse
import json
from pathlib import Path

import ollama


SOURCE_DIR = Path("data/sources/bhagavad-gita/slok")
OUTPUT_DIR = Path("data/enriched")
MODEL = "llama3.1:8b"

SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 5,
            "maxItems": 8,
        },
    },
    "required": ["themes"],
}


def load_records():
    records = []

    for path in SOURCE_DIR.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        english = data.get("prabhu", {}).get("et", "").strip()

        if not english or "did not comment" in english:
            continue

        records.append(
            {
                "id": data["_id"],
                "english": english,
            }
        )

    records.sort(
        key=lambda item: tuple(
            int(part) for part in item["id"][2:].split(".")
        )
    )

    assert len(records) == 700, (
        f"Expected 700 records, found {len(records)}"
    )

    return records


def validate(result):
    themes = result.get("themes", [])

    if not isinstance(themes, list):
        return False

    themes = [
        theme.strip().lower()
        for theme in themes
        if isinstance(theme, str) and theme.strip()
    ]

    themes = list(dict.fromkeys(themes))

    if not 5 <= len(themes) <= 8:
        return False

    if any(len(theme.split()) > 5 for theme in themes):
        return False

    result["themes"] = themes
    return True


def enrich(record):
    system_prompt = (
        "Extract 5 to 8 short retrieval themes from the supplied "
        "Bhagavad Gita translation. Each theme must contain 1 to 5 "
        "lowercase words. Extract concepts, actions, motivations, and "
        "consequences explicitly supported by the translation. Prefer "
        "specific terms from the translation over generic moral terms. "
        "Do not summarize the verse. Do not invent facts, people, "
        "citations, chapter numbers, or unsupported concepts."
    )

    for attempt in range(1, 4):
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": record["english"],
                },
            ],
            format=SCHEMA,
            options={"temperature": 0},
        )

        try:
            result = json.loads(response.message.content)
        except json.JSONDecodeError:
            result = {}

        if validate(result):
            return result

        print(
            f"{record['id']}: invalid output, "
            f"retrying ({attempt}/3)"
        )

    raise RuntimeError(
        f"Could not generate valid enrichment for {record['id']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Process one verse, such as BG16.21")
    parser.add_argument("--limit", type=int, help="Process only N verses")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()

    if args.id:
        records = [
            record for record in records
            if record["id"] == args.id
        ]

        if not records:
            raise SystemExit(f"Verse not found: {args.id}")

    if args.limit:
        records = records[:args.limit]

    generated = 0
    skipped = 0

    for index, record in enumerate(records, start=1):
        target = OUTPUT_DIR / f"{record['id']}.json"

        if target.exists():
            skipped += 1
            print(
                f"[{index}/{len(records)}] "
                f"Skipped existing {record['id']}"
            )
            continue

        result = enrich(record)

        output = {
            "id": record["id"],
            "themes": result["themes"],
        }

        temporary = target.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary.replace(target)

        generated += 1
        print(
            f"[{index}/{len(records)}] "
            f"Generated {record['id']}"
        )

    print(
        f"Finished: generated={generated}, "
        f"skipped={skipped}"
    )


if __name__ == "__main__":
    main()
