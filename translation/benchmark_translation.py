import csv
import statistics
import time
from pathlib import Path

from translation.translator import TranslationEngine


TEST_SENTENCES = [
    "I want to go to the airport.",
    "Hello everyone, today I want to go to the airport.",
    "Can you tell me where the train station is?",
    "I want to book a hotel room for two people for three nights.",
    "If the flight is delayed, can we change the pickup time?",
]

OUTPUT_DIR = Path("evaluation_results")
OUTPUT_FILE = OUTPUT_DIR / "translation_benchmark.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    translator = TranslationEngine()
    rows = []

    for index, sentence in enumerate(TEST_SENTENCES, start=1):
        result = translator.translate(sentence)

        row = {
            "test_id": index,
            "english": sentence,
            "vietnamese": result["text"],
            "latency_ms": round(result["latency_ms"], 2),
        }
        rows.append(row)

        print(f"\nTest {index}")
        print(f"English: {row['english']}")
        print(f"Vietnamese: {row['vietnamese']}")
        print(f"Latency: {row['latency_ms']:.2f} ms")

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    latencies = [row["latency_ms"] for row in rows]

    print("\nTranslation benchmark summary")
    print(f"Number of sentences: {len(rows)}")
    print(f"Average latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"Minimum latency: {min(latencies):.2f} ms")
    print(f"Maximum latency: {max(latencies):.2f} ms")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
