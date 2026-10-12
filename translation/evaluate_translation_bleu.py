import csv
from pathlib import Path

import sacrebleu

from translation.translation_references import REFERENCE_TRANSLATIONS


RESULT_DIR = Path("evaluation_results")

MODELS = {
    "OPUS-MT": RESULT_DIR / "translation_benchmark.csv",
    "OpenAI gpt-4.1-mini": RESULT_DIR / "openai_translation_benchmark.csv",
}


def load_translations(file_path):
    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return [row["vietnamese"].strip() for row in rows]


def main():
    references = REFERENCE_TRANSLATIONS

    for model_name, file_path in MODELS.items():
        if not file_path.exists():
            print(f"Missing result file: {file_path}")
            continue

        translations = load_translations(file_path)

        if len(translations) != len(references):
            print(
                f"{model_name}: number of translations "
                f"({len(translations)}) does not match "
                f"references ({len(references)})."
            )
            continue

        bleu = sacrebleu.corpus_bleu(
            translations,
            [references],
        )

        print(f"\nModel: {model_name}")
        print(f"Corpus BLEU: {bleu.score:.2f}")
        print(f"Number of sentences: {len(translations)}")


if __name__ == "__main__":
    main()
