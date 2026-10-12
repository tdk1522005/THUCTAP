import csv
import time
from pathlib import Path

from stt.benchmark_stt import load_audio
from stt.evaluate_stt import calculate_wer
from stt.evaluation_sentences import EVALUATION_SENTENCES
from stt.openai_transcriber import OpenAITranscriber


AUDIO_DIR = Path("evaluation_audio")
RESULT_DIR = Path("evaluation_results")


def main():
    audio_files = [
        AUDIO_DIR / f"sentence_{i:02d}.wav"
        for i in range(1, len(EVALUATION_SENTENCES) + 1)
    ]

    missing = [str(path) for path in audio_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing audio files:\n" + "\n".join(missing)
        )

    transcriber = OpenAITranscriber()
    print(f"OpenAI STT model: {transcriber.model}")
    print(f"Audio files: {len(audio_files)}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for index, (audio_path, reference) in enumerate(
        zip(audio_files, EVALUATION_SENTENCES),
        start=1,
    ):
        audio = load_audio(audio_path)

        try:
            result = transcriber.transcribe(audio, sample_rate=16000)
        except Exception as exc:
            print(f"\n[{index:02d}] API error for {audio_path.name}: {exc}")
            raise

        hypothesis = result["text"]
        latency_seconds = result["latency_ms"] / 1000
        duration_seconds = len(audio) / 16000
        metrics = calculate_wer(reference, hypothesis)

        row = {
            "model": transcriber.model,
            "file": audio_path.name,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": metrics["wer"],
            "substitutions": metrics["substitutions"],
            "deletions": metrics["deletions"],
            "insertions": metrics["insertions"],
            "reference_words": metrics["reference_words"],
            "errors": metrics["errors"],
            "audio_seconds": duration_seconds,
            "processing_seconds": latency_seconds,
            "real_time_factor": (
                latency_seconds / duration_seconds
                if duration_seconds > 0 else 0.0
            ),
        }
        results.append(row)

        print(
            f"\n[{index:02d}/{len(audio_files)}] {audio_path.name} | "
            f"WER={metrics['wer']:.2%} | "
            f"API latency={latency_seconds:.2f}s"
        )
        print(f"  REF: {reference}")
        print(f"  HYP: {hypothesis}")

    total_errors = sum(row["errors"] for row in results)
    total_words = sum(row["reference_words"] for row in results)
    total_latency = sum(row["processing_seconds"] for row in results)
    corpus_wer = total_errors / total_words if total_words else 0.0

    details_path = RESULT_DIR / "benchmark_openai_stt_details.csv"
    with details_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path = RESULT_DIR / "benchmark_openai_stt_summary.csv"
    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "corpus_wer",
            "total_errors",
            "reference_words",
            "total_processing_seconds",
            "average_processing_seconds_per_file",
        ])
        writer.writerow([
            transcriber.model,
            corpus_wer,
            total_errors,
            total_words,
            total_latency,
            total_latency / len(results) if results else 0.0,
        ])

    print("\nOpenAI STT benchmark completed.")
    print(f"Corpus WER: {corpus_wer:.2%}")
    print(f"Total errors: {total_errors}/{total_words} reference words")
    print(f"Average API latency: {total_latency / len(results):.2f}s/file")
    print(f"Detailed results: {details_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
