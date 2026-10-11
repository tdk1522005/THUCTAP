import csv
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

from stt.evaluate_stt import calculate_wer
from stt.evaluation_sentences import EVALUATION_SENTENCES


AUDIO_DIR = Path("evaluation_audio")
RESULT_DIR = Path("evaluation_results")
MODEL_SIZES = ["base", "small"]


def load_audio(path):
    sample_rate, audio = wavfile.read(path)

    if sample_rate != 16000:
        raise ValueError(
            f"{path}: expected 16000 Hz, got {sample_rate} Hz"
        )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    return audio


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = [
        AUDIO_DIR / f"sentence_{i:02d}.wav"
        for i in range(1, len(EVALUATION_SENTENCES) + 1)
    ]

    missing = [str(path) for path in audio_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing audio files:\n" + "\n".join(missing)
        )

    all_results = []

    for model_size in MODEL_SIZES:
        print(f"\n{'=' * 60}")
        print(f"Loading model: {model_size}")

        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
        )

        model_results = []
        total_reference_words = 0
        total_errors = 0
        total_processing_seconds = 0.0

        for index, (audio_path, reference) in enumerate(
            zip(audio_files, EVALUATION_SENTENCES),
            start=1,
        ):
            audio = load_audio(audio_path)

            start_time = time.perf_counter()
            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segments = list(segments)
            elapsed = time.perf_counter() - start_time

            hypothesis = " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            )

            metrics = calculate_wer(reference, hypothesis)
            duration_seconds = len(audio) / 16000

            row = {
                "model": model_size,
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
                "processing_seconds": elapsed,
                "real_time_factor": (
                    elapsed / duration_seconds
                    if duration_seconds > 0 else 0.0
                ),
                "detected_language": info.language,
            }

            model_results.append(row)
            all_results.append(row)

            total_reference_words += metrics["reference_words"]
            total_errors += metrics["errors"]
            total_processing_seconds += elapsed

            print(
                f"[{index:02d}/{len(audio_files)}] "
                f"{audio_path.name} | "
                f"WER={metrics['wer']:.2%} | "
                f"time={elapsed:.2f}s"
            )
            print(f"  REF: {reference}")
            print(f"  HYP: {hypothesis}")

        corpus_wer = (
            total_errors / total_reference_words
            if total_reference_words else 0.0
        )
        print(f"\nSummary for {model_size}:")
        print(f"Corpus WER: {corpus_wer:.2%}")
        print(f"Total errors: {total_errors}")
        print(f"Reference words: {total_reference_words}")
        print(f"Total processing time: {total_processing_seconds:.2f}s")

        del model

    details_path = RESULT_DIR / "benchmark_details.csv"
    with details_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_results)

    summary_path = RESULT_DIR / "benchmark_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "corpus_wer",
            "total_errors",
            "reference_words",
            "total_processing_seconds",
            "average_processing_seconds_per_file",
        ])

        for model_size in MODEL_SIZES:
            rows = [
                row for row in all_results
                if row["model"] == model_size
            ]
            total_errors = sum(row["errors"] for row in rows)
            total_words = sum(row["reference_words"] for row in rows)
            total_time = sum(row["processing_seconds"] for row in rows)
            corpus_wer = total_errors / total_words if total_words else 0.0

            writer.writerow([
                model_size,
                corpus_wer,
                total_errors,
                total_words,
                total_time,
                total_time / len(rows) if rows else 0.0,
            ])

    print("\nBenchmark completed.")
    print(f"Detailed results: {details_path}")
    print(f"Summary results: {summary_path}")


if __name__ == "__main__":
    main()
