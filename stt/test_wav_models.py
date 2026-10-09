
import time
import soundfile as sf
from faster_whisper import WhisperModel

AUDIO_PATH = "test.wav"

audio, sample_rate = sf.read(AUDIO_PATH, dtype="float32")

if audio.ndim > 1:
    import numpy as np
    audio = np.mean(audio, axis=1)

print(f"Audio duration: {len(audio) / sample_rate:.2f}s")
print(f"Sample rate: {sample_rate} Hz")

for model_size in ["base", "small"]:
    print(f"\n{'=' * 45}")
    print(f"Testing model: {model_size}")
    print("Loading model...")

    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    start = time.perf_counter()

    segments, info = model.transcribe(
        audio,
        language="en",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    results = list(segments)
    elapsed_ms = (time.perf_counter() - start) * 1000

    text = " ".join(
        segment.text.strip()
        for segment in results
        if segment.text.strip()
    )

    print(f"Detected language: {info.language}")
    print(f"Processing time: {elapsed_ms:.0f} ms")
    print(f"Transcript: {text}")

    print("\nSegment details:")
    for segment in results:
        print(
            f"[{segment.start:.2f}s - {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )

    del model

print("\nTest completed.")
