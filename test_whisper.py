from faster_whisper import WhisperModel
import time

MODEL_SIZE = "medium"

print("loading model...")


model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)

print("Model loader.")

audio_file = "test.wav"

start_time = time.perf_counter()

segments, info = model.transcribe(
    audio_file,
    language="vi",
    beam_size=5,
)

elapsed = time.perf_counter() - start_time

print("\n--- RESULT ---")

for segment in segments:
    print(
        f"[{segment.start:.2f}s -> {segment.end:2f}s]"
        f"{segment.text}"
    )

print("\n--- INFO ---")
print("Dectected language: ", info.language)
print("Language probability: ", info.language_probability)
print(f"Processing time: {elapsed:.2f}s")

