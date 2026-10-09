import soundfile as sf
from faster_whisper import WhisperModel

from translation.translator import TranslationEngine


# =========================
# 1. Load STT model
# =========================

print("Loading Whisper model...")

stt_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded.")


# =========================
# 2. Load Translation model
# =========================

translator = TranslationEngine()


# =========================
# 3. Read audio
# =========================

audio_path = "test.wav"

audio_data, sample_rate = sf.read(audio_path)

print()
print("Audio loaded.")
print("Sample rate:", sample_rate)


# =========================
# 4. Speech → Text
# =========================

print()
print("Transcribing...")

segments, info = stt_model.transcribe(
    audio_data,
    beam_size=5,
    language="en"
)

english_text = " ".join(
    segment.text for segment in segments
).strip()


print()
print("English:")
print(english_text)


# =========================
# 5. Text → Vietnamese
# =========================

print()
print("Translating...")

result = translator.translate(english_text)


print()
print("Vietnamese:")
print(result["text"])

print()
print(f"Translation latency: {result['latency_ms']:.0f} ms")