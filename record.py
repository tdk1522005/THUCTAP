import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
DURATION = 10
OUTPUT_FILE = "test.wav"

print(f"Recording audio for {DURATION} seconds...")
print("Speak English now!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16",
)

sd.wait()

write(OUTPUT_FILE, SAMPLE_RATE, audio)

print(f"Saved: {OUTPUT_FILE}")