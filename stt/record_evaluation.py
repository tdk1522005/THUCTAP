import time
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write

from stt.evaluation_sentences import EVALUATION_SENTENCES


SAMPLE_RATE = 16000
OUTPUT_DIR = Path("evaluation_audio")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("STT evaluation recording")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Number of sentences: {len(EVALUATION_SENTENCES)}")
    print("Read each sentence naturally and clearly.")
    print("Press Ctrl+C to stop.\n")

    for index, sentence in enumerate(EVALUATION_SENTENCES, start=1):
        output_file = OUTPUT_DIR / f"sentence_{index:02d}.wav"

        while True:
            print(f"\n[{index}/{len(EVALUATION_SENTENCES)}] {sentence}")
            input("Press Enter when ready to record...")

            duration = 10
            print(f"Recording for up to {duration} seconds. Speak now!")

            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            print("Recording finished.")
            choice = input("Save this recording? [Y = save / R = record again]: ")
            if choice.strip().lower() == "r":
                continue

            write(str(output_file), SAMPLE_RATE, audio)
            print(f"Saved: {output_file}")
            break

    print(f"\nFinished. Recordings are in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
