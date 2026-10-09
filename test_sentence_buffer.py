from stt.transcript_stabilizer import TranscriptStabilizer
from stt.sentence_buffer import SentenceBuffer


stabilizer = TranscriptStabilizer()
buffer = SentenceBuffer()


texts = [
    "Hello everyone, today I will",
    "Today I want to go to the airport.",
    "the airport and I need to take a taxi.",
]


for text in texts:

    print("Whisper :", text)

    new_text = stabilizer.update(text)

    print("New     :", new_text)

    if new_text:

        sentence = buffer.add(new_text)

        if sentence:
            print("FINAL   :", sentence)
        else:
            print("Waiting...")

    print()