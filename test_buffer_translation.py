from stt.sentence_buffer import SentenceBuffer
from translation.translator import TranslationEngine


buffer = SentenceBuffer()
translator = TranslationEngine()


texts = [
    "I want to go",
    "to the airport."
]


for text in texts:

    sentence = buffer.add(text)

    print("STT input :", text)

    if sentence:
        print("Final sentence:", sentence)

        result = translator.translate(sentence)

        print("Vietnamese:", result["text"])
        print(f"Translation latency: {result['latency_ms']:.0f} ms")

    else:
        print("Waiting for complete sentence...")

    print()