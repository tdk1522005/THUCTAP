import time

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationEngine:
    def __init__(
        self,
        model_name="Helsinki-NLP/opus-mt-en-vi"
    ):
        print("Loading translation model...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        print("Translation model loaded.")

    def translate(self, text):
        text = text.strip()

        if not text:
            return {
                "text": "",
                "latency_ms": 0
            }

        start_time = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        )

        outputs = self.model.generate(
            **inputs
        )

        translation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        latency = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "text": translation,
            "latency_ms": latency
        }


if __name__ == "__main__":

    translator = TranslationEngine()

    text = "I want to go to the airport."

    result = translator.translate(text)

    print()
    print("Input :", text)
    print("Output:", result["text"])
    print(f"Latency: {result['latency_ms']:.0f} ms")