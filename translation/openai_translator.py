import os
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class OpenAITranslationEngine:
    def __init__(self, client=None, model=None):
        self.model = model or os.getenv(
            "OPENAI_TRANSLATION_MODEL",
            "gpt-4.1-mini",
        )
        self.client = client or OpenAI()

    def translate(self, text):
        text = text.strip()

        if not text:
            return {
                "text": "",
                "latency_ms": 0,
            }

        start_time = time.perf_counter()

        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "Translate the user's English text into natural, "
                "accurate Vietnamese. Preserve the original meaning. "
                "Return only the Vietnamese translation."
            ),
            input=text,
            max_output_tokens=500,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "text": response.output_text.strip(),
            "latency_ms": latency_ms,
        }
