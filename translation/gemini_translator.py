import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiTranslationEngine:
    def __init__(self, client=None, model=None):
        self.model = model or os.getenv("GEMINI_TRANSLATION_MODEL", "gemini-2.5-flash")
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return {"text": "", "latency_ms": 0}

        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=(
                "Translate the following English speech into natural Vietnamese. "
                "Preserve meaning and tone. Return only the Vietnamese translation.\n\n"
                + text
            ),
        )
        return {
            "text": (getattr(response, "text", None) or "").strip(),
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
