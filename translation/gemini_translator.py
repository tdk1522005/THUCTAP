import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiTranslationEngine:
    def __init__(self, client=None, model=None, retry_attempts=3):
        self.model = model or os.getenv("GEMINI_TRANSLATION_MODEL", "gemini-3.8-flash")
        self.client = client or genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.retry_attempts = max(1, int(retry_attempts))

    def translate(self, text):
        text = (text or "").strip()
        if not text:
            return {"text": "", "latency_ms": 0}

        started = time.perf_counter()
        contents = (
            "Translate the following English speech into natural Vietnamese. "
            "Preserve meaning and tone. Return only the Vietnamese translation.\n\n"
            + text
        )

        for attempt in range(self.retry_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                return {
                    "text": (getattr(response, "text", None) or "").strip(),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            except Exception as exc:
                # Retry only transient capacity errors; configuration/auth errors
                # should surface immediately instead of wasting time or quota.
                if getattr(exc, "code", None) != 503 and getattr(exc, "status_code", None) != 503:
                    raise
                if attempt + 1 >= self.retry_attempts:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("Translation retry loop ended unexpectedly")
