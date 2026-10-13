from unittest.mock import MagicMock, patch

from translation.translator import TranslationEngine


def make_engine_without_loading_model():
    with patch("translation.translator.AutoTokenizer") as tokenizer_class, \
         patch("translation.translator.AutoModelForSeq2SeqLM") as model_class:

        tokenizer = MagicMock()
        model = MagicMock()

        tokenizer_class.from_pretrained.return_value = tokenizer
        model_class.from_pretrained.return_value = model

        engine = TranslationEngine()

    return engine, tokenizer, model


def test_empty_input_returns_empty_translation():
    engine, tokenizer, model = make_engine_without_loading_model()

    result = engine.translate("   ")

    assert result["text"] == ""
    assert result["latency_ms"] == 0
    tokenizer.assert_not_called()
    model.generate.assert_not_called()


def test_translate_returns_decoded_text():
    engine, tokenizer, model = make_engine_without_loading_model()

    tokenizer.return_value = {"input_ids": "mock-input"}
    model.generate.return_value = ["mock-output"]
    tokenizer.decode.return_value = "Tôi muốn đi đến sân bay."

    result = engine.translate("  I want to go to the airport.  ")

    assert result["text"] == "Tôi muốn đi đến sân bay."
    assert result["latency_ms"] >= 0
    tokenizer.assert_called_once_with(
        "I want to go to the airport.",
        return_tensors="pt",
    )
    model.generate.assert_called_once_with(input_ids="mock-input")
    tokenizer.decode.assert_called_once_with(
        "mock-output",
        skip_special_tokens=True,
    )
