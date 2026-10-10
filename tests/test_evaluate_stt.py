from stt.evaluate_stt import calculate_wer


def test_wer_perfect_match():
    result = calculate_wer("Hello everyone.", "Hello everyone!")
    assert result["wer"] == 0.0
    assert result["errors"] == 0


def test_wer_substitution():
    result = calculate_wer("I am happy", "I am hungry")
    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert abs(result["wer"] - 1 / 3) < 1e-9


def test_wer_deletion():
    result = calculate_wer("I really like coffee", "I like coffee")
    assert result["deletions"] == 1
    assert result["insertions"] == 0
    assert abs(result["wer"] - 1 / 4) < 1e-9


def test_wer_insertion():
    result = calculate_wer("I like coffee", "I really like coffee")
    assert result["insertions"] == 1
    assert result["deletions"] == 0
    assert abs(result["wer"] - 1 / 3) < 1e-9
