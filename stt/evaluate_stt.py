import re
from typing import Dict, List


def normalize_text(text: str) -> List[str]:
    """Normalize text for word-level comparison."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def calculate_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Calculate Word Error Rate (WER).

    S: substitutions
    D: deletions
    I: insertions
    N: number of words in reference
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)

    n = len(ref)
    m = len(hyp)

    # Dynamic programming edit-distance table.
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    # Store operation type for backtracking.
    op = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        op[i][0] = "D"
    for j in range(1, m + 1):
        op[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "M"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
                dp[i][j], op[i][j] = min(choices, key=lambda x: x[0])

    substitutions = deletions = insertions = 0
    i, j = n, m

    while i > 0 or j > 0:
        operation = op[i][j]

        if operation == "M":
            i -= 1
            j -= 1
        elif operation == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif operation == "D":
            deletions += 1
            i -= 1
        elif operation == "I":
            insertions += 1
            j -= 1
        else:
            break

    errors = substitutions + deletions + insertions
    wer = errors / n if n else (0.0 if m == 0 else float("inf"))

    return {
        "wer": wer,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "reference_words": n,
        "errors": errors,
    }


if __name__ == "__main__":
    reference = "I hope you are doing well."
    hypothesis = "I hope you are growing well."

    result = calculate_wer(reference, hypothesis)
    print(f"Reference: {reference}")
    print(f"Hypothesis: {hypothesis}")
    print(f"WER: {result['wer']:.2%}")
    print(
        f"S={result['substitutions']}, "
        f"D={result['deletions']}, "
        f"I={result['insertions']}, "
        f"N={result['reference_words']}"
    )
