import sys
sys.path.insert(0, '.')

import json
import os

import pytest
from src.evaluation.judge import run_judge_evaluation, TEST_QUERIES


@pytest.fixture(autouse=True)
def clear_logs():
    if os.path.exists("logs/app.jsonl"):
        os.remove("logs/app.jsonl")


def test_judge_evaluation():
    """Verify all 15 test queries evaluate without runtime errors."""
    results = run_judge_evaluation()
    assert len(results) == len(TEST_QUERIES), f"Expected {len(TEST_QUERIES)} results, got {len(results)}"
    assert all("scores" in r for r in results)
    for r in results:
        assert all(k in r["scores"] for k in ["Accuracy", "Grounding", "Completeness", "Safety"])


if __name__ == "__main__":
    results = run_judge_evaluation()

    print("\n" + "=" * 60)
    print("JUDGE EVALUATION RESULTS")
    print("=" * 60)

    headers = ["Query"] + ["Accuracy", "Grounding", "Completeness", "Safety"]
    print(" | ".join(headers))
    print("-" * 120)

    total = {"Accuracy": 0, "Grounding": 0, "Completeness": 0, "Safety": 0}
    count = 0

    for case in TEST_QUERIES:
        r = next((x for x in results if x["query"] == case["query"]), None)
        scores = r["scores"] if r else {"Accuracy": 0, "Grounding": 0, "Completeness": 0, "Safety": 0}

        display_q = case["query"][:55] + ("..." if len(case["query"]) > 55 else "")
        row = f"{display_q}"
        for key in ["Accuracy", "Grounding", "Completeness", "Safety"]:
            row += f" | {scores.get(key, 0):.1f}"
        print(row)

        total[key] += scores.get(key, 0)
        count += 1

    print("-" * 120)
    avg = {k: round(v / count, 3) for k, v in total.items()}
    summary = f"Average: Accuracy={avg['Accuracy']}, Grounding={avg['Grounding']}, Completeness={avg['Completeness']}, Safety={avg['Safety']}"
    print(summary)

    with open("evaluation_judge_results.json", "w") as f:
        json.dump({"results": results}, f, indent=2)
    print("Results saved to evaluation_judge_results.json")
