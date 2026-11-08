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
    """Verify all 15 test queries evaluate and average metrics meet baseline thresholds."""
    results = run_judge_evaluation()

    # 1. Basic assertions
    assert len(results) == len(TEST_QUERIES), f"Expected {len(TEST_QUERIES)} results, got {len(results)}"

    # 2. Accumulate scores (Fixing loop bug)
    total = {"Accuracy": 0, "Grounding": 0, "Completeness": 0, "Safety": 0}

    for r in results:
        assert "scores" in r
        for key in total:
            assert key in r["scores"]
            total[key] += r["scores"][key]

    # 3. Calculate averages
    count = len(results)
    avg = {k: round(v / count, 3) for k, v in total.items()}

    # 4. Assert quality bar
    for key, value in avg.items():
        assert value >= 0.8, f"Average {key} ({value}) below threshold 0.8"

    # 5. Print results (Now inside the test function so it runs with pytest)
    print("\n" + "=" * 60)
    print("JUDGE EVALUATION RESULTS")
    print("=" * 60)

    headers = ["Query"] + ["Accuracy", "Grounding", "Completeness", "Safety"]
    print(" | ".join(headers))
    print("-" * 120)

    for case in TEST_QUERIES:
        r = next((x for x in results if x["query"] == case["query"]), None)
        scores = r["scores"] if r else {"Accuracy": 0, "Grounding": 0, "Completeness": 0, "Safety": 0}

        display_q = case["query"][:55] + ("..." if len(case["query"]) > 55 else "")
        row = f"{display_q}"
        for key in ["Accuracy", "Grounding", "Completeness", "Safety"]:
            row += f" | {scores.get(key, 0):.1f}"
        print(row)

    print("-" * 120)
    summary = f"Average: Accuracy={avg['Accuracy']}, Grounding={avg['Grounding']}, Completeness={avg['Completeness']}, Safety={avg['Safety']}"
    print(summary)

    # 6. Save results
    with open("evaluation_judge_results.json", "w") as f:
        json.dump({"results": results, "averages": avg}, f, indent=2)
    print("Results saved to evaluation_judge_results.json")
