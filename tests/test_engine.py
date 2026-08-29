"""End-to-end benchmark and functional test suite for SentimentInferenceEngine."""

import json
from pathlib import Path
import pytest
from src.engine import SentimentInferenceEngine


@pytest.fixture(scope="module")
def engine():
    eng = SentimentInferenceEngine()
    eng.load()
    return eng


def test_engine_output_format(engine):
    sample = "አገልግሎቱ በጣም ፈጣንና አስተማማኝ ነው፤ እጅግ በጣም ወደድኩት።"
    res = engine.predict(sample)

    assert isinstance(res, dict)
    assert "class" in res
    assert "confidence" in res
    assert "p_pos" in res
    assert "p_neg" in res
    assert "cleaned_text" in res
    assert "latency_ms" in res

    assert res["class"] in ["Positive", "Negative", "Neutral", "Mixed"]
    assert 0.0 <= res["confidence"] <= 100.0
    assert 0.0 <= res["p_pos"] <= 1.0
    assert 0.0 <= res["p_neg"] <= 1.0
    assert res["latency_ms"] >= 0.0


def test_empty_input(engine):
    res = engine.predict("")
    assert res["class"] == "Neutral"
    assert res["confidence"] == 100.0
    assert res["p_pos"] == 0.0
    assert res["p_neg"] == 0.0


def test_all_10_golden_benchmark_cases(engine):
    bench_path = Path(__file__).parent / "benchmark_cases.json"
    assert bench_path.exists(), "benchmark_cases.json missing"

    with open(bench_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 10, "Must have exactly 10 Golden Benchmark Cases"

    mismatches = []
    for item in cases:
        case_id = item["id"]
        category = item["category"]
        text = item["text"]
        expected = item["expected_class"]

        result = engine.predict(text)
        actual = result["class"]

        if actual.lower() != expected.lower():
            mismatches.append({
                "id": case_id,
                "category": category,
                "text": text,
                "expected": expected,
                "actual": actual,
                "p_pos": result["p_pos"],
                "p_neg": result["p_neg"],
                "conf": result["confidence"],
            })

    assert len(mismatches) == 0, f"Benchmark test failures: {json.dumps(mismatches, ensure_ascii=False, indent=2)}"
