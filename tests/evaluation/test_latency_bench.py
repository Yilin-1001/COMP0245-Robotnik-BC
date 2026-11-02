"""The MLP cost model matches the exported weights, and the isolated latency benchmark returns sane statistics."""
import numpy as np
import pytest

from conftest import TASK_DIR

R4 = TASK_DIR / ("checkpoints/ablation/2026-10-28__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28/"
                 "R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28.npz")


def test_cost_matches_known_counts():
    from evaluation.latency_bench import mlp_cost
    assert mlp_cost(22, [256] * 3) == {"parameters": 141321, "flops": 278016}
    assert mlp_cost(22, [32] * 3) == {"parameters": 3337, "flops": 6080}
    assert mlp_cost(22, [256, 128, 64]) == {"parameters": 48521, "flops": 94336}


@pytest.mark.realdata
def test_cost_agrees_with_exported_model():
    from evaluation.latency_bench import mlp_cost
    if not R4.exists():
        pytest.skip("R4 v1 checkpoint not present")
    with np.load(R4) as z:
        stored = sum(z[k].size for k in z.files if k != "metadata_json")
    assert mlp_cost(22, [256] * 3)["parameters"] == stored


@pytest.mark.realdata
def test_time_predict_returns_positive_stats():
    from evaluation.latency_bench import time_predict
    if not R4.exists():
        pytest.skip("R4 v1 checkpoint not present")
    t = time_predict(str(R4), calls=200, warmup=20, repeats=2)
    assert 0 < t["median_us"] <= t["p99_us"] and len(t["repeat_medians_us"]) == 2


@pytest.mark.realdata
def test_cli_writes_one_row_per_model(tmp_path):
    import csv
    from evaluation.latency_bench import main
    if not R4.exists():
        pytest.skip("R4 v1 checkpoint not present")
    out = tmp_path / "latency.csv"
    assert main(["--models", str(R4), "--out", str(out), "--calls", "50"]) == 0
    (row,) = list(csv.DictReader(out.open()))
    assert row["architecture"] == "MLP01-256x3" and int(row["parameters"]) == 141321 and int(row["flops"]) == 278016
    assert float(row["median_us"]) > 0 and int(row["npz_bytes"]) > 0
