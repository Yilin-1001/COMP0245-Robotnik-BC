"""Architecture-ablation report: model-id parsing, run comparability, pooling, the lr rule and the recommendation."""
import copy

import pytest


def test_split_model_id_handles_tapered_labels():
    from experiments.ablation.architecture_report import split_model_id
    assert split_model_id("R4-dart-gate__MLP07-256-128-64__s0__v1__2026-11-03") == ("R4", "MLP07-256-128-64")
    assert split_model_id("R6-gate-dagger-r3__MLP01-256x3__s2__v2__2026-11-01") == ("R6", "MLP01-256x3")


def _run():
    return {"scene_manifest_sha256": "m1",
            "identity": {"code_sha256": {"core/env.py": "a", "evaluation/suite_rollout.py": "b"},
                         "protocol": {"id": "continuous-hold-v1"}, "interface": {"control_hz": 1000}}}


def test_check_comparable_rejects_code_or_manifest_change():
    from experiments.ablation.architecture_report import check_comparable
    check_comparable(_run(), _run())
    changed = _run()
    changed["identity"]["code_sha256"]["core/env.py"] = "z"
    with pytest.raises(ValueError, match="code"):
        check_comparable(_run(), changed)
    changed = _run()
    changed["scene_manifest_sha256"] = "m2"
    with pytest.raises(ValueError, match="manifest"):
        check_comparable(_run(), changed)
    changed = _run()
    changed["identity"]["protocol"] = {"id": "other"}
    with pytest.raises(ValueError, match="protocol"):
        check_comparable(_run(), changed)


def _rollout(model_id, scene, success, suite="T0", grade="A", time="10.0"):
    row = "expert" if model_id.startswith("Markovian") else model_id.split("-")[0]
    return {"model_id": model_id, "row": row, "scenario_id": scene, "suite": suite, "success": str(success),
            "grade": grade, "completion_time_s": time}


def test_aggregate_pools_seeds_per_architecture_and_drops_expert():
    from experiments.ablation.architecture_report import aggregate
    m02 = "R4-dart-gate__MLP02-32x3__s{}__v1__2026-11-03"
    rows = [_rollout(m02.format(0), "a", True), _rollout(m02.format(0), "b", True),
            _rollout(m02.format(1), "a", True), _rollout(m02.format(1), "b", False, grade="E", time=""),
            _rollout("R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28", "a", True),
            _rollout("R1-baseline__MLP01-256x3__s0__v1__2026-10-28", "a", False),
            _rollout("Markovian-expert-A", "a", True)]
    table = aggregate(rows, {"R4"})
    entry = next(e for e in table if (e["row"], e["architecture"], e["suite"]) == ("R4", "MLP02-32x3", "all"))
    assert (entry["k"], entry["n"]) == (3, 4) and entry["rate"] == 0.75
    assert entry["grade_A"] == 3 and entry["grade_E"] == 1 and entry["median_completion_s"] == 10.0
    assert any((e["row"], e["architecture"], e["suite"]) == ("R4", "MLP02-32x3", "T0") for e in table)
    assert {e["row"] for e in table} == {"R4"}


def test_lr_decision():
    from experiments.ablation.architecture_report import lr_decision
    holds = {"MLP02-32x3": {3e-4: 20, 1e-3: 25, 3e-3: 26}, "MLP10-512x3": {3e-4: 30, 1e-3: 29, 3e-3: 28}}
    assert lr_decision(holds) == "keep"
    worse = copy.deepcopy(holds)
    worse["MLP02-32x3"][1e-3] = 23
    assert lr_decision(worse) == "screen"


def test_recommend_picks_smallest_within_margin():
    from experiments.ablation.architecture_report import recommend
    table = [{"row": "R4", "architecture": a, "suite": "all", "parameters": p, "rate": r}
             for a, p, r in [("MLP01-256x3", 141321, 0.95), ("MLP02-32x3", 3337, 0.90),
                             ("MLP04-64x3", 10761, 0.93), ("MLP06-128x3", 37897, 0.96)]]
    table.append({"row": "R4", "architecture": "MLP02-32x3", "suite": "T0", "parameters": 3337, "rate": 1.0})
    assert recommend(table) == {"R4": "MLP04-64x3"}


def _write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_cli_end_to_end_on_synthetic_runs(tmp_path):
    import json
    from experiments.ablation.architecture_report import main
    archs = {"MLP01-256x3": (141321, 278016, 80.0), "MLP02-32x3": (3337, 6080, 20.0)}
    ids = {(row, a, s): f"{row}-x__{a}__s{s}__v1__2026-11-03" for row in ("R4", "R8") for a in archs for s in (0, 1)}
    roll = lambda mid, scene, ok, suite: {"model_id": mid, "row": mid.split("-")[0], "scenario_id": scene, "suite": suite,
                                          "success": str(ok), "grade": "A" if ok else "E", "completion_time_s": "9.5" if ok else ""}
    expert = [{**roll("Markovian-expert-A", f"T{i}-a", True, f"T{i}"), "row": "expert"} for i in range(6)]
    run = {"status": "complete", "scene_manifest_sha256": "m", "identity": {"code_sha256": {"a": "1"}, "protocol": {}, "interface": {}}}
    ref, new = tmp_path / "ref", tmp_path / "new"
    for d, arch in ((ref, "MLP01-256x3"), (new, "MLP02-32x3")):
        rows = [roll(ids[(row, arch, s)], f"T{i}-a", (i + s) % 3 != 0, f"T{i}") for row in ("R4", "R8") for s in (0, 1) for i in range(6)]
        _write_csv(d / "rollouts.csv", rows + expert)
        (d / "run.json").write_text(json.dumps(run))
    exp = tmp_path / "exp"
    runs = []
    for (row, a, s), mid in ids.items():
        _write_csv(tmp_path / "ckpt" / mid / "metrics.csv", [{"val_arm_rmse_nm": 1.0 + s}])
        runs.append({"row": f"{row}-x", "architecture": a, "seed": s, "student_hold": 30, "student_n": 32,
                     "hold_test_median_stiffness_eig_max": 1.5, "hold_test_median_damping_diag": -29.9,
                     "hold_test_median_max_real_eig": 0.02, "model": str(tmp_path / "ckpt" / mid / f"{mid}.npz")})
    _write_csv(exp / "runs.csv", runs)
    _write_csv(tmp_path / "latency.csv", [{"model": "m", "architecture": a, "parameters": p, "flops": f, "npz_bytes": 1,
                                           "median_us": us, "p99_us": us * 2} for a, (p, f, us) in archs.items()])
    assert main(["--experiment", str(exp), "--reference-test", str(ref), "--tests", str(new),
                 "--latency", str(tmp_path / "latency.csv"), "--rows", "R4", "R8"]) == 0
    text = (exp / "architecture_summary.md").read_text()
    assert "| R4 | MLP02-32x3 |" in text and "Expert A determinism" in text and "new 6/6" in text
    for name in ("success_vs_parameters", "success_vs_latency", "width_and_depth_lines", "offline_rmse_vs_test_success"):
        assert (exp / "figures/architecture" / f"{name}.png").stat().st_size > 0
    assert "architecture_summary.md" in (new / "README.md").read_text()


def test_choose_lr_takes_most_holds_with_tie_rule():
    from experiments.ablation.architecture_report import choose_lr
    assert choose_lr({3e-4: 3, 1e-3: 11, 3e-3: 20, 1e-2: 18}) == 3e-3
    assert choose_lr({3e-4: 29, 1e-3: 29, 3e-3: 28, 1e-2: 20}) == 1e-3   # a tie including 1e-3 keeps 1e-3
    assert choose_lr({3e-4: 10, 1e-3: 11, 3e-3: 20, 1e-2: 20}) == 3e-3   # otherwise the lower lr
