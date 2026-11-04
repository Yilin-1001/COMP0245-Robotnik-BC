"""Training-curve figures: run names map to rows, rounds and seeds; metrics (including the JSON extra) are read."""
import csv
import json

from evaluation.plot_training_curves import (discover, plot_rounds, plot_row, read_dagger_summary, read_metrics,
                                             series_for_row)

HEADER = ["update", "elapsed_s", "lr", "train_loss", "train_arm_loss", "train_wheel_loss", "val_arm_rmse_nm",
          "val_wheel_rmse_rad_s", "val_hold_arm_rmse_nm", "val_hold_bias_max_abs_nm", "val_hold_bias_gate_pass",
          "val_damping_median_diag", "val_damping_frac_below_minus33", "extra"]


def write_run(root, name, n=4):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for k in range(1, n + 1):
            w.writerow([500 * k, k, 1e-3, 1.0 / k, 0.5 / k, 0.1 / k, 0.6, 0.04, 0.4, 2.0, False, -29.9, 0.01,
                        json.dumps({"jacobian_loss": 0.2 / k})])
    return d


def test_runs_map_to_rows_rounds_and_seeds(tmp_path):
    write_run(tmp_path, "R4-dart-gate__MLP01-256x3__s1__v1__2026-10-28")
    write_run(tmp_path, "R6-gate-dagger-r2__MLP01-256x3__s0__v2__2026-11-01")
    write_run(tmp_path, "R4-dart-gate-s150__MLP01-256x3__s0__v2__2026-11-01")
    (tmp_path / "not-a-run").mkdir()
    runs = {r["name"]: (r["row"], r["round"], r["seed"]) for r in discover(tmp_path)}
    assert runs == {"R4-dart-gate__MLP01-256x3__s1__v1__2026-10-28": ("R4-dart-gate", None, 1),
                    "R6-gate-dagger-r2__MLP01-256x3__s0__v2__2026-11-01": ("R6-gate-dagger", 2, 0),
                    "R4-dart-gate-s150__MLP01-256x3__s0__v2__2026-11-01": ("R4-dart-gate-s150", None, 0)}
    assert discover(tmp_path / "missing") == []


def test_metrics_include_the_jacobian_loss_from_the_extra_column(tmp_path):
    m = read_metrics(write_run(tmp_path, "R1-baseline__MLP01-256x3__s0__v1__2026-10-28") / "metrics.csv")
    assert list(m["update"]) == [500, 1000, 1500, 2000]
    assert m["jacobian_loss"][1] == 0.1 and "extra" not in m


def test_figures_are_written_as_png_and_pdf(tmp_path):
    exp = tmp_path / "exp"
    write_run(exp, "R6-gate-dagger-r1__MLP01-256x3__s0__v2__2026-11-01")
    runs = discover(exp)
    paths = plot_row("R6-gate-dagger", series_for_row("R6-gate-dagger", runs), tmp_path / "out")
    summary = tmp_path / "summary.csv"
    summary.write_text("row,round,seed,sustained_hold,n\nR2-gate,0,0,1/32,32\nR2-gate,1,0,24/32,32\nR2-gate,1,1,25/32,32\n")
    paths += plot_rounds({"R6-gate-dagger": read_dagger_summary(summary)}, tmp_path / "out")
    assert sorted(p.name for p in paths) == ["R6-DAgger-Gate.pdf", "R6-DAgger-Gate.png", "dagger_rounds.pdf",
                                             "dagger_rounds.png"]
    assert all(p.stat().st_size > 0 for p in paths)
