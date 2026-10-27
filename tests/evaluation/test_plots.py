import csv

from evaluation.evaluate import RESULT_COLUMNS, append_results
from evaluation.plots import main


def row(**kw):
    r = {c: "" for c in RESULT_COLUMNS}
    r.update(regime="clean", fit_stage="selection", evaluation_id="rollout-v1", n=32, **kw)
    return r


def test_figures_and_summary_come_from_recorded_rows_only(tmp_path):
    index = tmp_path / "results.csv"
    append_results(index, [
        row(run_id="a1", teacher="A", objective="bc", architecture="m1", seed=0, checkpoint_step=2000,
            evaluation_split="validation", sustained_hold_success=0.5, failures=1),
        row(run_id="a2", teacher="A", objective="bc", architecture="m2", seed=0, checkpoint_step=4000,
            evaluation_split="validation", sustained_hold_success=0.75, failures=0),
        row(run_id="b2", teacher="B", objective="bc", architecture="m2", seed=0, checkpoint_step=4000,
            evaluation_split="validation", sustained_hold_success=0.25, failures=0),
        row(run_id="a2", teacher="A", objective="bc", architecture="m2", seed=0, checkpoint_step=2000,
            evaluation_split="screening", sustained_hold_success=1.0, failures=0)])
    out = tmp_path / "figures"
    assert main(["--index", str(index), "--out", str(out)]) == 0
    assert (out / "validation_hold_by_architecture.png").stat().st_size > 0
    summary = list(csv.DictReader(open(out / "validation_summary.csv")))
    assert {(r["teacher"], r["architecture"], r["runs"], r["mean_sustained_hold"]) for r in summary} == \
        {("A", "m1", "1", "0.5"), ("A", "m2", "1", "0.75"), ("B", "m2", "1", "0.25")}
    assert {r["failures"] for r in summary if r["architecture"] == "m1"} == {"1"}
