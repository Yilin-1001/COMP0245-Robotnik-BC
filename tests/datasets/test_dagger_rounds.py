"""experiments.dagger_rounds: the v2 defaults are unchanged, and --version v3 --variant-suffix=-nodev --dev-scenes 0
reproduces the names and variant files of the 2026-11-05 run without development scenes."""
import csv
import shutil

import pytest
import yaml

from conftest import TASK_DIR

import experiments.dagger_rounds as dr

ROWS = ["R5-baseline-dagger", "R6-gate-dagger", "R7-dart-dagger", "R8-dart-gate-dagger"]
VARIANTS = TASK_DIR / "experiments/configs/datasets/variants"
V3_RESULTS = TASK_DIR / "results/ablation/2026-11-05__training-ablation-v3/dagger"


@pytest.fixture
def tmp_task(tmp_path, monkeypatch):
    (tmp_path / "experiments/configs/datasets/variants").mkdir(parents=True)
    monkeypatch.setattr(dr, "TASK_DIR", tmp_path)
    return tmp_path


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_v2_defaults_write_the_tracked_variant_byte_for_byte(tmp_task, row, k):
    name = dr.write_variant(row, k)
    assert name == f"{row}-r{k}"
    written = (tmp_task / "experiments/configs/datasets/variants" / f"{name}.yaml").read_bytes()
    assert written == (VARIANTS / f"{name}.yaml").read_bytes()


@pytest.mark.parametrize("row", ROWS)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_v3_options_match_the_nodev_variants_and_keep_them_byte_identical(tmp_task, row, k):
    recorded = VARIANTS / f"{row}-nodev-r{k}.yaml"
    if not recorded.exists():
        pytest.skip("v3 variant files not present")
    # Fresh write: same data as the recorded file.
    name = dr.write_variant(row, k, "v3", "-nodev", 0)
    assert name == f"{row}-nodev-r{k}"
    fresh = tmp_task / "experiments/configs/datasets/variants" / f"{name}.yaml"
    assert yaml.safe_load(fresh.read_text()) == yaml.safe_load(recorded.read_text())
    assert f"dagger-{row.split('-')[0]}-v3-r{k}__" in fresh.read_text()
    # Existing file (training recorded its SHA-256): left exactly as it is.
    shutil.copyfile(recorded, fresh)
    dr.write_variant(row, k, "v3", "-nodev", 0)
    assert fresh.read_bytes() == recorded.read_bytes()


def test_variant_with_different_data_is_never_overwritten(tmp_task):
    path = tmp_task / "experiments/configs/datasets/variants/R8-dart-gate-dagger-nodev-r1.yaml"
    path.write_text("data:\n  feature_definition: features-v1-gate\n  datasets: [something-else]\n")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        dr.write_variant("R8-dart-gate-dagger", 1, "v3", "-nodev", 0)


@pytest.mark.parametrize("row", ROWS)
def test_v3_model_paths_match_the_recorded_run(row):
    summary = V3_RESULTS / row / "summary.csv"
    if not summary.exists():
        pytest.skip("v3 results not present")
    for r in csv.DictReader(open(summary)):
        if r["round"] == "0":
            continue
        variant = f"{row}-nodev-r{r['round']}"
        path = dr.model_path(variant, int(r["seed"]), "2026-11-05", "v3")
        assert path.relative_to(TASK_DIR).as_posix() == r["model"]


@pytest.mark.parametrize("argv", [["--dev-scenes", "0"], ["--variant-suffix=-nodev"]])
def test_non_default_collection_needs_its_own_version(argv, capsys):
    with pytest.raises(SystemExit):
        dr.main(["--row", "R8-dart-gate-dagger", "--date", "2026-11-05", *argv])
    assert "needs its own --version" in capsys.readouterr().err


def test_round_zero_reuses_a_result_from_another_experiment_only(tmp_task):
    stem = "R4-dart-gate__MLP01-256x3__s0__v1__2026-10-28"
    old = tmp_task / "results/ablation/2026-11-01__training-ablation-v2/dagger/R8-dart-gate-dagger"
    new = tmp_task / "results/ablation/2026-11-05__training-ablation-v3/dagger/R8-dart-gate-dagger"
    (old / "validation").mkdir(parents=True)
    (old / "validation" / f"{stem}.json").write_text("{}")
    assert dr.earlier_validation("R8-dart-gate-dagger", stem, new) == old / "validation" / f"{stem}.json"
    assert dr.earlier_validation("R8-dart-gate-dagger", stem, old) is None


@pytest.mark.parametrize("row", ["R6-gate-dagger", "R8-dart-gate-dagger"])
def test_extra_seed_collectors_match_the_recorded_runs(row):
    from experiments.dagger_extra_seeds import collector
    for version, suffix, date, results in (("v2", "", "2026-11-01", "2026-11-01__training-ablation-v2"),
                                           ("v3", "-nodev", "2026-11-05", "2026-11-05__training-ablation-v3")):
        summary = TASK_DIR / "results/ablation" / results / "dagger" / row / "summary.csv"
        if not summary.exists():
            pytest.skip("DAgger results not present")
        recorded = {(r["round"], r["seed"]): r["collector"] for r in csv.DictReader(open(summary))}
        for k in (1, 2, 3):
            assert collector(row, k, version, suffix, date) == recorded[(str(k), "0")]
