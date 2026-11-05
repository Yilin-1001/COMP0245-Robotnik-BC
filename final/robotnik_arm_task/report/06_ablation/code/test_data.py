"""The unified-test records of R1-R8, merged from two runs of the same frozen 240-scene suite.

  data/test/2026-11-01_R1-R8/      R1-R4 (ablation v1) and the expert, from results/test/2026-11-01_215340__R1-R8
                                   (that run also tested the superseded v2 R5-R8; those rows are not used)
  data/test/2026-11-05_R5-R8_v3/   R5-R8 (ablation v3, DAgger without development scenes) and the expert,
                                   from results/test/2026-11-05_130144__nodev-dagger-R5-R8

Both runs use the byte-identical scene manifest and identical simulator/policy code hashes (run.json); the expert's 240
rollouts are identical field for field, which is checked here on every load. Only the summary/plot scripts differ
(report names). v2 vs v3 is compared in ../../08_dev_scene_removal/.
"""
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
V1_RUN, V3_RUN = DATA / "test/2026-11-01_R1-R8", DATA / "test/2026-11-05_R5-R8_v3"
FROM_V1_RUN, FROM_V3_RUN = {"expert", "R1", "R2", "R3", "R4"}, {"R5", "R6", "R7", "R8"}
EXPERT_FIELDS = ["success", "grade", "completion_time_s", "final_err_mm", "final1s_max_err_mm", "min_err_mm", "steps"]


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _check_runs():
    a, b = (V1_RUN / "scene_manifest.json").read_bytes(), (V3_RUN / "scene_manifest.json").read_bytes()
    assert a == b, "the two test runs used different scene manifests"


def load_rollouts():
    """Raw rollout rows (strings), R1-R4 + expert from the 2026-11-01 run and R5-R8 from the 2026-11-05 run."""
    _check_runs()
    first, second = _read(V1_RUN / "rollouts.csv"), _read(V3_RUN / "rollouts.csv")
    key = lambda r: r["scenario_id"]
    e1 = sorted((r for r in first if r["row"] == "expert"), key=key)
    e2 = sorted((r for r in second if r["row"] == "expert"), key=key)
    assert len(e1) == len(e2) == 240 and all(x["scenario_id"] == y["scenario_id"] and
                                             all(x[c] == y[c] for c in EXPERT_FIELDS) for x, y in zip(e1, e2)), \
        "expert rollouts differ between the two test runs"
    second_rows = [r for r in second if r["row"] in FROM_V3_RUN]
    assert all("-nodev-r3__" in r["model_id"] and "__v3__" in r["model_id"] for r in second_rows)
    rows = [r for r in first if r["row"] in FROM_V1_RUN] + second_rows
    assert len(rows) == 240 + 8 * 3 * 240, len(rows)
    return rows


def load_summary():
    """The suites' own row-level summary lines, taken from the run each row comes from."""
    _check_runs()
    return ([r for r in _read(V1_RUN / "summary.csv") if r["row"] in FROM_V1_RUN] +
            [r for r in _read(V3_RUN / "summary.csv") if r["row"] in FROM_V3_RUN])
