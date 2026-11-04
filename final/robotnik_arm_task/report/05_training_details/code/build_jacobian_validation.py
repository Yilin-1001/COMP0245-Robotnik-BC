"""Rebuild the R4 BC / BC+J validation tables and figure from this report's copies.

Run from final/robotnik_arm_task/:
    python report/05_training_details/code/build_jacobian_validation.py
No training, simulator, checkpoint weights or network access is required.
"""
import csv
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/objective_ablation"
ROWS = {"bc": "R4-dart-gate-bc", "bcj": "R4-dart-gate"}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    path = ROOT / "tables" / name
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"tables/{name}: {len(rows)} rows")


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def load_runs(objective):
    rows = sorted((r for r in read_csv(DATA / objective / "runs.csv") if r["row"] == ROWS[objective]),
                  key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == [0, 1, 2], objective
    for r in rows:
        config_path = DATA / objective / "configs" / f"{r['run']}.yaml"
        r["config"] = yaml.safe_load(config_path.read_text())
        assert r["config"]["identity"]["objective"] == objective
        assert r["config"]["training"]["updates"] == 42000
        r["scenarios"] = {}
        for mode in ("student", "arm"):
            detail = read_csv(DATA / objective / "validation" / f"{r['run']}__validation-{mode}.csv")
            ids = {d["scenario_id"] for d in detail}
            assert len(detail) == len(ids) == int(r[f"{mode}_n"]) == 32
            assert sum(d["sustained_hold_pass"] == "True" for d in detail) == int(r[f"{mode}_hold"])
            assert sum(d["ever_reached"] == "True" for d in detail) == int(r[f"{mode}_reach"])
            r["scenarios"][mode] = ids
    return rows


def audit_pairs(bc, bcj):
    audit = []
    for a, b in zip(bc, bcj):
        ca, cb = a["config"], b["config"]
        ma, mb = ca["model"], cb["model"]
        arch = ca["identity"]["architecture"]
        checks = {
            "same_seed": ca["identity"]["seed"] == cb["identity"]["seed"],
            "same_selected_architecture": arch == cb["identity"]["architecture"]
                and ma["architectures"][arch] == mb["architectures"][arch]
                and {k: v for k, v in ma.items() if k != "architectures"}
                == {k: v for k, v in mb.items() if k != "architectures"},
            "same_training": ca["training"] == cb["training"],
            "same_data_settings": ca["data"] == cb["data"],
            "same_evaluation": ca["evaluation"] == cb["evaluation"],
            "same_interface": ca["interface"] == cb["interface"],
            "same_action_loss": {k: v for k, v in ca["loss"].items() if k != "jacobian"}
                == {k: v for k, v in cb["loss"].items() if k != "jacobian"},
            "same_validation_scenarios": a["scenarios"] == b["scenarios"],
        }
        for key in ("datasets", "validation_dataset", "fitting_ids", "validation_ids", "split_sha256", "normaliser_sha256"):
            va, vb = ca["provenance"][key], cb["provenance"][key]
            checks[f"same_{key}"] = va == vb
        assert all(checks.values()), (a["seed"], checks)
        # Recorded for provenance, not assumed equal across runs made on different dates.
        audit.append({"seed": a["seed"], **checks,
                      "bc_code_digest": ca["provenance"]["code_digest"],
                      "bcj_code_digest": cb["provenance"]["code_digest"]})
    write_csv("jacobian_config_audit.csv", audit)


def tables(groups):
    per_seed, summary = [], []
    for objective, rows in groups.items():
        for r in rows:
            per_seed.append({"objective": objective, "seed": r["seed"], "run": r["run"],
                             **{k: r[k] for k in ("student_n", "student_hold", "student_reach", "arm_n", "arm_hold",
                                 "hold_test_scenarios", "hold_test_arm_stayed", "hold_test_median_damping_diag")}})
        n = sum(int(r["student_n"]) for r in rows)
        k = sum(int(r["student_hold"]) for r in rows)
        low, high = wilson(k, n)
        summary.append({"objective": objective, "seeds": len(rows),
                        "student_hold_per_seed": "/".join(r["student_hold"] for r in rows),
                        "student_hold": k, "student_n": n, "student_hold_rate": k / n,
                        "wilson95_low": low, "wilson95_high": high,
                        "arm_hold": sum(int(r["arm_hold"]) for r in rows),
                        "arm_n": sum(int(r["arm_n"]) for r in rows),
                        "hold_test_arm_stayed": sum(int(r["hold_test_arm_stayed"]) for r in rows),
                        "hold_test_scenarios": sum(int(r["hold_test_scenarios"]) for r in rows),
                        "hold_test_median_damping_diag": median(float(r["hold_test_median_damping_diag"]) for r in rows)})
    write_csv("jacobian_closed_loop_per_seed.csv", per_seed)
    write_csv("jacobian_closed_loop.csv", summary)


def figure(groups):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for objective, label, colour, marker, style in [("bc", "BC", "#777777", "o", "--"),
                                                   ("bcj", "BC+J", "#2a78d6", "s", "-")]:
        rows = groups[objective]
        seeds = [int(r["seed"]) for r in rows]
        axes[0].plot(seeds, [100 * int(r["student_hold"]) / int(r["student_n"]) for r in rows],
                     color=colour, marker=marker, linestyle=style, label=label)
        axes[1].plot(seeds, [float(r["hold_test_median_damping_diag"]) for r in rows],
                     color=colour, marker=marker, linestyle=style, label=label)
    axes[0].set(title="(a) Full-student validation hold", ylabel="success (%)", ylim=(65, 100))
    axes[1].axhline(-30, color="#52514e", linestyle=":", label="expert −30")
    axes[1].set(title="(b) Hold-test damping diagnostic", ylabel="N·m·s/rad")
    for ax in axes:
        ax.set_xticks([0, 1, 2])
        ax.set_xlabel("training seed")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"jacobian_closed_loop.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("figures/jacobian_closed_loop.{pdf,png}")


if __name__ == "__main__":
    groups = {objective: load_runs(objective) for objective in ROWS}
    audit_pairs(groups["bc"], groups["bcj"])
    tables(groups)
    figure(groups)
