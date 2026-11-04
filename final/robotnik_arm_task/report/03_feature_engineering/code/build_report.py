"""Rebuild feature-engineering evidence from frozen source and data snapshots.

Run from final/robotnik_arm_task/: python report/03_feature_engineering/code/build_report.py
No training, simulator rollout or test-set trajectory is used.
"""
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "code/source_snapshot"))

from core import feature_sets, features
from core.interfaces import Observation
from training.common.dataset import FeatureBuilder
from training.common.policies import MLPPolicy


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (ROOT / "tables" / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "figures" / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def check_representation():
    """Synthetic geometry checks, separate from historical empirical evidence."""
    rng = np.random.default_rng(20261104)
    n = 256
    base = np.c_[rng.uniform(-.5, .5, (n, 2)), np.zeros(n)]
    yaw = rng.uniform(-np.pi, np.pi, n)
    q, qd = rng.normal(0, .3, (n, 7)), rng.normal(0, .1, (n, 7))
    target = np.c_[rng.uniform(-3, 3, (n, 2)), rng.uniform(.1, 1.25, n)]
    fk = features.DeployableFK()
    x = features.features_a(base, yaw, q, qd, target, fk)
    theta = .91
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved_base, moved_target = base.copy(), target.copy()
    moved_base[:, :2] = base[:, :2] @ rotation.T + [1.7, -.8]
    moved_target[:, :2] = target[:, :2] @ rotation.T + [1.7, -.8]
    moved = features.features_a(moved_base, yaw + theta, q, qd, moved_target, fk)
    invariance_error = float(np.max(np.abs(x - moved)))
    assert invariance_error < 1e-12
    extended = feature_sets.extend("A", "features-v1-gate", x)
    assert extended.shape == (n, 22) and np.array_equal(extended[:, :21], x)
    builder = FeatureBuilder("A", definition="features-v1-gate")
    meta = {"teacher": "A", "input_dim": 22, "feature_definition": "features-v1-gate",
            "interface": {"torque_limit": 30, "wheel_limit": 6, "control_hz": 1000}}
    norm = {"x_mean": np.zeros(22), "x_scale": np.ones(22), "y_mean": np.zeros(9), "y_scale": np.ones(9)}
    policy = MLPPolicy(None, norm, meta, "numpy", "report-feature-check")
    differences = []
    for i in range(16):
        offline = builder({"obs_base_pos": base[i:i+1], "obs_base_yaw": yaw[i:i+1],
                           "obs_q_mes": q[i:i+1], "obs_qd_mes": qd[i:i+1]},
                          {"target_ee_pos_W": target[i]})[0]
        online = policy.features(Observation(base[i], yaw[i], q[i], qd[i], target[i]))
        differences.append(float(np.max(np.abs(offline - online))))
    assert max(differences) == 0
    rows = [{"check": "planar_translation_and_yaw_invariance", "observations": n,
             "max_abs_error": invariance_error, "tolerance": 1e-12},
            {"check": "gate_preserves_first_21_features", "observations": n,
             "max_abs_error": 0.0, "tolerance": 0.0},
            {"check": "offline_builder_equals_online_policy_features", "observations": 16,
             "max_abs_error": max(differences), "tolerance": 0.0}]
    write_csv("representation_checks.csv", rows)
    return rows


def evidence_tables():
    names = feature_sets.feature_names("A", "features-v1-gate")
    assert (feature_sets.input_dim("A"), len(names), feature_sets.input_dim("B")) == (21, 22, 26)
    rows = []
    for i, name in enumerate(names):
        block = "joint_position" if i < 7 else "joint_velocity" if i < 14 else \
                "relative_target" if i < 17 else "fk_target_error" if i < 20 else \
                "world_target_height" if i == 20 else "reach_gate"
        rows.append({"index_0based": i, "feature": name, "block": block,
                     "unit": "rad" if i < 7 else "rad/s" if i < 14 else "m" if i < 21 else "binary",
                     "in_features_v1": i < 21, "in_features_v1_gate": True})
    write_csv("feature_schema.csv", rows)
    qa_rows = []
    for path in sorted((DATA / "dataset_qa").glob("*.json")):
        qa = json.loads(path.read_text())
        assert qa["passed"] and qa["finite"] and qa["feature_dim"] == qa["expected_dim"]
        gate = qa.get("reach_gate_vs_teacher_flag")
        prefix = qa["v1_prefix_equal_to_sibling"]
        if gate:
            assert gate["mismatched_rows"] == 0 and prefix["equal"]
        qa_rows.append({"dataset": path.stem, "episodes": qa["episodes"], "rows": qa["rows"],
                        "inputs": qa["feature_dim"], "passed": qa["passed"],
                        "gate_mismatches": gate["mismatched_rows"] if gate else "not_applicable",
                        "v1_prefix_equal": prefix["equal"] if isinstance(prefix, dict) else "not_checked"})
    write_csv("dataset_qa.csv", qa_rows)
    validation = read_csv(DATA / "validation_v1_summary.csv")
    gate_rows = []
    for row in validation:
        counts = [int(v) for v in row["student_hold_per_seed"].split("/")]
        holds = sum(counts)
        rate = holds / (32 * len(counts))
        assert np.isclose(rate, float(row["student_hold_rate"]))
        gate_rows.append({"row": row["row"], "data": "Clean" if row["row"].startswith(("R1", "R2")) else "OU",
                          "gate": row["row"].startswith(("R2", "R4")), "holds": holds,
                          "rollouts": 32 * len(counts), "hold_rate": rate,
                          "hold_per_seed": row["student_hold_per_seed"]})
    write_csv("gate_validation.csv", gate_rows)
    fk = json.loads(gzip.decompress((DATA / "fk/fk_accuracy.json.gz").read_bytes()))
    contract = (DATA / "fk/fk_accuracy_contract.json").read_bytes()
    assert hashlib.sha256(contract).hexdigest() == fk["contract_sha256"]
    errors, fk_rows = {}, []
    for role in ("expert", "student"):
        selected = [r for r in fk["results"] if r["role"] == role]
        assert len(selected) == 4
        error = np.array([s["fk_error_mm"] for r in selected for s in r["samples"]])
        errors[role] = error
        stats = {"role": role, "trajectories": len(selected), "sampled_observations": len(error),
                 "median_mm": float(np.median(error)), "p95_mm": float(np.percentile(error, 95)),
                 "sampled_max_mm": float(error.max()), "fraction_above_10mm": float(np.mean(error > 10))}
        assert np.isclose(stats["median_mm"], fk["by_role"][role]["all"]["median_fk_error_mm"])
        assert np.isclose(stats["p95_mm"], fk["by_role"][role]["all"]["p95_fk_error_mm"])
        fk_rows.append(stats)
    write_csv("fk_accuracy.csv", fk_rows)
    return gate_rows, errors


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 2.1))
    ax.set(xlim=(0, 10.5), ylim=(0, 2.1)); ax.axis("off")
    boxes = [(0, 1.9, "Observation", r"$p_b,\ \psi,\ q,\ \dot q,\ p_t$"),
             (2.4, 2.6, "Geometry (21)", r"$q,\ \dot q,\ t_b,\ e_b,\ z_t$"),
             (5.5, 2.1, "Gate (22)", r"$[x_{21},\ g]$"),
             (8.1, 2.3, "Standardisation", r"$(x-\mu_{\rm fit})/s_{\rm fit}$")]
    for x, width, title, formula in boxes:
        ax.add_patch(Rectangle((x, .45), width, 1.1, fill=False, linewidth=.7, edgecolor=".25"))
        ax.text(x + width / 2, 1.18, title, ha="center", va="center")
        ax.text(x + width / 2, .8, formula, ha="center", va="center", fontsize=11)
    for left, right in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((left[0] + left[1] + .04, 1), (right[0] - .04, 1),
                                     arrowstyle="->", mutation_scale=10, linewidth=.7))
    save(fig, "feature_pipeline")


def draw_gate():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    rule = feature_sets.GATE_RULE
    axes[0].plot(rule["standoff_z_points"], rule["standoff_points"], "o-", color="#365c82", lw=1, ms=3)
    axes[0].set(xlabel=r"$z_t$ (m)", ylabel=r"$s(z_t)$ (m)", title="(a) Standoff distance")
    d, h = np.meshgrid(np.linspace(-.2, .2, 401), np.linspace(-.15, .15, 401))
    x = np.zeros((*d.shape, 21))
    planar = .65 + d
    x[..., 14], x[..., 15], x[..., 20] = planar * np.cos(h), planar * np.sin(h), .6
    gate = feature_sets.reach_gate(x)
    axes[1].contourf(d, h, gate, levels=[-.5, .5, 1.5], colors=["white", "#dbe5ed"])
    dt, ht = rule["dist_tol_m"], rule["heading_tol_rad"]
    axes[1].add_patch(Rectangle((-dt, -ht), 2 * dt, 2 * ht, fill=False, linewidth=.8, linestyle="--", edgecolor="#365c82"))
    axes[1].text(0, 0, r"$g=1$", ha="center", va="center")
    axes[1].set(xlabel=r"$d_{\rm err}$ (m)", ylabel=r"$h$ (rad)", title="(b) Reach gate",
                xticks=[-.12, 0, .12], yticks=[-.075, 0, .075])
    save(fig, "reach_gate")


def draw_evidence(gate_rows, errors):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for role, color in (("expert", "#365c82"), ("student", "#ac713c")):
        error = np.sort(errors[role])
        axes[0].plot(error, np.arange(1, len(error) + 1) / len(error), label=role.capitalize(), color=color, lw=1)
    axes[0].set(xlabel="FK position error (mm)", ylabel="Empirical CDF", title="(a) FK diagnostic", xlim=(0, 6), ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)
    for gate, offset, color, label in ((False, -.18, "#aebdcb", "21 inputs"), (True, .18, "#365c82", "22 inputs (+ gate)")):
        selected = [r for r in gate_rows if r["gate"] == gate]
        rates = [100 * r["hold_rate"] for r in selected]
        bars = axes[1].bar(np.arange(2) + offset, rates, width=.34, color=color, label=label)
        for bar, row in zip(bars, selected):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                         f"{row['holds']}/{row['rollouts']}", ha="center", fontsize=8)
    axes[1].set(xticks=[0, 1], xticklabels=["Clean", "OU"], ylabel="Validation hold (%)", ylim=(0, 105), title="(b) Gate comparison")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "feature_evidence")


def main():
    inventory = json.loads((DATA / "source_inventory.json").read_text())
    for entry in inventory:
        raw = (ROOT / entry["copy"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        original = gzip.decompress(raw) if entry["gzip"] else raw
        assert hashlib.sha256(original).hexdigest() == entry["source_sha256"]
    for folder in ("tables", "figures"):
        (ROOT / folder).mkdir(exist_ok=True)
    checks = check_representation()
    gate_rows, errors = evidence_tables()
    with plt.rc_context({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
                         "mathtext.fontset": "stix", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6}):
        draw_pipeline(); draw_gate(); draw_evidence(gate_rows, errors)
    (DATA / "verification.json").write_text(json.dumps({"copied_files_verified": len(inventory),
        "synthetic_seed": 20261104, "checks": checks}, indent=2) + "\n")
    print(f"Verified {len(inventory)} source/evidence hashes, planar invariance and offline/online equality.")
    print("Recomputed historical FK and gate evidence; wrote 5 tables and 3 PNG/PDF figure pairs.")


if __name__ == "__main__":
    main()
