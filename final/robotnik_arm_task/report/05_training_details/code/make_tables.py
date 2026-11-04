"""Training details: rebuild every table in ../tables from the copied run logs in ../data.

No training runs here. Each run directory under ../data holds the run's metrics.csv (one line per 500 updates: the
mean training losses over those updates, the learning rate and offline validation metrics on the expert's states)
and its resolved config.yaml. Tables:
  recipe_check.csv     every R1-R8 run's resolved recipe, to show that they share one (one row per distinct recipe)
  lambda_j.csv         the calibrated Jacobian-loss weight per training variant
  terminal_metrics.csv per run: terminal training losses, offline validation metrics, best offline arm RMSE, wall time
  jacobian_effect.csv  archived matched pair on clean data: M6 BC vs BC+J, 3 seeds each, terminal damping and errors

Usage (from final/robotnik_arm_task/):
    python report/05_training_details/code/make_tables.py
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]  # report/05_training_details
DATA, OUT = ROOT / "data", ROOT / "tables"
RUN = re.compile(r"^(?P<variant>.+?)__(?P<arch>[^_]+)__s(?P<seed>\d+)__(?P<version>v\d+)__(?P<date>[\d-]+)$")
ROUND = re.compile(r"^(?P<row>.+)-r(?P<round>\d+)$")


def read_metrics(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.update(json.loads(r.pop("extra") or "{}"))
    return rows


def num(v):
    return float(v) if v not in (None, "") else float("nan")


def write_csv(name, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def ablation_runs():
    """[(experiment, run name, row, round, seed, config, metrics)] for every copied R1-R8 run."""
    out = []
    for d in sorted((DATA / "runs").glob("*/*")):
        m = RUN.match(d.name)
        variant = m["variant"]
        r = ROUND.match(variant)
        row, rnd = (r["row"], int(r["round"])) if r else (variant, 0)
        out.append((d.parent.name, d.name, row, rnd, int(m["seed"]), yaml.safe_load((d / "config.yaml").read_text()),
                    read_metrics(d / "metrics.csv")))
    return out


def recipe(config: dict) -> dict:
    t, l, ident = config["training"], config["loss"], config["identity"]
    return {"objective": ident["objective"], "architecture": ident["architecture"],
            "hidden": "x".join(map(str, config["model"]["architectures"][ident["architecture"]])),
            "optimizer": t["optimizer"], "lr": t["lr"], "weight_decay": t["weight_decay"], "batch_size": t["batch_size"],
            "warmup_updates": t["warmup_updates"], "schedule": t["schedule"], "updates": t["updates"],
            "action_loss": l["action"], "huber_delta": l["huber_delta"], "arm_weight": l["arm_weight"],
            "wheel_weight": l["wheel_weight"], "jacobian_subbatch": l["jacobian"]["subbatch_size"],
            "jacobian_kd": l["jacobian"]["kd"], "calibration_ratio": l["jacobian"]["calibration_ratio"],
            "epsilon_tau_nm": l["epsilon_tau"], "arm_scale_floor_nm": config["data"]["arm_scale_floor"],
            "wheel_scale_floor_rad_s": config["data"]["wheel_scale_floor"], "layernorm_eps": config["model"]["layernorm_eps"]}


def recipe_check(runs):
    groups = defaultdict(list)
    for _, name, *_rest in runs:
        groups[json.dumps(recipe(_rest[3]), sort_keys=True)].append(name)
    write_csv("recipe_check.csv", [{"runs": len(names), **json.loads(key)} for key, names in groups.items()])


def lambda_j(runs):
    seen = {}
    for exp, name, row, rnd, seed, config, _ in runs:
        jac = config["loss"]["jacobian"]
        key = name.split("__")[0]
        lam = round(float(jac["lambda_j"]), 6)
        if key in seen:
            assert seen[key]["lambda_j"] == lam, key  # every seed of one variant shares the seed-0 calibration
            seen[key]["seeds"] += 1
            continue
        rec = jac.get("calibration_record") or {}
        seen[key] = {"variant": key, "row": row, "round": rnd, "feature_definition": config["data"]["feature_definition"],
                     "datasets": " + ".join(d.split("__")[0] for d in config["data"]["datasets"]), "lambda_j": lam,
                     "calibration": rec.get("path", ""), "seeds": 1}
    write_csv("lambda_j.csv", list(seen.values()))


def terminal(metrics: list[dict]) -> dict:
    last = metrics[-1]
    best = min(metrics, key=lambda r: num(r["val_arm_rmse_nm"]))
    return {"updates": int(last["update"]), "wall_min": round(num(last["elapsed_s"]) / 60, 2),
            "train_loss": f"{num(last['train_loss']):.3e}", "train_arm_loss": f"{num(last['train_arm_loss']):.3e}",
            "train_wheel_loss": f"{num(last['train_wheel_loss']):.3e}",
            "jacobian_loss": f"{num(last.get('jacobian_loss')):.3e}",
            "val_arm_rmse_nm": round(num(last["val_arm_rmse_nm"]), 3),
            "val_wheel_rmse_rad_s": round(num(last["val_wheel_rmse_rad_s"]), 4),
            "val_hold_arm_rmse_nm": round(num(last["val_hold_arm_rmse_nm"]), 3),
            "val_hold_bias_max_abs_nm": round(num(last["val_hold_bias_max_abs_nm"]), 2),
            "val_damping_median_diag": round(num(last["val_damping_median_diag"]), 2),
            "best_val_arm_rmse_nm": round(num(best["val_arm_rmse_nm"]), 3), "best_at_update": int(best["update"])}


def terminal_metrics(runs):
    rows = [{"experiment": exp, "row": row, "round": rnd, "seed": seed, **terminal(metrics)}
            for exp, name, row, rnd, seed, config, metrics in runs]
    rows.sort(key=lambda r: (r["row"][:2], r["row"], r["round"], r["seed"]))
    write_csv("terminal_metrics.csv", rows)


def jacobian_effect():
    rows = []
    for d in sorted((DATA / "jacobian_effect").glob("*")):
        config = yaml.safe_load((d / "config.yaml").read_text())
        ident = config["identity"]
        rows.append({"objective": ident["objective"], "seed": ident["seed"],
                     "architecture": "x".join(map(str, config["model"]["architectures"][ident["architecture"]])),
                     "lambda_j": (config["loss"].get("jacobian") or {}).get("lambda_j", 0.0) or 0.0,
                     **terminal(read_metrics(d / "metrics.csv")), "run": d.name})
    write_csv("jacobian_effect.csv", rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    runs = ablation_runs()
    recipe_check(runs)
    lambda_j(runs)
    terminal_metrics(runs)
    jacobian_effect()
