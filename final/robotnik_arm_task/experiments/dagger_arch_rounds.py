"""R8 (DART + DAgger) with a small network end to end, with the cost of every step recorded.

The chosen architecture is the round-0 student (trained on R4's data: clean253 + DART), it collects every DAgger round
itself, and it is retrained from scratch on the aggregate after each round. Contrast: the architecture ablation
(experiments/ablation/2026-11-02__architecture-ablation-v1.yaml) retrains each architecture on the round-3 aggregate
that the 256x3 chain collected. Here the collector is always the small network, so the question is whether a small
network needs (or profits from) correcting its own mistakes.

Everything else is R8's protocol (experiments/dagger_rounds.py): beta = 0, teacher A labels at 1 kHz, the 221 fitting
scenes plus 64 development scenes per round (the same scenes for every architecture), training from scratch, the
round chosen on seed-0 validation (a tie goes to the later round), then extra seeds for that round. The learning rate
is the architecture's lr_by_architecture entry of the architecture-ablation manifest (1e-3 if absent); lambda_J is
calibrated per run, as in the 256x3 DAgger runs.

Outputs (per architecture): results/ablation/<date>__small-net-dagger-v2/<arch>/
    summary.csv   validation per (round, seed), as dagger_rounds writes it
    cost.csv      one row per step that actually ran: collect / build / train / validate wall seconds, the training's
                  own elapsed time (metrics.csv), device, episodes and rows of the new DAgger round and of the whole
                  training set
    logs/, validation/
Models: checkpoints/ablation/<date>__training-ablation-v2/. Every step is skipped when its output exists, so an
interrupted run continues where it stopped (finished steps keep their recorded cost).

Usage (from final/robotnik_arm_task/):
    python -m experiments.dagger_arch_rounds --arch MLP04 --date 2026-11-04 [--rounds 3] [--final-seeds 1 2] [--workers 20]
"""
import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import yaml

from experiments import dagger_rounds as dr

TASK_DIR = dr.TASK_DIR
ROW, BASE = "R8-dart-gate-dagger", "R4-dart-gate"
_, DEFINITION, OWN = dr.ROWS[ROW]
VERSION = dr.VERSION
EXPERIMENT = "small-net-dagger"
MANIFEST = TASK_DIR / "experiments/ablation/2026-11-02__architecture-ablation-v1.yaml"
TAG = ""  # set by --tag; empty keeps the original names
COST_FIELDS = ["arch", "round", "seed", "step", "wall_s", "train_elapsed_s", "device", "new_episodes", "new_rows",
               "train_episodes", "train_rows", "model", "detail", "written"]


def registry() -> dict:
    return yaml.safe_load((TASK_DIR / "experiments/configs/datasets/base.yaml").read_text())["model"]["architectures"]


def label(arch: str) -> str:
    from training.datasets.train import architecture_label
    return architecture_label(arch, registry()[arch])


def learning_rate(arch: str) -> str:
    lrs = (yaml.safe_load(MANIFEST.read_text()) or {}).get("lr_by_architecture") or {} if MANIFEST.exists() else {}
    return str(lrs.get(arch, "1.0e-3"))


def dataset(arch: str, k: int) -> str:
    return f"dagger-R8-{VERSION}-{arch}{TAG}-r{k}"


def variant_name(arch: str, k: int) -> str:
    return BASE if k == 0 else f"{ROW}-{arch}{TAG}-r{k}"


def out_dir_for(date: str, arch: str, tag: str = None) -> Path:
    tag = TAG if tag is None else tag
    return TASK_DIR / f"results/ablation/{date}__{EXPERIMENT}{'-' + tag if tag else ''}-{VERSION}/{arch}"


def write_variant(arch: str, k: int) -> str:
    name = variant_name(arch, k)
    if k == 0:
        return name
    datasets = [f"{d}__{DEFINITION}" for d in OWN] + [f"{dataset(arch, i)}__{DEFINITION}" for i in range(1, k + 1)]
    path = TASK_DIR / "experiments/configs/datasets/variants" / f"{name}.yaml"
    text = (f"# {ROW} with {arch} end to end, DAgger round {k} ({VERSION}): {BASE}'s data ({', '.join(OWN)}) plus the\n"
            f"# DAgger rounds 1..{k} that the {arch} students collected. Written by experiments/dagger_arch_rounds.py.\n"
            f"data:\n  feature_definition: {DEFINITION}\n"
            f"  datasets: [{', '.join(datasets)}]\n  validation_dataset: clean253__{DEFINITION}\n")
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return name


def model_path(arch: str, k: int, seed: int, date: str) -> Path:
    run_id = f"{variant_name(arch, k)}__{label(arch)}__s{seed}__{VERSION}__{date}"
    return TASK_DIR / f"checkpoints/ablation/{date}__training-ablation-{VERSION}/{run_id}/{run_id}.npz"


def dataset_complete(name: str) -> bool:
    return (TASK_DIR / "data/datasets/A" / f"{name}__{DEFINITION}" / "COMPLETE.json").exists()


def device_name() -> str:
    try:
        import torch
        return f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal
        return f"unknown ({exc.__class__.__name__})"


class Cost:
    """cost.csv, one row per (round, seed, step); a re-run step replaces its row."""

    def __init__(self, out_dir: Path, arch: str, device: str):
        self.path, self.arch, self.device = out_dir / "cost.csv", arch, device

    def add(self, k: int, seed: int, step: str, wall_s: float, **extra):
        rows = list(csv.DictReader(open(self.path))) if self.path.exists() else []
        rows = [r for r in rows if not (r["round"] == str(k) and r["seed"] == str(seed) and r["step"] == step)]
        rows.append({f: "" for f in COST_FIELDS} | {"arch": self.arch, "round": k, "seed": seed, "step": step,
                    "wall_s": round(wall_s, 1), "device": self.device, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
                    | {key: value for key, value in extra.items() if key in COST_FIELDS})
        rows.sort(key=lambda r: (int(r["round"]), int(r["seed"]), ["collect", "build", "train", "validate"].index(r["step"])))
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COST_FIELDS)
            w.writeheader()
            w.writerows(rows)


def train(arch: str, k: int, seed: int, date: str, out_dir: Path, cost: Cost) -> Path:
    model = model_path(arch, k, seed, date)
    if model.exists():
        return model
    shutil.rmtree(model.parent, ignore_errors=True)  # a stopped run leaves a partial directory the trainer refuses
    name = write_variant(arch, k)
    cmd = ["-m", "training.datasets.train", "--variant", name, "--set", f"identity.date={date}",
           "--set", f"identity.seed={seed}", "--set", f"identity.architecture={arch}"]
    for key, value in {**dr.RECIPE, "training.lr": learning_rate(arch)}.items():
        cmd += ["--set", f"{key}={value}"]
    t0 = time.time()
    dr.run(cmd, out_dir / "logs" / f"r{k}-s{seed}-train.log")
    wall = time.time() - t0
    elapsed = ""
    metrics = model.parent / "metrics.csv"
    if metrics.exists():
        rows = list(csv.DictReader(open(metrics)))
        elapsed = rows[-1].get("elapsed_s", "") if rows else ""
    config = yaml.safe_load((model.parent / "config.yaml").read_text())
    used = (config.get("provenance") or {}).get("datasets") or {}
    new = used.get(f"{dataset(arch, k)}__{DEFINITION}", {}) if k else {}
    cost.add(k, seed, "train", wall, train_elapsed_s=elapsed, model=model.stem,
             train_episodes=sum(v["episodes"] for v in used.values()), train_rows=sum(v["rows"] for v in used.values()),
             new_episodes=new.get("episodes", ""), new_rows=new.get("rows", ""),
             detail=json.dumps({"lr": learning_rate(arch), "lambda_j": config["loss"]["jacobian"].get("lambda_j")}))
    return model


def validate(arch: str, k: int, seed: int, date: str, model: Path, collector: str, workers: int, out_dir: Path,
             cost: Cost) -> dict:
    cached = (out_dir / "validation" / f"{model.stem}.json").exists()
    t0 = time.time()
    res = dr.train_and_validate(f"{ROW}-{arch}", k, seed, date, collector, workers, out_dir, model=model)
    if not cached:
        cost.add(k, seed, "validate", time.time() - t0, model=model.stem, detail=json.dumps({"hold": res["sustained_hold"]}))
    return res


def reuse_round0(arch: str, date: str, out_dir: Path, cost: Cost) -> None:
    """Copy the round-0 cost rows and validation of the untagged run of `date` (same model, nothing retrained)."""
    src = out_dir_for(date, arch, tag="")
    model = model_path(arch, 0, 0, date)
    if not model.exists():
        raise SystemExit(f"--round0-date {date}: {model} not found")
    have = {(r["round"], r["seed"], r["step"]) for r in csv.DictReader(open(cost.path))} if cost.path.exists() else set()
    for r in (csv.DictReader(open(src / "cost.csv")) if (src / "cost.csv").exists() else []):
        if r["round"] == "0" and r["seed"] == "0" and ("0", "0", r["step"]) not in have:
            cost.add(0, 0, r["step"], float(r["wall_s"]), **{k: r[k] for k in COST_FIELDS if k not in
                     ("arch", "round", "seed", "step", "wall_s", "device", "written")} | {"detail": f"reused from {src}"})
    cached = src / "validation" / f"{model.stem}.json"
    if cached.exists():
        (out_dir / "validation").mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, out_dir / "validation" / cached.name)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--arch", required=True, help="registered architecture, e.g. MLP04 (64x3)")
    p.add_argument("--date", required=True, help="experiment date YYYY-MM-DD, fixed for the whole experiment")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--dev-scenes", type=int, default=64)
    p.add_argument("--final-seeds", type=int, nargs="*", default=[1, 2], help="extra seeds for the chosen round")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--tag", default="", help="letters/digits added to dataset, variant and result names, e.g. nodev")
    p.add_argument("--round0-date", help="reuse the round-0 model (and its recorded cost and validation) of the "
                                         "untagged run of this date instead of training it again")
    args = p.parse_args(argv)
    global TAG
    if args.tag and not args.tag.isalnum():
        raise SystemExit("--tag must be letters and digits only")
    TAG = args.tag
    if args.arch not in registry():
        raise SystemExit(f"unknown architecture {args.arch}; registered: {', '.join(registry())}")
    out_dir = out_dir_for(args.date, args.arch)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = Cost(out_dir, args.arch, device_name())
    print(f"\n===== {args.arch} ({label(args.arch)}, lr {learning_rate(args.arch)}, {cost.device}) round 0: {BASE} "
          f"(dev scenes per round: {args.dev_scenes}) =====", flush=True)
    r0_date = args.round0_date or args.date
    if args.round0_date:
        reuse_round0(args.arch, r0_date, out_dir, cost)
    student = train(args.arch, 0, 0, r0_date, out_dir, cost)
    validate(args.arch, 0, 0, r0_date, student, "-", args.workers, out_dir, cost)
    results = []
    for k in range(1, args.rounds + 1):
        print(f"\n===== {args.arch} DAgger round {k}: collector {student.name} =====", flush=True)
        name = dataset(args.arch, k)
        if not dataset_complete(name):
            t0 = time.time()
            out = dr.run(["-m", "data_collection.collect_dagger", "--student", str(student), "--round", f"r{k}",
                          "--dev-scenes", str(args.dev_scenes), "--workers", str(args.workers)],
                         out_dir / "logs" / f"r{k}-collect.log")
            collected = {key: v for key, v in dr.last_json(out).items() if key != "root"}
            cost.add(k, 0, "collect", time.time() - t0, model=student.stem, detail=json.dumps(collected))
            print(f"  collected: {json.dumps(collected)}", flush=True)
            t0 = time.time()
            dr.run(["-m", "training.datasets.build", "--source", dr.last_json(out)["root"], "--name", name,
                    "--features", DEFINITION, "--workers", str(args.workers)], out_dir / "logs" / f"r{k}-build.log")
            cost.add(k, 0, "build", time.time() - t0)
        model = train(args.arch, k, 0, args.date, out_dir, cost)
        results.append(validate(args.arch, k, 0, args.date, model, student.name, args.workers, out_dir, cost))
        student = model
    if args.final_seeds and results:
        best = max(results, key=lambda r: (r["hold"], r["round"]))
        print(f"\n===== {args.arch}: extra seeds for round {best['round']} (seed-0 validation {best['sustained_hold']}) =====",
              flush=True)
        for seed in args.final_seeds:
            model = train(args.arch, best["round"], seed, args.date, out_dir, cost)
            validate(args.arch, best["round"], seed, args.date, model, best["collector"], args.workers, out_dir, cost)
    print(f"\nSummary: {out_dir / 'summary.csv'}\nCost: {cost.path}")
    print((out_dir / "summary.csv").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
