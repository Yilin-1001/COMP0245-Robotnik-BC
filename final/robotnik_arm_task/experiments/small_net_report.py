"""Summary of the small-network DART + DAgger runs (experiments.dagger_arch_rounds): size, speed and success.

One line per architecture: parameters, FLOPs, .npz size, isolated single-thread latency (evaluation.latency_bench),
training time per model, the wall-clock cost of the whole pipeline up to the chosen round, validation of the chosen
round (3 seeds) and success on the 240-scene unified test (evaluation.test_suite), pooled over the 3 seeds, in the
groups of the low-round report: in range (T0+T4+T5), range edges (T2+T3), far (T1, 3-15 m) and all 240.

Usage (from final/robotnik_arm_task/):
    python -m experiments.small_net_report --date 2026-11-04 --list-models [--seeds 0 1 2]   # step_42000 paths
    python -m experiments.small_net_report --date 2026-11-04 [--test results/test/<run>] [--latency <latency.csv>]
"""
import argparse
import csv
import json
import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
ORDER = ["MLP02", "MLP04", "MLP06", "MLP01"]
GROUPS = {"in range (T0+T4+T5)": {"T0", "T4", "T5"}, "edges (T2+T3)": {"T2", "T3"}, "far (T1)": {"T1"},
          "all 240": {"T0", "T1", "T2", "T3", "T4", "T5"}}
STEPS = ["collect", "build", "train", "validate"]


def _csv(path: Path) -> list[dict]:
    return list(csv.DictReader(open(path))) if path.exists() else []


def _frac(text: str) -> tuple[int, int]:
    a, b = text.split("/")
    return int(a), int(b)


def chosen_round(summary: list[dict]) -> int | None:
    """Seed-0 validation over rounds >= 1, a tie goes to the later round (as dagger_rounds and the test suite)."""
    cands = [r for r in summary if int(r["round"]) > 0 and int(r["seed"]) == 0]
    if not cands:
        return None
    return int(max(cands, key=lambda r: (_frac(r["sustained_hold"])[0] / _frac(r["sustained_hold"])[1], int(r["round"])))["round"])


def arch_dirs(date: str) -> dict:
    root = TASK_DIR / f"results/ablation/{date}__small-net-dagger-v2"
    found = {d.name: d for d in sorted(root.glob("MLP*")) if d.is_dir()} if root.exists() else {}
    return {a: found[a] for a in sorted(found, key=lambda a: ORDER.index(a) if a in ORDER else 99)}


def chosen_models(date: str, seeds: list[int]) -> list[Path]:
    out = []
    for arch, d in arch_dirs(date).items():
        summary = _csv(d / "summary.csv")
        k = chosen_round(summary)
        for r in summary:
            if k is not None and int(r["round"]) == k and int(r["seed"]) in seeds:
                step = TASK_DIR / Path(r["model"]).parent / "step_42000.npz"
                if step.exists():
                    out.append(step)
    return out


def arch_of(model_id: str) -> str | None:
    m = re.search(r"__(MLP\d\d)-", model_id)
    return m[1] if m else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True)
    p.add_argument("--list-models", action="store_true", help="print the chosen round's step_42000 models and stop")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--test", type=Path, help="unified test run directory (rollouts.csv)")
    p.add_argument("--latency", type=Path, help="evaluation.latency_bench CSV")
    args = p.parse_args(argv)
    if args.list_models:
        print(" ".join(str(m.relative_to(TASK_DIR)) for m in chosen_models(args.date, args.seeds)))
        return 0
    root = TASK_DIR / f"results/ablation/{args.date}__small-net-dagger-v2"
    latency_path = args.latency or root / "latency.csv"
    latency = {row["architecture"].split("-")[0]: row for row in _csv(latency_path)}
    tests = _csv(args.test / "rollouts.csv") if args.test else []
    lines = []
    for arch, d in arch_dirs(args.date).items():
        summary, cost = _csv(d / "summary.csv"), _csv(d / "cost.csv")
        k = chosen_round(summary)
        lat = latency.get(arch, {})
        trains = [float(r["train_elapsed_s"]) for r in cost if r["step"] == "train" and r["train_elapsed_s"]]
        trains.sort()

        def wall(rows):
            return sum(float(r["wall_s"]) for r in rows) / 60

        chain = [r for r in cost if r["seed"] == "0"]
        line = {"arch": arch, "parameters": lat.get("parameters", ""), "flops": lat.get("flops", ""),
                "npz_kb": round(int(lat["npz_bytes"]) / 1024, 1) if lat.get("npz_bytes") else "",
                "latency_p50_us": round(float(lat["median_us"]), 1) if lat.get("median_us") else "",
                "latency_p99_us": round(float(lat["p99_us"]), 1) if lat.get("p99_us") else "",
                "train_min_median": round(trains[len(trains) // 2] / 60, 2) if trains else "",
                "device": next((r["device"] for r in cost if r["device"]), ""),
                "chosen_round": k if k is not None else "",
                "pipeline_min_to_chosen": round(wall([r for r in chain if k is not None and int(r["round"]) <= k]), 1) if cost else "",
                "pipeline_min_all_rounds": round(wall(chain), 1) if cost else "",
                **{f"{s}_min_total": round(wall([r for r in cost if r["step"] == s]), 1) for s in STEPS}}
        vals = [r for r in summary if k is not None and int(r["round"]) == k]
        holds = [_frac(r["sustained_hold"]) for r in sorted(vals, key=lambda r: int(r["seed"]))]
        line["validation_per_seed"] = " / ".join(str(a) for a, _ in holds)
        line["validation_pooled"] = f"{sum(a for a, _ in holds)}/{sum(b for _, b in holds)}" if holds else ""
        line["validation_by_round_s0"] = " / ".join(r["sustained_hold"] for r in sorted(
            (r for r in summary if r["seed"] == "0"), key=lambda r: int(r["round"])))
        mine = [r for r in tests if arch_of(r["model_id"]) == arch]
        for g, suites in GROUPS.items():
            rows = [r for r in mine if r["suite"] in suites]
            ok = sum(r["success"] in ("True", "true", "1") for r in rows)
            line[f"test {g}"] = f"{100 * ok / len(rows):.1f}% ({ok}/{len(rows)})" if rows else ""
        lines.append(line)
    if tests:
        expert = [r for r in tests if r["row"] == "expert"]
        e = {g: sum(r["success"] in ("True", "true", "1") for r in expert if r["suite"] in s) for g, s in GROUPS.items()}
        n = {g: sum(r["suite"] in s for r in expert) for g, s in GROUPS.items()}
        lines.append({"arch": "teacher A", **{f"test {g}": f"{100 * e[g] / n[g]:.1f}% ({e[g]}/{n[g]})" for g in GROUPS if n[g]}})
    if not lines:
        raise SystemExit(f"nothing found under {root}")
    root.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(f for line in lines for f in line))
    with open(root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(lines)
    cols = [("Network", "arch"), ("Params", "parameters"), ("FLOPs", "flops"), (".npz KB", "npz_kb"),
            ("Latency p50 / p99 (µs)", None), ("Train min / model", "train_min_median"),
            ("Pipeline min, all rounds", "pipeline_min_all_rounds"), ("Chosen round", "chosen_round"),
            ("Pipeline min to chosen round", "pipeline_min_to_chosen"),
            ("Validation s0 by round", "validation_by_round_s0"), ("Validation chosen (s0/s1/s2)", "validation_per_seed"),
            ("Validation pooled", "validation_pooled")] + [(f"Test {g}", f"test {g}") for g in GROUPS]
    md = ["# Small-network DART + DAgger (R8 end to end)", "",
          f"Date {args.date}. Pipeline minutes: seed-0 chain (collect + build + train + validate), all rounds (the real "
          "cost: the chosen round is only known after the last round) and, for reference, up to the chosen round. Latency: isolated single-thread NumPy predict(), batch 1. Test: 240-scene "
          "unified suite, 3 seeds pooled.", "",
          "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for line in lines:
        cells = []
        for _, key in cols:
            if key is None:
                cells.append(f"{line.get('latency_p50_us', '')} / {line.get('latency_p99_us', '')}" if line.get("latency_p50_us") != "" and "latency_p50_us" in line else "")
            else:
                cells.append(str(line.get(key, "")))
        md.append("| " + " | ".join(cells) + " |")
    devices = sorted({line.get("device") for line in lines if line.get("device")})
    md += ["", f"Training device: {', '.join(devices) or 'unknown'}. Inputs: `results/ablation/{args.date}__small-net-dagger-v2/"
           "<arch>/{summary,cost}.csv`" + (f", `{args.test}`" if args.test else "") + "."]
    (root / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwritten {root / 'summary.md'} and summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
