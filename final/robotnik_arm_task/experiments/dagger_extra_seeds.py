"""Extra training seeds for early DAgger rounds of ablation v2, on the existing round datasets (no new collection).

The 2026-11-01 v2 run (experiments.dagger_rounds) trained seeds 1-2 only for the chosen round (r3). This script trains
seeds for the earlier rounds k on the same aggregated datasets (dagger-R<n>-v2-r1..rk), with the same recipe, so that
"DART + k DAgger rounds" (R8-rk) and "DAgger only, k rounds" (R6-rk) can be compared at k = 1, 2 with three seeds.
The DAgger data of round k was collected by the 2026-11-01 seed-0 model of round k - 1; that model stays seed 0.

Models: checkpoints/ablation/<date>__training-ablation-v2/; results: results/ablation/<date>__training-ablation-v2/dagger/<row>/.

Usage (from final/robotnik_arm_task/):
    python -m experiments.dagger_extra_seeds --date 2026-11-03 --rows R8-dart-gate-dagger R6-gate-dagger \
        --rounds 1 2 --seeds 1 2 --workers 20
"""
import argparse
import time
from pathlib import Path

from experiments.dagger_rounds import ROWS, TASK_DIR, VERSION, train_and_validate

SOURCE_DATE = "2026-11-01"  # the v2 run whose DAgger datasets and seed-0 collectors are reused


def collector(row: str, k: int) -> str:
    """The seed-0 model that collected round k's data, as dagger_rounds records it."""
    if k == 1:
        base = ROWS[row][0]
        return f"{base}__MLP01-256x3__s0__v1__2026-10-28.npz"
    return f"{row}-r{k - 1}__MLP01-256x3__s0__{VERSION}__{SOURCE_DATE}.npz"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", required=True, help="date of this supplementary experiment, YYYY-MM-DD")
    p.add_argument("--rows", nargs="+", choices=list(ROWS), required=True)
    p.add_argument("--rounds", nargs="+", type=int, required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--workers", type=int, default=20)
    args = p.parse_args(argv)
    if args.date == SOURCE_DATE:
        raise SystemExit("use a new date; the 2026-11-01 run is the reference record")
    for k in args.rounds:
        for row in args.rows:
            out_dir = TASK_DIR / f"results/ablation/{args.date}__training-ablation-{VERSION}/dagger/{row}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                print(f"\n===== {row} round {k} seed {seed} ({time.strftime('%H:%M:%S')}) =====", flush=True)
                train_and_validate(row, k, seed, args.date, collector(row, k), args.workers, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
