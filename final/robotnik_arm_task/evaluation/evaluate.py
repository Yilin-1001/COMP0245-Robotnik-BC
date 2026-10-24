"""Closed-loop evaluation. Appends one row per run to results/results.csv.

Usage: python -m evaluation.evaluate --policy expert --split val
       python -m evaluation.evaluate --policy mlp --ckpt checkpoints/mlp_r3.pt --split T1
"""
import argparse

RESULTS_CSV = "results/results.csv"
COLUMNS = ["run_id", "policy", "ckpt", "split", "n", "success_10mm", "time_to_reach",
           "final_err", "path_len", "torque_jerk", "finger_drift"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["expert", "mlp", "act"], required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", choices=["val", "T1", "T2", "T3"], default="val")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
