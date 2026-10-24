"""Train ACT (optional): 5-10 step state history, chunks of 10-20 actions, small KL weight.

Output: checkpoints/act_*.pt

Usage: python -m training.train_act --data data/D0.npz --out checkpoints/act_r0.pt
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/D0.npz"])
    parser.add_argument("--out", default="checkpoints/act_r0.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
