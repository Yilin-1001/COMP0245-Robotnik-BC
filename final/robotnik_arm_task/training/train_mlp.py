"""Train the BC-MLP: Huber loss, AdamW, cosine schedule.

Checkpoints are chosen by closed-loop validation success, not validation loss.
Output: checkpoints/mlp_*.pt

Usage: python -m training.train_mlp --data data/D0.npz --out checkpoints/mlp_r0.pt
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data/D0.npz"])
    parser.add_argument("--out", default="checkpoints/mlp_r0.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
