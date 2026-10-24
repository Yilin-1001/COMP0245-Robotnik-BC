"""Run the expert headless on training scenarios with DART noise.

Noise is added to the executed action; the clean expert action is stored as the label.
Output: data/D0.npz

Usage: python -m data_collection.collect --n 512 --seed-start 0
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", default="data/D0.npz")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
