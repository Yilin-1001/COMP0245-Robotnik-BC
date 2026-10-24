"""DAgger: roll out the learner (mixed with the expert at probability beta),
have the expert relabel visited states, aggregate and retrain.

Output: data/D_round{N}.npz, checkpoints/mlp_r{N}.pt

Usage: python -m data_collection.dagger --rounds 4
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--episodes-per-round", type=int, default=100)
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
