"""Deprecated: use `python -m training.baseline.train` (clean BC) or `python -m training.jacobian.train` (BC+J).

The old --data/--out arguments referred to a single pre-strided D0.npz and a fixed checkpoint path; they are not
reinterpreted. This wrapper forwards new-style arguments to the baseline entry point unchanged.
"""
import sys

MESSAGE = ("training.train_mlp is deprecated: run `python -m training.baseline.train --teacher A|B "
           "[--set key=value ...]` (see README). The old --data/--out arguments are not supported.")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or any(a.split("=")[0] in ("--data", "--out", "--seed") for a in argv):
        print(MESSAGE, file=sys.stderr)
        raise SystemExit(2)
    print(MESSAGE, file=sys.stderr)
    from training.baseline.train import main as baseline_main
    return baseline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
