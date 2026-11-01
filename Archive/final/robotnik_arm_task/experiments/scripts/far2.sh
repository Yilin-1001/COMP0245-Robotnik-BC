#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
D=checkpoints/ablation/2026-10-30__training-ablation-v1
run() {  # $1 = output name, rest = model run names
  OUT=results/far_probe/$1; shift
  if [ -f $OUT/summary.json ]; then echo "skip $OUT (done)"; return; fi
  M=""; for R in "$@"; do M="$M $D/$R/$R.npz"; done
  echo "===== $OUT start $(date)"
  python -m evaluation.far_probe --workers 22 --out $OUT --models $M 2>&1 | grep -v -E "No feet|file path|init_motor" | tail -8
}
run v1-R2D-s0 R2D-dagger-R2v1-r3__MLP01-256x3__s0__v1__2026-10-30
run v1-R1D-s12 R1D-dagger-R1v1-r3__MLP01-256x3__s1__v1__2026-10-30 R1D-dagger-R1v1-r3__MLP01-256x3__s2__v1__2026-10-30
run v1-R2D-s12 R2D-dagger-R2v1-r3__MLP01-256x3__s1__v1__2026-10-30 R2D-dagger-R2v1-r3__MLP01-256x3__s2__v1__2026-10-30
echo "===== FAR2 DONE $(date)"
