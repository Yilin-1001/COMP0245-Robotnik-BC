#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
L=checkpoints/ablation/2026-10-29__training-ablation-v1
D=checkpoints/ablation/2026-10-30__training-ablation-v1
echo "===== far probe seeds 1-2 start $(date)"
for s in 1 2; do
  python -m evaluation.far_probe --workers 22 --out results/far_probe/v1-s$s --models \
    $L/R4-dart-gate__MLP01-256x3__s${s}__v1__2026-10-29/R4-dart-gate__MLP01-256x3__s${s}__v1__2026-10-29.npz \
    $D/R6-dagger-v1-r2__MLP01-256x3__s${s}__v1__2026-10-30/R6-dagger-v1-r2__MLP01-256x3__s${s}__v1__2026-10-30.npz \
    2>&1 | grep -v -E "No feet|file path|init_motor" | tail -8
done
echo "===== far probe other DAgger rows start $(date)"
python -m evaluation.far_probe --workers 22 --out results/far_probe/v1-rows --models \
  $D/R1D-dagger-R1v1-r3__MLP01-256x3__s0__v1__2026-10-30/R1D-dagger-R1v1-r3__MLP01-256x3__s0__v1__2026-10-30.npz \
  $D/R3D-dagger-R3v1-r3__MLP01-256x3__s0__v1__2026-10-30/R3D-dagger-R3v1-r3__MLP01-256x3__s0__v1__2026-10-30.npz \
  2>&1 | grep -v -E "No feet|file path|init_motor" | tail -8
~/sizes.sh
