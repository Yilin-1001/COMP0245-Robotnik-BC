#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
R4=checkpoints/ablation/2026-10-29__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-29/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-29.npz
RECIPE="--set identity.version=v1 --set identity.date=2026-10-30 --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5"
mkdir -p results/r5

echo "===== DAgger start $(date)"
python -m experiments.dagger_rounds --start $R4 --rounds 3 --tag v1 --date 2026-10-30 \
  --dev-scenes 64 --final-seeds 1 2 --workers 20

echo "===== R5 start $(date)"
for s in 0 1 2; do
  RUN=R5-dart-gate-split__MLP01-256x3__s${s}__v1__2026-10-30
  python -m training.datasets.train --variant R5-dart-gate-split --set identity.seed=$s $RECIPE > results/r5/train_s$s.log 2>&1
  python -m evaluation.evaluate --policy mlp --split validation --workers 20 \
    --artifact checkpoints/ablation/2026-10-30__training-ablation-v1/$RUN/$RUN.npz 2>/dev/null | tail -16 > results/r5/val_s$s.txt
  echo "R5 seed $s:"; grep -E '"sustained_hold_success"|"reach_success"' results/r5/val_s$s.txt
done
echo "===== ALL DONE $(date)"
