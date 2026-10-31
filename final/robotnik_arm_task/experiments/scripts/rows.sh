#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
for ROW in R3-dart R2-gate R1-baseline; do
  TAG=$(echo $ROW | cut -d- -f1)v1
  echo "===== $ROW start $(date)"
  python -m experiments.dagger_rounds --row $ROW --rounds 3 --tag $TAG --date 2026-10-30 \
    --baseline-seeds 1 2 --final-seeds 1 2 --dev-scenes 64 --workers 20
done
echo "===== ALL ROWS DONE $(date)"
