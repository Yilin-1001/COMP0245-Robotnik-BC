#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
RECIPE="--set identity.version=v1 --set identity.date=2026-10-30 --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5"
CK=checkpoints/ablation/2026-10-30__training-ablation-v1
mkdir -p results/sizes

echo "===== network sizes start $(date)"
for A in "MLP02 [64, 64] 64x2" "MLP03 [128, 128] 128x2" "MLP04 [128, 128, 128] 128x3"; do
  set -- $A; NAME=$1; LABEL=${@: -1}; SHAPE=$(echo "$A" | sed -E 's/^MLP[0-9]+ (\[.*\]) .*/\1/')
  for s in 0 1 2; do
    RUN=R6-dagger-v1-r2__${NAME}-${LABEL}__s${s}__v1__2026-10-30
    if [ ! -f $CK/$RUN/$RUN.npz ]; then
      python -m training.datasets.train --variant R6-dagger-v1-r2 --set identity.seed=$s \
        --set identity.architecture=$NAME --set "model.architectures={MLP01: [256, 256, 256], $NAME: $SHAPE}" \
        $RECIPE > results/sizes/train_${NAME}_s$s.log 2>&1 || { echo "TRAIN FAILED $RUN"; continue; }
    fi
    python -m evaluation.evaluate --policy mlp --split validation --workers 20 --artifact $CK/$RUN/$RUN.npz \
      2>/dev/null | tail -16 > results/sizes/val_${NAME}_s$s.txt
    echo "$NAME $SHAPE seed $s: $(grep -E '"sustained_hold_success"|"latency_p50_us"' results/sizes/val_${NAME}_s$s.txt | tr -d '\n ')"
  done
done

echo "===== ablation test sets start $(date)"
for R in R1D-dagger-R1v1-r3 R2D-dagger-R2v1-r3 R3D-dagger-R3v1-r3; do
  for s in 0 1 2; do
    RUN=${R}__MLP01-256x3__s${s}__v1__2026-10-30
    [ -f results/test/${R}-s$s/rollouts.csv ] && continue
    python -m evaluation.test_diagnosis --modes student --workers 20 --models $CK/$RUN/$RUN.npz \
      --out results/test/${R}-s$s > /dev/null 2>&1 || echo "TEST FAILED $RUN"
    echo "$R seed $s test: $(python -c "import csv;r=[x for x in csv.DictReader(open('results/test/${R}-s$s/rollouts.csv')) if x['mode']=='student'];print(sum(x['sustained_hold_pass']=='True' for x in r),'/',len(r))")"
  done
done
echo "===== ALL DONE $(date)"
