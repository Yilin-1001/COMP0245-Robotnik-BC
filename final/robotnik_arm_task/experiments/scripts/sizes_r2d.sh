#!/bin/bash
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
RECIPE="--set identity.version=v1 --set identity.date=2026-10-31 --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5"
CK=checkpoints/ablation/2026-10-31__training-ablation-v1
VAR=R2D-dagger-R2v1-r3
mkdir -p results/sizes_r2d
echo "===== R2+DAgger network sizes start $(date)"
for A in "MLP02 [64, 64] 64x2" "MLP03 [128, 128] 128x2" "MLP04 [128, 128, 128] 128x3"; do
  set -- $A; NAME=$1; LABEL=${@: -1}; SHAPE=$(echo "$A" | sed -E 's/^MLP[0-9]+ (\[.*\]) .*/\1/')
  for s in 0 1 2; do
    RUN=${VAR}__${NAME}-${LABEL}__s${s}__v1__2026-10-31
    if [ ! -f $CK/$RUN/$RUN.npz ]; then
      rm -rf $CK/$RUN
      python -m training.datasets.train --variant $VAR --set identity.seed=$s \
        --set identity.architecture=$NAME --set "model.architectures={MLP01: [256, 256, 256], $NAME: $SHAPE}" \
        $RECIPE > results/sizes_r2d/train_${NAME}_s$s.log 2>&1 || { echo "TRAIN FAILED $RUN"; continue; }
    fi
    V=results/sizes_r2d/val_${NAME}_s$s.txt
    [ -s $V ] || python -m evaluation.evaluate --policy mlp --split validation --workers 22 --artifact $CK/$RUN/$RUN.npz \
      2>/dev/null | tail -16 > $V
    echo "$NAME $SHAPE seed $s: $(grep -E '"sustained_hold_success"|"latency_p50_us"' $V | tr -d '\n ')"
  done
done
echo "===== SIZES R2D DONE $(date)"
