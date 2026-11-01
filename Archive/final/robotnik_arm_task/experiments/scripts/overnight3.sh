#!/bin/bash
# Overnight 3 (31 Oct): robustness probe, R6 rounds far probe, equal-data control, DART noise-strength ablation.
# Every step skips finished work, so the script can be stopped (pkill -f overnight3) and started again.
cd ~/COMP0245-Robotnik-BC/final/robotnik_arm_task
source ../../.venv/bin/activate
L=checkpoints/ablation/2026-10-29__training-ablation-v1
D=checkpoints/ablation/2026-10-30__training-ablation-v1
N=checkpoints/ablation/2026-10-31__training-ablation-v1
RECIPE="--set identity.version=v1 --set identity.date=2026-10-31 --set identity.objective=bcj --set training.lr=1.0e-3 --set training.updates=42000 --set loss.action=huber --set loss.arm_weight=0.5 --set loss.wheel_weight=0.5"
FILTER='No feet|file path|init_motor'
mkdir -p results/overnight3
m() { echo "$1/$2/$2.npz"; }

# ---------- 1. R4 + DAgger rounds 1 and 3 on far targets (why does round 2 fail far?) ----------
if [ ! -f results/far_probe/v1-R6-rounds/summary.json ]; then
  echo "===== 1. far probe R6 rounds 1 and 3 $(date)"
  python -m evaluation.far_probe --workers 22 --out results/far_probe/v1-R6-rounds --models \
    $(m $D R6-dagger-v1-r1__MLP01-256x3__s0__v1__2026-10-30) $(m $D R6-dagger-v1-r3__MLP01-256x3__s0__v1__2026-10-30) \
    2>&1 | grep -v -E "$FILTER" | tail -8
fi

# ---------- 2. robustness probe on the candidate final models ----------
if [ ! -f results/robust_probe/v1/summary.json ]; then
  echo "===== 2. robustness probe $(date)"
  python -m evaluation.robust_probe --workers 22 --n 12 --out results/robust_probe/v1 --models \
    $(m $L R4-dart-gate__MLP01-256x3__s0__v1__2026-10-29) \
    $(m $D R6-dagger-v1-r2__MLP01-256x3__s0__v1__2026-10-30) \
    $(m $D R2D-dagger-R2v1-r3__MLP01-256x3__s0__v1__2026-10-30) \
    $(m $D R1D-dagger-R1v1-r3__MLP01-256x3__s0__v1__2026-10-30) \
    2>&1 | grep -v -E "$FILTER" | tail -14
fi

# ---------- helper: one DART protocol -> collection -> gate dataset -> R4-style row, 3 seeds, validation ----------
dart_row() {  # $1 protocol name, $2 arm sigma, $3 wheel sigma, $4 draws, $5 dataset name, $6 variant name
  local P=$1 ARM=$2 WHEEL=$3 DRAWS=$4 DS=$5 VAR=$6
  local FREE=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
  if [ "$FREE" -lt 25 ]; then echo "SKIP $VAR: only ${FREE} GB free disk"; return; fi
  echo "===== $VAR ($P) $(date)"
  cat > experiments/configs/dart/$P.yaml <<YAML
# DART noise protocol $P (overnight3): d2-s075-x2 with arm sigma $ARM N*m, wheel sigma $WHEEL rad/s, $DRAWS draws.
protocol: $P
teacher: A
scenarios:
  source: data/Markovian/Clean_253/scenarios_train.csv
  split_manifest: experiments/splits/clean253.json
  ids: fitting_ids.clean253
draws_per_scenario: $DRAWS
seed: 20261028
noise:
  type: ou
  tau_s: 0.2
  alpha: [0.25, 1.0]
  arm_sigma_nm: $ARM
  wheel_sigma_rad_s: $WHEEL
  wheel_off: first_reach_active
  all_off_after_reach_s: 1.5
YAML
  cat > experiments/configs/datasets/variants/$VAR.yaml <<YAML
# $VAR (overnight3): Clean_253 fitting + DART $P, 22 inputs (features-v1 + reach_gate).
data:
  feature_definition: features-v1-gate
  datasets: [clean253__features-v1-gate, ${DS}__features-v1-gate]
  validation_dataset: clean253__features-v1-gate
YAML
  if [ ! -f data/datasets/A/${DS}__features-v1-gate/COMPLETE.json ]; then
    python -m data_collection.collect_dart --protocol experiments/configs/dart/$P.yaml --workers 22 2>&1 | grep -v -E "$FILTER" | tail -2
    local DIR=$(ls -d data/Markovian/DART/*__collection__$P-* | tail -1)
    python -m training.datasets.build --source $DIR --name $DS --features features-v1-gate --workers 22 | tail -3 \
      || { echo "BUILD FAILED $VAR"; return; }
  fi
  for s in 0 1 2; do
    local RUN=${VAR}__MLP01-256x3__s${s}__v1__2026-10-31
    if [ ! -f $N/$RUN/$RUN.npz ]; then
      rm -rf $N/$RUN
      python -m training.datasets.train --variant $VAR --set identity.seed=$s $RECIPE \
        > results/overnight3/train_${VAR}_s$s.log 2>&1 || { echo "TRAIN FAILED $RUN"; continue; }
    fi
    local V=results/overnight3/val_${VAR}_s$s.txt
    [ -s $V ] || python -m evaluation.evaluate --policy mlp --split validation --workers 22 --artifact $N/$RUN/$RUN.npz \
      2>/dev/null | tail -16 > $V
    echo "$VAR seed $s validation: $(grep -E '"sustained_hold_success"|"reach_success"' $V | tr -d '\n ')"
  done
}

# ---------- 3. equal-data control: R2 + ~880 extra DART episodes vs R2 + ~855 DAgger episodes (R2D) ----------
dart_row d2-s075-x4 0.75 0.15 4 dart-d2s075x4 R4X4-dart4-gate

# ---------- 4. DART noise strength: half and double sigma (2 draws, as R4) ----------
dart_row d2-s0375-x2 0.375 0.075 2 dart-d2s0375x2 R4H-dart-half-gate
dart_row d2-s150-x2 1.5 0.3 2 dart-d2s150x2 R4D-dart-double-gate

echo "===== OVERNIGHT3 DONE $(date)"
