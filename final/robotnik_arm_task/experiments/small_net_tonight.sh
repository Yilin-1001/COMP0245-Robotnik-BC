#!/bin/bash
# After experiments/small_net_overnight.sh (MLP02/04/06 chains done):
#   1. 240-scene unified test of the chosen rounds (local manifest: the 240 scenes are identical to
#      experiments/splits/unified-test-v1.json, only the machine-specific source audit differs)
#   2. 256x3 reference on this machine: latency of the old R4 + DAgger r2 model (benchmarked together with the small
#      models) and its recorded training time
#   3. training-budget scan for MLP04 round 3 (experiments.step_budget)
#   4. summary tables
# Resumable: stop with  pkill -f small_net_tonight; pkill -f evaluation.test_suite; pkill -f experiments.step_budget
# Usage (from final/robotnik_arm_task/):  nohup bash experiments/small_net_tonight.sh > tonight.log 2>&1 &
set -u -o pipefail
cd "$(dirname "$0")/.."
source ../../.venv/bin/activate
DATE=${DATE:-2026-11-04}
W=${WORKERS:-20}
OUT=results/ablation/${DATE}__small-net-dagger-v2
TEST=results/test/${DATE}__small-net-dagger
REF_DIR=checkpoints/ablation/2026-10-30__training-ablation-v1
REF=R6-dagger-v1-r2__MLP01-256x3__s0__v1__2026-10-30     # = R8 (R4 + DAgger r2), 256x3, seed 0

echo "===== 1. unified test $(date)"
if [ -f "$TEST/run.json" ]; then
  if grep -q '"status": "complete"' "$TEST/run.json"; then echo "test already complete"
  else python -m evaluation.test_suite --resume "$TEST" --workers "$W" || echo "!!!!! test FAILED"; fi
else
  rm -rf "$TEST"
  MODELS=$(python -m experiments.small_net_report --date "$DATE" --list-models --seeds 0 1 2)
  python -m evaluation.test_suite --models $MODELS --manifest experiments/splits/unified-test-v1-local.json \
    --out "$TEST" --workers "$W" || echo "!!!!! test FAILED"
fi

echo "===== 2. 256x3 reference $(date)"
S0=$(python -m experiments.small_net_report --date "$DATE" --list-models --seeds 0)
python -m evaluation.latency_bench --models $S0 "$REF_DIR/$REF/step_42000.npz" --out "$OUT/latency_with_256.csv" \
  && cat "$OUT/latency_with_256.csv" || echo "!!!!! latency with 256x3 FAILED (small-network latency.csv is still valid)"
for RUN in "$REF" R6-dagger-v1-r1__MLP01-256x3__s0__v1__2026-10-30 R6-dagger-v1-r3__MLP01-256x3__s0__v1__2026-10-30; do
  M="$REF_DIR/$RUN/metrics.csv"
  [ -f "$M" ] && echo "256x3 training time $RUN: $(tail -1 "$M" | cut -d, -f2) s"
done
R4M=checkpoints/ablation/2026-10-29__training-ablation-v1/R4-dart-gate__MLP01-256x3__s0__v1__2026-10-29/metrics.csv
[ -f "$R4M" ] && echo "256x3 training time R4 (round 0): $(tail -1 "$R4M" | cut -d, -f2) s"

echo "===== 3. training budget, MLP04 round 3 $(date)"
python -m experiments.step_budget --arch MLP04 --round 3 --date "$DATE" --budgets 21000 10500 5000 \
  --confirm-seeds 1 2 --workers "$W" 2>&1 | grep -v -E 'No feet|file path|init_motor' || echo "!!!!! step budget FAILED"

echo "===== 4. summary $(date)"
if [ -f "$TEST/rollouts.csv" ]; then
  python -m experiments.small_net_report --date "$DATE" --test "$TEST" --latency "$OUT/latency_with_256.csv" 2>/dev/null \
    || python -m experiments.small_net_report --date "$DATE" --test "$TEST"
else
  python -m experiments.small_net_report --date "$DATE"
fi
echo "===== TONIGHT DONE $(date)"
