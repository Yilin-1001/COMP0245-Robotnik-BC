#!/bin/bash
# Small-network DART + DAgger end to end, unattended:
#   1. R8 chains with 64x3 (MLP04), 32x3 (MLP02), 128x3 (MLP06) and the 256x3 reference (MLP01), 3 rounds, seeds 1-2
#      for the chosen round (experiments.dagger_arch_rounds; cost of every step in .../<arch>/cost.csv)
#   2. isolated latency benchmark of the chosen seed-0 models (run while nothing else runs)
#   3. 240-scene unified test of the chosen round, 3 seeds per network, plus teacher A
#   4. summary table: results/ablation/<date>__small-net-dagger-v2/summary.md
# Every step skips finished work. Stop:  pkill -f small_net_overnight; pkill -f experiments.dagger_arch_rounds
# Start again with the same command to continue.
#
# Usage (from final/robotnik_arm_task/):
#   nohup bash experiments/small_net_overnight.sh > small_net.log 2>&1 &
#   DATE=2026-11-04 WORKERS=20 ARCHS="MLP04 MLP02 MLP06 MLP01" bash experiments/small_net_overnight.sh
set -u -o pipefail
cd "$(dirname "$0")/.."
source ../../.venv/bin/activate
DATE=${DATE:-2026-11-04}
W=${WORKERS:-20}
ARCHS=${ARCHS:-"MLP04 MLP02 MLP06 MLP01"}
OUT=results/ablation/${DATE}__small-net-dagger-v2
TEST=results/test/${DATE}__small-net-dagger
FILTER='No feet|file path|init_motor'
mkdir -p "$OUT"

for A in $ARCHS; do
  echo "===== 1. $A chain $(date)"
  python -m experiments.dagger_arch_rounds --arch "$A" --date "$DATE" --rounds 3 --final-seeds 1 2 --workers "$W" \
    2>&1 | grep -v -E "$FILTER" || echo "!!!!! $A chain FAILED (see $OUT/$A/logs); continuing"
done

echo "===== 2. latency $(date)"
S0=$(python -m experiments.small_net_report --date "$DATE" --list-models --seeds 0)
if [ -n "$S0" ]; then
  python -m evaluation.latency_bench --models $S0 --out "$OUT/latency.csv" || echo "!!!!! latency FAILED"
fi

echo "===== 3. unified test $(date)"
MODELS=$(python -m experiments.small_net_report --date "$DATE" --list-models --seeds 0 1 2)
if [ -f "$TEST/run.json" ]; then
  if grep -q '"status": "complete"' "$TEST/run.json"; then echo "test already complete"
  else python -m evaluation.test_suite --resume "$TEST" --workers "$W" || echo "!!!!! test FAILED"; fi
elif [ -n "$MODELS" ]; then
  python -m evaluation.test_suite --models $MODELS --out "$TEST" --workers "$W" || echo "!!!!! test FAILED"
fi

echo "===== 4. summary $(date)"
if [ -f "$TEST/rollouts.csv" ]; then
  python -m experiments.small_net_report --date "$DATE" --test "$TEST"
else
  python -m experiments.small_net_report --date "$DATE"
fi
echo "===== SMALL NET DONE $(date)"
