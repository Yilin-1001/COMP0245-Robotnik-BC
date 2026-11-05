#!/bin/bash
# Small-network DART + DAgger WITHOUT the 64 extra development scenes per round: every DAgger round collects on the
# 221 fitting scenes only, so the learner sees exactly the scenes of the clean + DART data (221 + 441 episodes).
# Same three widths as small_net_overnight.sh; round 0 (R4 data, same model) is reused from the 2026-11-04 run.
#   1. chains MLP04, MLP02, MLP06 (--dev-scenes 0, tag nodev)          ~2 h
#   2. 240-scene unified test of the chosen rounds, 3 seeds (TEST=0 skips it)   ~2.5-3 h
#   3. summary: results/ablation/<date>__small-net-dagger-nodev-v2/summary.md
# Resumable: stop with  pkill -f small_net_nodev; pkill -f experiments.dagger_arch_rounds; pkill -f evaluation.test_suite
# Usage (from final/robotnik_arm_task/):  nohup bash experiments/small_net_nodev.sh > nodev.log 2>&1 &
set -u -o pipefail
cd "$(dirname "$0")/.."
source ../../.venv/bin/activate
DATE=${DATE:-2026-11-05}
R0=${R0_DATE:-2026-11-04}
W=${WORKERS:-20}
ARCHS=${ARCHS:-"MLP04 MLP02 MLP06"}
TEST_RUN=${TEST:-1}
OUT=results/ablation/${DATE}__small-net-dagger-nodev-v2
TDIR=results/test/${DATE}__small-net-dagger-nodev
FILTER='No feet|file path|init_motor'
mkdir -p "$OUT"

for A in $ARCHS; do
  echo "===== 1. $A chain, no development scenes $(date)"
  python -m experiments.dagger_arch_rounds --arch "$A" --date "$DATE" --tag nodev --round0-date "$R0" \
    --dev-scenes 0 --rounds 3 --final-seeds 1 2 --workers "$W" 2>&1 | grep -v -E "$FILTER" \
    || echo "!!!!! $A chain FAILED (see $OUT/$A/logs); continuing"
done

if [ "$TEST_RUN" = "1" ]; then
  echo "===== 2. unified test $(date)"
  if [ -f "$TDIR/run.json" ]; then
    if grep -q '"status": "complete"' "$TDIR/run.json"; then echo "test already complete"
    else python -m evaluation.test_suite --resume "$TDIR" --workers "$W" || echo "!!!!! test FAILED"; fi
  else
    rm -rf "$TDIR"
    MODELS=$(python -m experiments.small_net_report --date "$DATE" --tag nodev --list-models --seeds 0 1 2)
    # a fresh local manifest: the 240 scenes are fixed; only the machine's dataset inventory (source audit) changed
    python -m evaluation.test_suite --models $MODELS --manifest experiments/splits/unified-test-v1-local-nodev.json \
      --out "$TDIR" --workers "$W" || echo "!!!!! test FAILED"
  fi
fi

echo "===== 3. summary $(date)"
if [ -f "$TDIR/rollouts.csv" ]; then
  python -m experiments.small_net_report --date "$DATE" --tag nodev --test "$TDIR" \
    --latency results/ablation/${R0}__small-net-dagger-v2/latency.csv
else
  python -m experiments.small_net_report --date "$DATE" --tag nodev --latency results/ablation/${R0}__small-net-dagger-v2/latency.csv
fi
echo "===== NODEV DONE $(date)"
