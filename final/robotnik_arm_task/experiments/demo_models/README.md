# Demo model recommendations — 2026-10-27

`2026-10-27__selection__best-validated.yaml` records recommendations from the 29 existing full-validation
evaluations (32 scenarios each). It is a demo manifest accepted by `evaluation.live_compare --protocol`;
it is **not** a frozen final-test protocol. Original protocols, checkpoints and evaluation rows are unchanged.

The original architecture selector compared three-seed mean sustained-hold success among two finalists
per teacher, then preferred the smaller network when tied within 1/32. A's M3/M6 and B's M5/M1 all held
0/32 for every seed. This selected A M6 and B M1 by size, without using reach or final error to break that tie.

This manifest ranks individual validated artifacts by sustained hold, reach success, then median final error
(using the existing `rank_rollout` rule). Test outcomes do not enter this recommendation.

| Manifest entry | Architecture | Data | Seed | Step | Validation hold | Reach | Median final error |
|---|---|---|---:|---:|---:|---:|---:|
| A-best-validation | M6: 256/128/64, BC+J | Clean_253 | 0 | 30,000 | 0/32 | 4/32 | 768 mm |
| B-best-validation | M1: 64/64, BC | Clean_223 | 0 | 2,000 | 0/32 | 1/32 | 1,699 mm |
| B-lowest-final-error | M5: 256/256/256, BC | Clean_253 | 1 | 16,000 | 0/32 | 0/32 | 813 mm |

B's one reach is transient: that scenario ends 861 mm from the target. It does not establish a robust advantage
over M5. For the requested physical demo, **use A-best-validation and B-lowest-final-error**. B M5's median
final errors across seeds 0/1/2 are 1,080/813/993 mm, versus 1,825/5,957/3,940 mm for B M1 on Clean_253;
this favors M5 for observing smaller drift despite neither architecture achieving sustained hold.
The task-ranked B Clean_223 M1 remains available as an alternate. Clean_223 also
omits the lowest training-target interval; validation still uses the same full-range 32 scenarios.

All candidates fail sustained hold. These are the best **among existing full-validation results under the stated
ranking**, not successful controllers or the proven best of every saved checkpoint. Most checkpoint files have
only screening/offline evidence. No retraining or additional model selection on the test set was performed.

Run from `final/robotnik_arm_task/` with the project's Python environment:

```bash
# A: best task-ranked validation result
python -m evaluation.live_compare --protocol experiments/demo_models/2026-10-27__selection__best-validated.yaml --teacher A --objective bcj --seed 0

# B: recommended physical demo, lowest median final error, M5 seed 1
python -m evaluation.live_compare --protocol experiments/demo_models/2026-10-27__selection__best-validated.yaml --teacher B --objective bc --seed 1

# B alternate: task-ranked validation result (one transient reach), Clean_223 M1 seed 0
python -m evaluation.live_compare --protocol experiments/demo_models/2026-10-27__selection__best-validated.yaml --teacher B --objective bc --seed 0
```

Append `--target X Y Z` for a custom target. These commands load the exact `step_<k>.npz` export used in the
recorded full-validation evaluation, whose hash is included in the manifest. Without `--protocol`, the demo's
original default remains the frozen final protocol and its BC seed-0 models.
