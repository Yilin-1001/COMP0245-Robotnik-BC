# 06. Ablation Experiment

This folder holds the evidence for the report's ablation section: the eight training-method rows R1–R8, their validation results, the unified test, the per-round DAgger results, and the cost and efficiency of each row. It contains the figures, the tables, the code that produced them and the data behind every number.

**Question.** What does each part of the training pipeline contribute: OU-noise data, DAgger, their combination (f-DAgger), and the reach-gate input?

## Naming

The row numbers and run ids are unchanged. Only the names used in the report are new: each of four data-collection methods is trained without and with the Gate. In the run ids, "dart" means the phase-gated OU-noise data, which this report calls **OU**.

| Row | Report name | Run id | Gate | OU noise | DAgger | Inputs | Training data |
|---|---|---|:---:|:---:|:---:|---|---|
| R1 | **Baseline** | `R1-baseline` | | | | 21 | 221 clean expert episodes |
| R2 | **Baseline + Gate** | `R2-gate` | ✓ | | | 22 | 221 clean expert episodes |
| R3 | **OU** | `R3-dart` | | ✓ | | 21 | 221 clean + 441 OU-noise episodes |
| R4 | **OU + Gate** | `R4-dart-gate` | ✓ | ✓ | | 22 | 221 clean + 441 OU-noise episodes |
| R5 | **DAgger** | `R5-baseline-dagger` | | | ✓ | 21 | R1 data + DAgger rounds 1–3 |
| R6 | **DAgger + Gate** | `R6-gate-dagger` | ✓ | | ✓ | 22 | R2 data + DAgger rounds 1–3 |
| R7 | **f-DAgger** | `R7-dart-dagger` | | ✓ | ✓ | 21 | R3 data + DAgger rounds 1–3 |
| R8 | **f-DAgger + Gate** | `R8-dart-gate-dagger` | ✓ | ✓ | ✓ | 22 | R4 data + DAgger rounds 1–3 |

`tables/naming.csv` has this mapping in machine-readable form. Every table and figure here carries both the row number and the report name.

- **Gate.** A 22nd input, `reach_gate`, that is 1 when the expert's own rule says the arm should reach: the base is within 0.12 m of the standoff distance and within 0.075 rad of the target heading. It is computed exactly from the other 21 inputs, so it adds no information. What it adds is the expert's switch from "hold the home pose" to "reach" as a single feature the network can use directly.
- **OU.** The 221 fitting scenarios are collected twice more, with Ornstein–Uhlenbeck noise (τ = 0.2 s; arm σ 0.75 N·m, wheels σ 0.15 rad/s, × α ~ U[0.25, 1] per episode) added to the *executed* command. The label is always the clean expert action at the visited state. 441 of the 442 episodes reached a sustained hold and are used. Design and parameter evidence: [`../04_f_dagger/`](../04_f_dagger/README.md).
- **DAgger.** The student drives alone (β = 0) on the 221 fitting scenarios plus 64 fresh development scenes per round, biased towards low and near targets. Expert A labels every visited state. The next student is trained from scratch on all data so far. Three rounds; round 3 had the best seed-0 validation result in every row, so it is the model tested.
- **f-DAgger** = OU noise, then DAgger: the DAgger rounds start from the OU row's data and model (R7 from R3, R8 from R4).
- **Shared by all eight rows.** Expert A labels; the MLP01 network (256×3, LayerNorm, SiLU); BC + damping-Jacobian loss, Huber, AdamW, lr 1e-3, 42k updates; the terminal checkpoint; seeds 0, 1, 2. Details: [`../05_training_details/`](../05_training_details/README.md).

## How uncertainty is reported

Every number in this folder is one of the four kinds below. Each table and figure says which one it uses.

| Quantity | What is reported | Method | Unit of resampling | Where |
|---|---|---|---|---|
| Success rate of one row | rate, k/n, **95% CI** | Percentile bootstrap over **scenes**, 1,000 resamples, seed 1101031 (the test suite's own method). The 3 seed rollouts of a scene are resampled together, so they are not treated as independent. | scene (32 validation, 240 test) | `validation.csv` (`scene_boot95_*`), `test_by_band.csv` / `test_by_suite.csv` (`scene_boot95_*`), all error bars |
| Validation, reference only | Wilson 95% score interval | Binomial on 96 trials pooled over seeds. It assumes 96 independent trials and ignores that the same 32 scenes are run 3 times, so it is slightly narrower. Kept for comparison with older documents. | rollout | `validation.csv` (`wilson95_*_iid`) |
| Difference between two rows | difference in pp, **95% paired CI**, **p value**, **Holm-adjusted p** | Both rows run on the same scenes, so the comparison is paired. The per-scene difference in success (mean over 3 seeds) is bootstrapped over scenes (10,000 resamples). p: two-sided sign-flip permutation test of the mean per-scene difference (20,000 draws, p = (1 + #extreme) / (1 + 20,000)). Holm's step-down correction over all 112 differences in `comparisons.csv`. | scene | `comparisons.csv` (all 12 one-factor contrasts of the 2×2×2 design plus 2 cross-method contrasts, on validation and each test suite), `gate_effect.csv`, `figures/gate_effect.*` |
| Seed-to-seed spread | per-seed counts; sign agreement over the 9 seed pairings | Descriptive only. With 3 seeds per row, no interval over training runs is attempted. "9/9" means that every pairing of a seed of one row with a seed of the other gives a difference of the same sign as the pooled one. | training seed | `hold_per_seed`, `per_seed`, `seed_pairings_same_sign` |

What the intervals do and do not cover:

- They cover **scene sampling**: how much the rate would change on another draw of scenes from the same distributions. They are **conditional on the 3 trained models** of each row and do not cover the variation from training more seeds.
- **Significance claims use the Holm-adjusted p < 0.05**, over every comparison in `comparisons.csv`. A raw p or a CI that excludes 0 without Holm significance is reported as "indicative", not as an effect. With 20,000 permutations the smallest possible raw p is 5×10⁻⁵ and the smallest Holm p 0.0056, so these values mean "at the resolution limit".
- Validation chose the DAgger round count, so validation differences between R5–R8 are slightly optimistic. The test intervals are the unbiased ones.
- The percentile bootstrap can undercover when there are few scenes: T2, T4 and T5 have 24 scenes each and T3 has 32. Treat their intervals as approximate.
- DAgger-round counts (`validation_by_round.csv`), the hold test, terminal grades, failure modes and cost are **descriptive counts with no interval**.

## Main result

![Ablation overview](figures/ablation_overview.png)

`figures/ablation_overview.pdf` is the IEEE double-column version. Hatched orange bars are without the Gate, solid blue bars with it. **Error bars: 95% percentile bootstrap CI over scenes** (validation 32 scenes, T0 64, all 240), with each scene's 3 seed rollouts resampled together.

**Validation** (`tables/validation.csv`; 32 validation scenes × 3 seeds; sustained hold = continuous < 10 mm until the end):

| Row | Report name | Hold per seed | Pooled | 95% CI (scene bootstrap) | Wilson 95% (iid, reference) | Median final error |
|---|---|---|---|---|---|---|
| R1 | Baseline | 0 / 1 / 0 | 1/96 | 0.00–0.03 | 0.00–0.06 | 930 mm |
| R2 | Baseline + Gate | 2 / 3 / 5 | 10/96 | 0.03–0.18 | 0.06–0.18 | 1051 mm |
| R3 | OU | 0 / 1 / 0 | 1/96 | 0.00–0.03 | 0.00–0.06 | 293 mm |
| R4 | OU + Gate | 29 / 27 / 27 | 83/96 | 0.77–0.95 | 0.78–0.92 | 1.3 mm |
| R5 | DAgger | 28 / 29 / 29 | 86/96 | 0.80–0.97 | 0.82–0.94 | 2.7 mm |
| R6 | DAgger + Gate | 31 / 29 / 30 | 90/96 | 0.87–0.99 | 0.87–0.97 | 2.2 mm |
| R7 | f-DAgger | 30 / 31 / 30 | 91/96 | 0.89–1.00 | 0.88–0.98 | 1.7 mm |
| R8 | **f-DAgger + Gate** | 31 / 31 / 31 | **93/96** | 0.91–1.00 | 0.91–0.99 | **1.2 mm** |

The rows R5–R8 overlap in their intervals; the paired differences between them are in the findings below.

**Unified test** (`tables/test_by_suite.csv`; 240 scenes × 3 seeds; success = continuous < 10 mm for 3 s, or 20 s on T5; the last column is the 95% scene-bootstrap CI; per-suite CIs are in `tables/test_by_band.csv`):

| Row | Report name | T0 in range | T1 far 3–15 m | T2 near | T3 height edges | T4 shifted start | T5 long hold | All 240 [95% CI, scene bootstrap] |
|---|---|---|---|---|---|---|---|---|
| – | Expert A (teacher) | 100 | 100 | 100 | 81.2 | 100 | 100 | 97.5 [95.4, 99.2] |
| R1 | Baseline | 2.1 | 0.0 | 2.8 | 0.0 | 2.8 | 0.0 | 1.1 [0.4, 1.9] |
| R2 | Baseline + Gate | 20.8 | 0.0 | 1.4 | 3.1 | 22.2 | 0.0 | 8.3 [5.7, 11.3] |
| R3 | OU | 1.6 | 3.7 | 6.9 | 14.6 | 4.2 | 0.0 | 4.6 [3.1, 6.4] |
| R4 | OU + Gate | 96.9 | 90.7 | 47.2 | 54.2 | 94.4 | 68.1 | **81.2** [76.8, 85.6] |
| R5 | DAgger | 96.4 | 97.7 | 37.5 | 12.5 | 94.4 | 51.4 | 75.0 [70.6, 79.3] |
| R6 | DAgger + Gate | 97.9 | **100** | 50.0 | 21.9 | 95.8 | 72.2 | 80.8 [76.5, 85.1] |
| R7 | f-DAgger | 97.4 | 88.9 | 50.0 | 25.0 | 93.1 | 54.2 | 75.7 [71.2, 80.0] |
| R8 | **f-DAgger + Gate** | **99.0** | 44.4 | **81.9** | **66.7** | **97.2** | **87.5** | 75.3 [70.4, 79.7] |

Bold marks the highest point estimate in each column; it does not mean the row is significantly better than the others (see below).

![Test success per suite](figures/test_suites.png)

The heatmap shows point estimates only. Per band (T1 3–6 / 6–10 / 10–15 m, T3 low / high), with the 95% scene-bootstrap CI, per-seed counts, median completion time and median terminal error: `tables/test_by_band.csv`.

## What the ablation shows

Each claim gives the paired difference Δ (with minus without, percentage points) with its **95% paired scene-bootstrap CI** and the **Holm-adjusted p** (`tables/comparisons.csv`).

1. **Without DAgger, OU and the Gate work only together.**
   - OU alone vs Baseline (R3 − R1), validation: Δ = 0.0 [−3.1, 3.1], p_Holm = 1.
   - Gate alone (R2 − R1), validation: Δ = +9.4 [+2.1, +17.7], raw p = 0.046, p_Holm = 1 (indicative only).
   - Adding the Gate to OU (R4 − R3), validation: Δ = **+85.4** [+75.0, +93.8]. Adding OU to the Gate (R4 − R2): Δ = **+76.0** [+65.6, +86.5]. Both have p_Holm = 0.0056 and 9/9 seed pairings agree.
   - The two components fix different failures (`tables/hold_test.csv`, below): OU fixes the hold, the Gate fixes the start of the reach.
2. **With the Gate, OU data and three DAgger rounds are statistically indistinguishable in range.** DAgger + Gate − OU + Gate (R6 − R4): validation Δ = +7.3 [−2.1, +17.7]; T0 Δ = +1.0 [−1.6, +3.6]; all 240 Δ = −0.4 [−5.1, +4.3]; p_Holm = 1 for all three. The OU row needs no train→collect cycle (3.3 vs 28.4 min added; `tables/cost.csv`). The suites differ in opposite directions: DAgger + Gate is better on T1 (Δ = +9.3 [+5.6, +13.9], p_Holm = 0.0056), and OU + Gate is better on T3 (Δ = −32.3 [−52.1, −12.5], raw p = 0.0068, p_Holm = 0.33, indicative only).
3. **Starting DAgger from OU data (f-DAgger) helps at the range edges, not in range.**
   - In range there is no detectable difference. f-DAgger + Gate − DAgger + Gate (R8 − R6): validation Δ = +3.1 [−1.0, +8.3]; T0 Δ = +1.0 [−1.0, +3.6]; T4 Δ = +1.4 [−4.2, +8.3]; p_Holm = 1. Without the Gate (R7 − R5), validation Δ = +5.2 [−2.1, +12.5], p_Holm = 1. So the higher validation counts (93 vs 90, 91 vs 86 of 96) are **not** evidence of a difference.
   - At the edges the difference is clear: T2 near Δ = **+31.9** [+18.1, +45.8], p_Holm = 0.026; T3 height edges Δ = **+44.8** [+32.3, +57.3], p_Holm = 0.0056. All 9/9 seed pairings agree.
4. **Once DAgger is used, the Gate does not change in-range success but changes the edges** (`tables/gate_effect.csv`, `figures/gate_effect.*`).
   - In range: with DAgger, validation Δ = +4.2 [−3.1, +11.5] and T0 Δ = +1.6 [−0.5, +4.2]; with f-DAgger, validation Δ = +2.1 [−2.1, +7.3] and T0 Δ = +1.6 [−1.0, +5.2]; p_Holm = 1 for all four.
   - With f-DAgger (R8 − R7): T3 Δ = **+41.7** [+31.2, +52.1] (p_Holm = 0.0056); T5 Δ = +33.3 [+19.4, +47.2] (p_Holm = 0.047); T2 Δ = +31.9 [+16.7, +47.2] (raw p = 0.001, p_Holm = 0.057, indicative).
   - With DAgger (R6 − R5), no single suite reaches Holm significance. Pooled over all 240 scenes, Δ = +5.8 [+2.9, +8.8], p_Holm = 0.0091.
5. **Far targets (T1) reverse the order over all 240 scenes.** f-DAgger + Gate holds 44.4% on T1 (97.2% at 3–6 m, 36.1% at 6–10 m, 0% at 10–15 m).
   - It is significantly worse on T1 than OU + Gate (R8 − R4: Δ = −46.3 [−55.1, −37.5]), DAgger + Gate (R8 − R6: Δ = −55.6 [−65.3, −45.8]) and f-DAgger (R8 − R7: Δ = −44.4 [−52.8, −36.1]); p_Holm = 0.0056 for all three.
   - All 120 of its T1 failures never came within 10 mm (`tables/failure_modes.csv`); these are navigation failures.
   - Over all 240 scenes, f-DAgger + Gate − OU + Gate is Δ = −6.0 [−11.4, −0.7] (raw p = 0.033, p_Holm = 1). f-DAgger + Gate − DAgger + Gate is Δ = −5.6 [−11.7, +0.6] (p_Holm = 1). So OU + Gate, DAgger + Gate and f-DAgger + Gate are **not separable over all 240 scenes**; the differences between them are in which suites they win.
   - The T1 comparison is confounded by data coverage; see the caveats.
6. **Where the remaining failures are** (`tables/failure_modes.csv`; counts, no interval). Outside T1, the DAgger rows fail mainly at T2 and T3, as near misses: the median closest approach of the failures is 6–20 mm. On T5 (20 s hold) almost every failure first reached the target and then lost the hold. f-DAgger + Gate has the fewest T5 failures (9/72, against 20–35 for the other DAgger rows).

![Gate effect per method](figures/gate_effect.png)

`figures/gate_effect.pdf`: the Gate effect (with − without) per method and metric. Lines are 95% paired scene-bootstrap CIs; filled dots are Holm-significant (p < 0.05 over all 112 comparisons), open dots are not.

### Why the two components of OU + Gate are both needed

`tables/hold_test.csv` (R1–R4 only; 8 scenes × 3 seeds; descriptive counts and medians, no interval). The student is started at the expert's own hold state and must stay within 10 mm for 10 s. The stiffness is the largest eigenvalue of the linearised position feedback.

| Row | Report name | Arm stays on target | Max stiffness eigenvalue (median) | Max real eigenvalue of the closed loop (median) |
|---|---|---|---|---|
| R1 | Baseline | 8/24 | 143 N·m/rad | 0.73 |
| R2 | Baseline + Gate | 3/24 | 113 N·m/rad | 0.92 |
| R3 | OU | **24/24** | **1.4 N·m/rad** | 0.02 |
| R4 | OU + Gate | 23/24 | 1.7 N·m/rad | 0.03 |

The expert's closed-loop maximum real eigenvalue is 0.004 on the same states. OU data make the hold stable, with or without the Gate. Without the Gate, the OU student holds the home pose so firmly that it never starts the reach (R3 validation 1/96, median final error 293 mm). The Gate gives the network the expert's switch, and the reach starts.

### DAgger rounds

![Validation per DAgger round](figures/dagger_rounds.png)

`tables/validation_by_round.csv` (counts, no interval). Round 0 is the starting model (R1–R4, 3 seeds). For rounds 1–2, all 3 seeds exist only for the Gate rows (extra seeds of 2026-11-03); DAgger and f-DAgger without the Gate have seed 0 only there (open markers).

| Row | Report name | Round 0 | Round 1 | Round 2 | Round 3 |
|---|---|---|---|---|---|
| R5 | DAgger | 1/96 | 15/32 | 21/32 | 86/96 |
| R6 | DAgger + Gate | 10/96 | 71/96 | 85/96 | 90/96 |
| R7 | f-DAgger | 1/96 | 24/32 | 28/32 | 91/96 |
| R8 | f-DAgger + Gate | 83/96 | 87/96 | 85/96 | 93/96 |

Only f-DAgger + Gate starts high: its first round already records small on-policy deviations around a working controller. The others spend their first round recording an untrained student failing ([`../04_f_dagger/tables/dagger_collections.csv`](../04_f_dagger/README.md)).

### Terminal precision

![Terminal error on T0](figures/terminal_error_ecdf.png)

The final-second maximum error on the 64 T0 scenes, per method (empirical distribution of all 192 rollouts per row; no interval). `tables/terminal_grades.csv` has grades over all 240 scenes: A < 5, B 5–10, C 10–20, D 20–50, E ≥ 50 mm. The Gate shifts the f-DAgger curve furthest to the left: 76% of f-DAgger + Gate T0 rollouts end below 10 mm, against 40% for f-DAgger. With DAgger alone the two curves almost coincide.

## Cost and efficiency

`tables/cost.csv`. Added wall time and expert labels are counted beyond the shared clean collection and round-0 training. DAgger times are the seed-0 chain from the 2026-11-01 run log (collect + build + train + validate per round).

| Row | Report name | Training episodes | Training rows | Added expert labels | Added wall time | Sequential train→collect cycles |
|---|---|---|---|---|---|---|
| R1 / R2 | Baseline (+ Gate) | 221 | 2.66 M | 0 | 0 | 0 |
| R3 / R4 | OU (+ Gate) | 662 | 7.95 M | 5.29 M | 3.3 min | 0 |
| R5 | DAgger | 1,076 | 23.98 M | 21.32 M | 30.3 min | 3 |
| R6 | DAgger + Gate | 1,076 | 21.28 M | 18.62 M | 28.4 min | 3 |
| R7 | f-DAgger | 1,517 | 27.24 M | 24.58 M | 33.0 min | 3 |
| R8 | f-DAgger + Gate | 1,517 | 19.18 M | 16.53 M | 26.0 min | 3 |

f-DAgger + Gate has the *fewest* DAgger rows of the four DAgger rows. A DAgger episode ends 3 s after the first task reach or at the 40 s cap, and its collector students already reach from round 1 on (median round-1 episode 11,000 rows, against 40,000 for the other three rows). The OU collection needs no trained student, so it can run alongside the clean collection.

`tables/efficiency.csv`. All rows use the same network: 141,065 parameters with 21 inputs, 141,321 with 22 (551–552 KiB as float32). The Gate costs 256 parameters. In the test rollouts the median per-step inference latency (p50) was 168–229 µs, well inside the 1 ms control period. These rollouts ran 20 in parallel; on one idle CPU thread the measured cost is 83 µs per step (features + FK + MLP; [`../05_training_details/`](../05_training_details/README.md)).

## Layout

```
06_ablation/
  README.md
  figures/   ablation_overview.{pdf,png}     validation, test T0, test all 240; 4 methods x without/with Gate (report figure)
             test_suites.{pdf,png}           success per suite T0-T5 and all 240, R1-R8 and expert A
             dagger_rounds.{pdf,png}         validation per DAgger round, R5-R8
             terminal_error_ecdf.{pdf,png}   T0 final-second max error, one panel per method
             gate_effect.{pdf,png}           Gate effect per method and metric, 95% paired CIs, Holm significance
  tables/    naming.csv, validation.csv, validation_by_round.csv, test_by_suite.csv, test_by_band.csv,
             comparisons.csv (112 paired differences: CI, permutation p, Holm p, seed agreement), gate_effect.csv,
             failure_modes.csv, terminal_grades.csv, hold_test.csv, cost.csv, efficiency.csv
             (all from code/make_tables.py)
  data/      validation/   v1 R1-R4 summary, runs and manifest (2026-10-28); v2 R5-R8 per-round summaries
                           (2026-11-01) and extra seeds for R6/R8 rounds 1-2 (2026-11-03);
                           per_scene/<run>.csv: per-scene validation records of all 40 models (used for the
                           scene-level intervals and the paired tests)
             hold_test/    the 12 R1-R4 hold-test records (8 scenes each)
             test/         unified test of 2026-11-01: rollouts.csv (6,000 rollouts), summary.{csv,md}, scene_manifest.json,
                           run.json, report.json
             datasets/     episodes.csv of every training dataset (clean, OU, 12 DAgger rounds) and the clean253 split
             collections/  OU collection manifest and the 12 DAgger collection manifests
             logs/         v1 training log (simulator noise removed), v2 DAgger run log, extra-seed run log,
                           OU collection write log (attempts.jsonl)
  code/      make_tables.py    data/ -> tables/
             plot_ablation.py  tables/ + data/ -> figures/
```

## Reproduce

All commands run from `final/robotnik_arm_task/`. Every number in this folder is already in `data/` and `tables/`.

```bash
python report/06_ablation/code/make_tables.py      # tables/ from data/ (no simulation)
python report/06_ablation/code/plot_ablation.py    # figures/ from tables/ and data/ (needs matplotlib)
```

`make_tables.py` checks its per-suite counts against the test suite's own `summary.csv`, its per-scene validation counts against the run summaries and its hold-test counts against the v1 summary, and stops on any mismatch. The bootstrap and permutation draws use fixed seeds, so the intervals and p values are reproduced exactly.

The pipeline (simulation and training; outputs are git-ignored; full instructions in the project README):

```bash
# R1-R4: clean and OU data, 3 seeds each, validation and hold test
python -m experiments.ablation.run --manifest experiments/ablation/2026-10-28__training-ablation-v1.yaml
# R5-R8: 3 DAgger rounds from the seed-0 R1-R4 model, then seeds 1-2 of the chosen round
python -m experiments.dagger_rounds --row R5-baseline-dagger --rounds 3 --date 2026-11-01 --final-seeds 1 2 --workers 20
#   (the same for R6-gate-dagger, R7-dart-dagger, R8-dart-gate-dagger)
python -m experiments.dagger_extra_seeds --date 2026-11-03 --rows R6-gate-dagger R8-dart-gate-dagger --rounds 1 2 --seeds 1 2 --workers 20
# Unified test of the 24 tested models plus expert A
python -m evaluation.test_suite --models <step_42000.npz files> --out results/test/<run> --workers 20
```

Source runs: `results/ablation/2026-10-28__training-ablation-v1/` (R1–R4), `results/ablation/2026-11-01__training-ablation-v2/` (R5–R8), `results/ablation/2026-11-03__training-ablation-v2/` (extra seeds), `results/test/2026-11-01_215340__R1-R8/` (test; scene manifest sha256 `0a916445…`, run identity `09ca62fb…`).

## Provenance and caveats

- **Names.** These names are used in every report folder: "+ Gate" marks the 22-input models, so the main pipeline R4 → R8 is "f-DAgger + Gate", R2 → R6 is "DAgger + Gate" and full DART is "DART + Gate" ([`../07_training_time_comparison/`](../07_training_time_comparison/README.md)). "f-DAgger" alone is R7, or the algorithm in [`../04_f_dagger/`](../04_f_dagger/README.md). The run ids, the `results/` folders and older documents still use "dart" for OU.
- **T1 is confounded by data coverage.** DAgger + Gate's round-1 collector (R2) drove up to 798 m away, so its DAgger data contain 3.1 M rows with the target more than 3 m from the base. Every other dataset analysed in [`../07_training_time_comparison/`](../07_training_time_comparison/README.md) (`code/analysis/far_coverage.py`: clean, OU, DAgger + Gate rounds 2–3 and f-DAgger + Gate rounds 1–3) stops at 3.1 m. DAgger without the Gate (R5, 97.7% on T1) also started from an untrained collector; its far coverage was not analysed here. The T1 differences should not be read as an effect of the Gate or of OU noise alone.
- **Validation is slightly optimistic for R5–R8**, because the round count was chosen on it. The test run is the unbiased estimate. The selection rule considers rounds 1–3 only, so a round-0 model is never chosen; over all 240 test scenes, f-DAgger + Gate's round 0 (OU + Gate, 81.2%) is better than its round 3 (75.3%).
- **Rounds 1–2 without the Gate have seed 0 only.** The extra seeds of 2026-11-03 were trained for R6 and R8 only.
- **Test.** One run, after all 24 models were frozen. T0 had been viewed once during development (the R4 test diagnosis of 2026-10-29); T1–T5 had not.
- **Statistics.** The intervals resample scenes, not training runs, so they do not cover seed-to-seed variation. The comparisons were not pre-registered; Holm's correction over all 112 of them is deliberately conservative. A difference that is significant only before correction is called indicative. Details are in "How uncertainty is reported" above.
- **Hold test is R1–R4 only.** It was part of the v1 ablation; the DAgger rows were not hold-tested.
- **Timing.** v1 training ran three runs at a time (6.3 min each); v2 ran one at a time (about 4 min). The cost table uses the seed-0 chain of the 2026-11-01 run. The OU collection time (3.3 min) is the first-to-last write in its attempts log.
- **Data copies.** Everything in `data/` is copied unchanged from `results/` and `data/datasets/`, except the v1 training log, from which the PyBullet loader messages were removed (37 lines remain). The logs are saved as `.txt` because `*.log` is git-ignored. Checkpoints and the datasets themselves are git-ignored and not copied.
