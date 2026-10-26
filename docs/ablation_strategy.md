# Ablation Strategy

Oct 24, 2026 · @Yilin · Companion to `project_plan.md` and `milestones.md`

**Status update, 26 October:** the numbered matrix, budgets, commands and reference defaults below are the archived 24 October proposal, not the active training configuration. Current clean-baseline and BC+J experiments follow [the baseline plan](mlp_baseline_training_plan.md), [the numerical research recipe](mlp_training_research.md), and [the code structure design](superpowers/specs/2026-10-26-mlp-training-structure-design.md). Preserve the general principle of paired comparisons; translate later DART/DAgger rows into their own workflows before execution.

Current overrides: use the 221/32 clean split, 8/32 validation rollouts and 64 untouched test scenarios; use full-rate random episode/timestep sampling, M1–M6 with seven torque and two mirrored wheel outputs, a per-architecture/teacher three-LR sweep, and a pilot-derived common update budget. BC+J is a separate paired objective experiment. Config templates live in `experiments/configs/`; resolved snapshots and globally unique run IDs follow the structure design. The old `experiments/overrides/`, flat checkpoint paths, stride 10, fixed reference LR, 50/300 validation counts and 21-config total below must not be used as current defaults. These overrides leave future DART/DAgger collection budgets to their declared experiment configurations.

## Archived 24 October strategy

## Summary

Every ablation changes one thing against a fixed reference model and keeps everything else the same. Variants are config overrides on one codebase (`develop`), not Git branches. The work runs in four stages (data, inputs, network, training), and each stage's winner is frozen before the next starts. That keeps the study at 21 configs instead of a full factorial of several hundred.

## Principles

1. **One reference model.** Each ablation is a delta from it, so every result answers one question.
2. **Config, not branches.** A branch only adds a capability, such as a new flag. After it merges into `develop`, the variant is a value in an override file.
3. **Store raw data.** `collect.py` and `dagger.py` save raw observations, `ee_pos` and clean expert labels at every 1 kHz step, never finished feature vectors. Inputs are built at load time (keeping every 10th step), so input ablations reuse the same datasets.
4. **One fixed training recipe.** Tune the learning rate once on the reference model (3 values), then freeze it. Every variant gets the same number of gradient steps.
5. **Paired comparisons.** All variants use seeds 0, 1, 2 and the same validation scenarios. Differences are tested scenario by scenario.
6. **Screen, then confirm.** Run each variant with 1 seed on 50 validation scenarios. Promote only the Must rows and variants close to the best to 3 seeds on the full validation set.
7. **Test once.** No test-set runs before the training freeze on 2 Nov. Each model runs once on T1–T3 on 3 Nov.

## Stages

Each stage varies one layer and fixes the winners of earlier stages. Data comes first because it is the most expensive to redo.

| Stage | What varies | Held fixed | Configs |
| --- | --- | --- | --- |
| A. Data recipe | DART, DAgger (with an equal-data BC control), data quantity, physics randomisation | Default inputs, 3×256 MLP | R + 7 |
| B. Inputs | Ladder and leave-one-out over input groups, plus one 2×2 interaction | A's frozen dataset, 3×256 MLP | 8 |
| C. Network | Size, head structure | B's best inputs | 3 |
| D. Training | Near-target sample weighting, model-selection rule | B and C winners | 2 |

**Frozen dataset for stages B–D:** D0 plus all DAgger rounds as they stand when the best MLP is tagged (31 Oct). DAgger states were collected by the reference policy, which slightly favours it. State this in the report. Only R and A4 run their own DAgger rounds; every other variant uses either D0 (stage A) or the frozen dataset (stages B–D).

## Stage B: input groups

Input groups are defined once in `core/features.py` and selected by name in the config. The network's input size follows from the groups listed. The normaliser is fitted per group on training data only.

| Group | Contents | Dim |
| --- | --- | --- |
| `arm_state` | `q_mes`, `qd_mes` (7 arm joints each) | 14 |
| `base_world` | base x, y, sin/cos of bearing | 4 |
| `target_world` | `TARGET_EE_POS` | 3 |
| `target_base` | Target in the base frame | 3 |
| `bearing_err` | sin/cos of the heading error to the target | 2 |
| `ee_err` | FK end-effector error, in the base frame | 3 |
| `phase` | Drive vs reach flag | 1 |
| `history` | Last k `arm_state` frames | 14·(k−1) |

```yaml
features:
  groups: [arm_state, target_base, bearing_err, ee_err, phase]
  history_len: 1
```

Two short series replace the 128 possible combinations:

- **Ladder (cumulative):** brief's raw inputs (`arm_state`, `base_world`, `target_world`) → base frame (`target_base`, `bearing_err`) → `+ee_err` → `+phase` → `+history` (k = 5). Shows what each group adds.
- **Leave-one-out (from the final model):** drop `ee_err`, drop `phase`, drop `bearing_err`, back to the world frame. Shows each group is still needed alongside the others.
- **One interaction:** derived features × network size (2×128 vs 3×256). The hypothesis is that FK features let a smaller network match a larger one, which also answers the efficiency question.

## Stage C: network

```yaml
model: {type: mlp, hidden: [256, 256, 256], heads: split, layer_norm: true}
```

- Size: 2×128, 3×256, 4×512. Report the smallest model on the plateau.
- Heads: split arm/wheel heads vs one shared head.
- ACT is set aside for now.

Every row reports success, parameter count and inference time per step against the 1 ms budget at 1 kHz.

## Data strategies: ablate, fix, or defer

Not every data strategy needs an ablation. Fixed choices are justified in Methodology with a citation or a quality check.

| Treatment | Strategies |
| --- | --- |
| **Ablate** (stage A) | DART noise, DAgger, data budget, physics randomisation (with vs without, judged on T3) |
| **Fix and cite** | Sobol scenario sampling in the robot's frame (target distance, relative azimuth, height), randomised start pose, 1 kHz control with no action hold, split by seed, quality checks (drop expert failures, `qmc.discrepancy`), coverage diagnostics (episodes per distance × azimuth × height cell; distance from each validation state to its nearest training state) |
| **Only if ahead of schedule** | Failure-driven sampling, regime shares, mid-episode target switches, gradually widening ranges (ADR) |

Decisions that follow from this:

- **Episode counts are powers of 2:** 512, 1,024 and 2,048, taken as prefixes of one Sobol sequence, so every learning-curve subset covers the space evenly.
- **No action hold.** Policies act at every 1 kHz physics step, as the template does. On 24 Oct the template expert reached 3/3 test targets with a hold of 1 or 2 steps, 1/3 at 3 steps and 0/3 at 4, 5, 10 and 20: its joint PD (kp 100, kd 30) is unstable when torques are held.
- **Training and T3 ranges stay separate in `settings.yaml`.** Train friction ×0.7–1.3, T3 ×0.5 / ×1.5. Train masses ±15%, T3 ±25%. Train motor gains ±20%, T3 nominal. Train sensor noise ×1, T3 ×2. The mass gap is small, and the report says so.
- **DAgger runs at least 4 rounds** and stops when validation success stops rising.

## Experiment matrix

Every configuration in the study. **Bold** cells are what differs from the reference model **R**. Stages B–D all train on the frozen dataset (D0 + DAgger rounds), so their data columns match R.

Input sets: **RAW** = `arm_state` + `base_world` + `target_world` (the brief's inputs). **BF** = `arm_state` + `target_base` + `bearing_err`. **FULL** = BF + `ee_err` + `phase`.

| ID | Question | DART | DAgger | Episodes | Physics rand. | Inputs | History | Network | Heads | Training | Compare with | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **R** | Reference model | ✓ | 4+ rounds | 2,048 | ✓ | FULL | 1 | 3×256 | split | uniform weights, closed-loop selection | — | Must |
| A1 | Plain BC | **✗** | **0** | 2,048 | ✓ | FULL | 1 | 3×256 | split | default | A2 | Must |
| A2 | BC + DART (π₀) | ✓ | **0** | 2,048 | ✓ | FULL | 1 | 3×256 | split | default | A1, R | Must |
| A3 | Equal-data control for DAgger | ✓ | **0** | **same samples as R** | ✓ | FULL | 1 | 3×256 | split | default | R | Must |
| A4 | DAgger without DART | **✗** | 4+ rounds | 2,048 | ✓ | FULL | 1 | 3×256 | split | default | R | Could |
| A5 | Data quantity | ✓ | **0** | **512** | ✓ | FULL | 1 | 3×256 | split | default | A6, A2 | Should |
| A6 | Data quantity | ✓ | **0** | **1,024** | ✓ | FULL | 1 | 3×256 | split | default | A5, A2 | Should |
| A7 | Physics randomisation | ✓ | **0** | 2,048 | **✗** | FULL | 1 | 3×256 | split | default | A2 (on T3) | Should |
| B1 | Ladder 0: brief's inputs | frozen | frozen | frozen | frozen | **RAW** | 1 | 3×256 | split | default | B2 | Should |
| B2 | Ladder 1: base frame | frozen | frozen | frozen | frozen | **BF** | 1 | 3×256 | split | default | B1, B3 | Should |
| B3 | Ladder 2: + `ee_err` (= leave out `phase`) | frozen | frozen | frozen | frozen | **BF + `ee_err`** | 1 | 3×256 | split | default | B2, R | Should |
| B4 | Ladder 4: + history | frozen | frozen | frozen | frozen | FULL | **5** | 3×256 | split | default | R | Should |
| B5 | Leave out `ee_err` | frozen | frozen | frozen | frozen | **FULL − `ee_err`** | 1 | 3×256 | split | default | R | Should |
| B6 | Leave out `bearing_err` | frozen | frozen | frozen | frozen | **FULL − `bearing_err`** | 1 | 3×256 | split | default | R | Should |
| B7 | World frame | frozen | frozen | frozen | frozen | **RAW + `ee_err` + `phase`** | 1 | 3×256 | split | default | R | Should |
| B8 | Interaction: raw inputs, small net | frozen | frozen | frozen | frozen | **RAW** | 1 | **2×128** | split | default | B1, C1, R | Should |
| C1 | Small network | frozen | frozen | frozen | frozen | FULL | 1 | **2×128** | split | default | R, B8 | Should |
| C2 | Large network | frozen | frozen | frozen | frozen | FULL | 1 | **4×512** | split | default | R | Should |
| C3 | Shared head | frozen | frozen | frozen | frozen | FULL | 1 | 3×256 | **shared** | default | R | Should |
| D1 | Near-target weighting | frozen | frozen | frozen | frozen | FULL | 1 | 3×256 | split | **near-target weights** | R | Should |
| D2 | Selection by validation loss | frozen | frozen | frozen | frozen | FULL | 1 | 3×256 | split | **lowest-loss checkpoint** | R | Should |

Notes:

- Ladder 3 (BF + `ee_err` + `phase`) is R, and "leave out `phase`" is B3, so neither has its own row.
- The 2×2 interaction uses B1, B8, C1 and R: inputs RAW vs FULL × network 2×128 vs 3×256.
- D2 needs no extra training. It picks a different checkpoint from R's training runs, so it only costs evaluation.
- A7 is the only stage A row judged mainly on T3. The others are judged on validation, then T1 and T2.
- A7 is compared with A2 at DAgger 0, so it needs no DAgger rounds of its own.
- Total: R plus 20 variants = 21 configurations. Must rows: 4. At 3 seeds each that is 63 training runs (60 excluding D2), before cutting Could rows.

## Layout and tracking

```
final/robotnik_arm_task/experiments/
├── ablations.yaml     registry: name, stage, override file, priority, dataset
├── overrides/         one small YAML per variant, only the keys it changes
└── run.py             python -m experiments.run B-loo-ee_err --seeds 0 1 2
```

- **Run names:** `<stage>-<series>-<variant>-s<seed>`, e.g. `A-dart-off-s0`, `B-ladder-ee_err-s1`, `C-size-2x128-s2`.
- **Per run:** `checkpoints/<run_id>/` holds the merged config, the Git commit hash and the normaliser statistics.
- **`results.csv` adds** `exp`, `stage`, `seed` and `git_sha` to the plan's columns. The header must change before the first real row is written.
- **Plots** group by `exp` and show Wilson intervals over seeds.

## Test-set protection

- Until 2 Nov, `experiments/run.py` evaluates on `val` only.
- `evaluate.py` refuses `--split T1/T2/T3` when a test row for that `exp` and seed already exists, unless `--allow-rerun` is passed.
- The commit used on 3 Nov is tagged `final-test`, and the README points to it.

## Budget

21 configs (see the experiment matrix). After screening, most are confirmed at 3 seeds. Training a small MLP takes minutes, so closed-loop evaluation is the bottleneck. On 24 Oct, 12 parallel headless template runs finished 8–21 s simulated episodes in 5–14 s of wall time each; re-measure when the expert is verified on 26 Oct. If the budget does not fit, cut the "only if ahead" rows first, then the Should rows in reverse order of cost.
