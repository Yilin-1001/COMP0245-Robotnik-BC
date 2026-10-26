# Clean BC Baseline and MLP Architecture Search

Planning date: 26 October 2026. Status: proposed implementation and experiment plan; training has not started as part of this planning task.

The objective is to obtain a working, evaluated behavior-cloning baseline today, then identify the best MLP among a small, controlled candidate set. Use clean demonstrations only. DART, DAgger, new collection, recurrent policies and teacher modification are outside this first training stage.

Unified rules (26 October): this plan owns the clean scenario split and evaluation predicate. The [research recipe](mlp_training_research.md) owns numerical training settings and the separate BC+J experiment; the [structure design](superpowers/specs/2026-10-26-mlp-training-structure-design.md) owns code locations, configuration resolution, run IDs and artifacts. Dataset manifests own recorded teacher/action interfaces. The settings below incorporate the revised recipe; older total-plan and ablation-matrix defaults do not override them. Baseline architecture screening remains action-loss-only BC.

## 1. Current implementation and resources

- `training/dataset.py`, `training/policies.py`, `training/train_mlp.py` and `evaluation/evaluate.py` still contain `NotImplementedError` placeholders.
- `core/features.py` already implements the deployable 21-dimensional Markovian and 26-dimensional Stateful inputs.
- The inspected simulator environment uses Python 3.10.20 and does not currently contain PyTorch.
- The initial inspection exposed 24 CPU threads and about 30 GiB RAM, with approximately 16 GiB available, and could not reach the GPU driver. The later research review reports a usable RTX 5070 Ti Laptop GPU; this earlier CPU-only assumption is superseded.

Verify a stable PyTorch ≥ 2.7 CUDA 12.8 build and `sm_120` support in the simulation environment, following the research recipe; use CPU as fallback. Record the actual dependency versions and device. Do not promise a complete search runtime before measuring training throughput and closed-loop evaluation cost.

## 2. Data and model-selection split

Use `Markovian/Clean_253` and `Stateful/Clean_253` for the primary full-range architecture search. Keep `Clean_223` as the paired data-selection comparison. This revises the initial suggestion to make 223 the default: a subsequent coverage audit found that filtering chatter removed **all 27 successful training scenarios below target height 0.22753 m**. The 223 cohort's minimum height is 0.22753 m, whereas the 253 cohort reaches 0.10014 m. Seven original test scenarios lie below that reduced training minimum.

Both options have limitations. The 253 cohort retains high-frequency hold labels, while the 223 cohort loses an entire low-target interval. Neither lower chatter counts nor occupied coarse coverage cells establish that one is automatically better. Keep the chatter flags, report height-stratified validation outcomes, and use the practical comparison to assess this tradeoff.

The existing 64 test scenarios remain held out. Do not select architectures, learning rates, checkpoints, normalizers or training duration using them.

Create one internal split from the 253 training scenario IDs:

| Cohort | Model-fitting scenarios per teacher | Internal validation scenarios per teacher | Final test scenarios per teacher |
| --- | ---: | ---: | ---: |
| Primary `Clean_253` search | 221 | 32 | 64, unopened for model selection |
| Later `Clean_223` data-quality comparison | 196 | The same 32, read from `Clean_253` | The same 64 |

Select the 32 validation scenarios once, reproducibly, with NumPy `default_rng(20261026)`: divide normalized target distance into two bins, height into four bins and relative azimuth into four sectors. Sort cells lexicographically by `(distance_bin, height_bin, azimuth_bin)`, sort each cell's scenarios by ID, and draw one with `rng.integers(len(cell))`. The current 253 cohort has 7–8 scenarios in each of these 32 cells, so each retains fitting data. The verified selection contains five targets below 0.22753 m. Use the same IDs for both experts and every architecture. Save the split manifest and source hashes before training.

Only 27 of those validation IDs belong to `Clean_223`; subtracting them leaves 196 fitting trajectories for that comparison. Load all 32 validation trajectories from the corresponding `Clean_253` folder, including its five excluded-from-223 scenes. Do not silently replace those low targets with easier validation scenes. This is validation of the full task range for both fitting strategies, not a claim that the reduced dataset covers the missing interval.

The expected validation IDs for this rule are: `train-0143`, `train-0219`, `train-0092`, `train-0188`, `train-0091`, `train-0243`, `train-0024`, `train-0104`, `train-0103`, `train-0023`, `train-0000`, `train-0140`, `train-0251`, `train-0195`, `train-0068`, `train-0236`, `train-0185`, `train-0201`, `train-0250`, `train-0082`, `train-0145`, `train-0085`, `train-0154`, `train-0162`, `train-0081`, `train-0005`, `train-0130`, `train-0186`, `train-0093`, `train-0245`, `train-0226`, `train-0182`. This plan records the proposed split; implementation must materialize and verify its manifest before training.

Split whole episodes. Never split adjacent timesteps from one episode across fitting and validation. Fit all learned normalization statistics on the actual fitting cohort only, not on the complete 223/253 directory or the test set.

## 3. First baseline definition

Train two independent policies, one for each teacher. Do not merge labels from the two teachers into an unlabelled dataset.

| Property | Markovian student | Stateful student |
| --- | --- | --- |
| Current-observation input | 21 deployable features | 26 deployable features |
| Output | 7 commanded arm torques + 4 wheel-speed targets | Same |
| First hidden widths | `[128, 128]` | `[128, 128]` |
| Previous action or observation history | Excluded | Excluded |
| Expert internal memory as input | Excluded | Excluded |

Use a shared MLP trunk and two linear heads of sizes seven and two. Expand right/left wheel predictions to the four-command order `[r, l, l, r]`, so the deployable action still has 11 values. The shared recipe is `Linear -> LayerNorm -> SiLU` for each hidden layer, with linear outputs. LayerNorm and activation remain fixed during the architecture screen; their usefulness is a later ablation, not assumed from this plan. Keep baseline and BC+J entry points separate as specified in the structure design.

The source Markovian recordings use `lead_time=0.50 s`, version `markovian-lt0.50-v1`. Learn those recorded labels; do not accidentally evaluate against a newly instantiated default `lead_time=0.15 s` teacher.

The Stateful MLP is intentionally a memoryless baseline for a stateful teacher. Its limitations may reflect unavailable memory or orientation information, rather than insufficient network size. Do not conclude that progressively larger MLPs must eventually reproduce the original controller.

## 4. Feature cache, normalization and sampling

### Cache construction

- Reuse the same feature builder in offline preprocessing and online policy inference.
- Align feature index `t` with `a_expert[t]` using the first T observations of a T-action episode.
- Use the deployable FK estimate, not simulator-truth EE position or full orientation as student inputs.
- Cache deterministic, unnormalized features and labels once, as float32 arrays with episode offsets and source/feature hashes. Preserve the float64 raw recordings.
- Do not repeatedly decompress an entire `.npz` or compute FK inside each minibatch.
- Do not load or cache test trajectories for architecture selection.

Full-rate feature and label caches for these low-dimensional datasets should be much smaller than the raw diagnostic recordings; measure actual memory use before choosing an in-memory or memory-mapped representation.

### Normalization and loss

Standardize inputs using fitting-only statistics. Standardize each output channel separately using fitting-only statistics and a recorded scale floor to avoid amplifying nearly constant channels. Initial scale-floor candidates are 0.1 Nm for torque and 0.05 rad/s for wheel speed; inspect the resulting scales and freeze them across architectures within a teacher/data condition.

Compare Huber loss with delta=1 against MSE once in the M2-A pilot, then freeze the action loss before architecture screening. The Huber candidate is:

```text
loss = 0.5 * mean_huber(arm_prediction, arm_target)
     + 0.5 * mean_huber(wheel_prediction, wheel_target)
```

The separate means give the two action groups equal weight rather than letting the seven torque dimensions outweigh the two independent wheel dimensions solely by dimension count. Fit wheel output scales on the right/left labels; expand to four commands after inverse normalisation. Log physical-unit errors after inverse normalization as well as standardized loss. The loss and scale choices are project hyperparameters, not established optimal values. Baseline loss has no Jacobian term; the later paired BC+J experiment follows research §4.4.

### Sampling and training budget

Sample an episode uniformly, then a timestep uniformly within that episode. This prevents a longer Stateful trajectory from automatically receiving more weight. Preserve all navigation, approach and holding samples; do not introduce changing phase weights during the structure screen.

Use randomly sampled full-rate observations for MLP training. Do not automatically apply the old `train_stride=10`: a fixed stride can alias high-frequency control behavior, and full-rate sampling is affordable for these feature dimensions. Online inference still runs every 1 ms regardless of offline sampling.

Control optimizer updates, batch size and sampling rules across architectures. Equal epoch counts do not imply equal update budgets when the teachers have different total trajectory lengths.

## 5. Candidate architecture matrix

These six candidates vary depth, width and taper while keeping features, normalization, activation, heads, selected action loss and update budget fixed. Tune LR separately for each architecture/teacher using the same three-value grid, then use each architecture's chosen LR for closed-loop comparison.

| ID | Hidden widths | Parameters with 21 inputs | Parameters with 26 inputs | Purpose |
| --- | --- | ---: | ---: | --- |
| M1 | `[64, 64]` | 6,409 | 6,729 | Small capacity/latency reference |
| M2 | `[128, 128]` | 21,001 | 21,641 | First baseline |
| M3 | `[256, 256]` | 74,761 | 76,041 | Width increase |
| M4 | `[128, 128, 128]` | 37,769 | 38,409 | Depth increase at moderate width |
| M5 | `[256, 256, 256]` | 141,065 | 142,345 | Larger reference from the earlier plan |
| M6 | `[256, 128, 64]` | 48,265 | 49,545 | Tapered representation |

Parameter counts include linear biases and affine LayerNorm parameters, and use linear seven- and two-output heads. The same architecture has identical parameter counts for BC and BC+J.

Unified training recipe, with loss and budget frozen after the pilot:

| Setting | Initial value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | Best of {1e-3, 3e-4, 1e-4} per architecture/teacher on the offline proxy |
| Weight decay | 1e-4 |
| Batch size | 1024 |
| Training updates | Common budget from the M2-A constant-LR pilot plateau, approximately 1.5× plateau updates |
| LR schedule | Short linear warm-up (e.g. 500 updates), then cosine over the frozen budget |
| Offline validation logging | Every 500 updates |
| Checkpoints | Every 2,000 updates and the terminal update |
| Screening rollouts | Evaluate saved checkpoints on the eight scenarios only for the selected LR of each architecture/teacher, after the offline LR sweep |
| Screening initialization seed | 0 |
| Finalist seeds | 0, 1 and 2 |

Run the M2-A constant-LR pilot at 1e-3 and 3e-4 for 50k updates and log every 1k updates, following research §4.2; use it to choose the budget and compare the loss candidates. Freeze the common budget before the matrix, rather than retaining the old fixed 10k guess. If the pilot has not plateaued or the budget is infeasible, document and resolve that before screening. Do not give larger models a different update budget without reporting that as a separate experiment.

## 6. Closed-loop validation and selection

Offline validation loss alone does not establish that a robot policy can complete the task. The robot imitation study by [Mandlekar et al.](https://arxiv.org/abs/2108.03298) discusses differing training/evaluation objectives and model-selection sensitivity. Use closed-loop validation to select models.

Freeze an eight-scenario screening subset of the 32 validation scenarios, balanced across relative target directions. Use it for frequent rollout checks. Evaluate the full 32 validation scenarios for each architecture's best screening candidate, and for the final seed comparison.

Run the student alone: no teacher actions, true `q_des`, expert phase or done flags may control the rollout. For B, compute the known fixed navigation goal once from task context; this does not require running the expert feedback controller.

At inference:

- Build the same deployable features, apply the saved fitting normalizer and invert the output normalization.
- Execute at 1 kHz with the source manifest's action order and execution limits: current Clean_253 A/B use ±30 N·m and ±6 rad/s. Freeze the interface in the checkpoint and reject runtime mismatches. Do not lower the control rate or hold actions to hide slow inference.
- Use one CPU inference thread initially; measure feature construction plus network inference, including median, p95 and p99 latency. A 1 ms control interval is the target, not an unmeasured guarantee.
- Use the collection's 40 s cap and three-second post-reach recording convention, without terminating on the known misleading stall diagnostic.

A student has no expert `reach_active` or `nav_done` output. Define its task reach from true EE error below 10 mm and t >8 s, then evaluate the full post-reach window and final one-second hold. Keep true EE information evaluation-only. Do not run an expert alongside the student to gate success or control. Report base-goal accuracy separately if required, and use this same evaluation predicate for any teacher reference comparison. Also record the ungated first arrival time.

Rank candidates by:

1. Sustained-hold success on the full validation scenarios.
2. Reaching success and final/hold error.
3. Stability, action saturation, command variation and phase-specific failures.
4. Measured end-to-end inference cost and model size when task performance is close.

Also record the ungated first arrival time: the 8 s minimum masks differences between fast Markovian rollouts. Report arm and wheel MAE/RMSE separately in physical units, including navigation, approach and hold breakdowns.

Differences of one episode on 32 validation scenarios are small. Prefer the smaller/faster model when repeated-seed task performance is effectively tied; do not call a one-episode difference a proven improvement.

If all architectures fail in closed loop, inspect feature/action alignment, scaling, action ordering, task context and distribution shift before expanding the network grid. A low action loss with zero task success is a failed control baseline, not evidence of a good architecture.

## 7. Today's execution order

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 1. Make the pipeline run | Install/verify PyTorch; implement loading, policy save/load and closed-loop evaluation; freeze the internal scenario split | Correct data alignment, normalizer round trip, checkpoint round trip and finite bounded inference |
| 2. Obtain one baseline | Train M2 on Markovian fitting data first; inspect a small fitting-scenario rollout check and validation rollouts | A saved checkpoint with actual offline and closed-loop results |
| 3. Screen structures | Six candidates × two teachers × three LRs, seed 0; one chosen LR per architecture/teacher goes to rollouts | 36 offline runs and 12 architecture/teacher closed-loop candidates |
| 4. Check repeatability | For each teacher, select the top two and add seeds 1 and 2 at its frozen LR | Three seeds per finalist; 8 additional runs, 44 including the offline screen, excluding pilots and BC+J |
| 5. Optional data-quality comparison | Train the selected architecture on `Clean_223` using 196 fitting scenarios and the same 32 validation IDs from `Clean_253` | Comparison of wider height coverage with chatter versus fewer paired chatter-free scenarios |
| 6. Freeze and evaluate | Freeze the chosen architecture/recipe and, if time permits, refit on all 253 primary training scenarios | Fixed final checkpoints and one planned evaluation on the original 64 test scenarios |

For optional final refitting, freeze the architecture, objective, LR schedule, update-count rule and seed set from validation before using all 253 primary trajectories (or all 223 for a separately declared reduced-data finalist). Create a new `fit_stage=refit` run, start from the declared seeded initial weights, and refit normalizers on that final training cohort. Train to the frozen terminal update and save `final.pt`, with no validation-based selection or test statistics. Independent-validation development runs use `fit_stage=selection` and save validation-selected `best.pt`. If final refitting is skipped, freeze and evaluate the selected `best.pt`. BC+J, if selected separately, freezes λ_J before refit and recomputes standardised derivative targets with its refitted scales.

If time is limited, obtain the verified M2-A baseline before expanding the search or adding optional comparisons. Do not sacrifice an evaluated working policy to complete a large unverified search table. The declared three-value per-architecture LR sweep is part of screening; any extra LR, activation or LayerNorm search is a separately labelled optimisation stage.

The optional 223-versus-253 comparison changes both scene membership and the amount of data, as well as retaining/excluding chatter. Interpret it as a practical data-selection comparison, not an isolated causal measurement of chatter alone.

## 8. Reporting and interpretation

For every run save architecture, teacher/data version, split hash, feature hash, normalizers, sampling rule, training budget, seed, source revision, optimizer settings, checkpoints, offline curves and closed-loop metrics. Use a results CSV with one row per checkpoint evaluation and retain per-scenario outcomes.

Use the structure design's unique run ID: `<regime>-<teacher>-<objective>-<cohort>-<architecture>-lr<lr>-s<seed>-<fit_stage>[-r<round>]-<config_id>`. This distinguishes BC/BC+J, Clean_253/Clean_223, selection/refit and LR candidates. The compact shared `results/results.csv` is versioned; detailed per-run metrics, figures, datasets and checkpoints stay local.

Within-teacher comparisons test architecture choices. Comparing an A-trained 21-input policy with a B-trained 26-input policy changes the teacher, information and trajectories, and is an end-to-end approach comparison rather than an isolated test of recurrence or memory.

The deliverable is the **best validated MLP within the tested search space and budget**, plus a reproducible baseline. It is not a globally optimal network, and clean-only performance does not establish disturbance recovery.

World x/y starts currently span +/-0.5 m, but relative target distance spans 0.5–3 m and relative azimuth and base yaw span nearly the full circle. For A on a uniform flat floor, expanding x/y alone mostly adds translated copies of similar relative geometry. For B, the world-origin goal rule makes translations change its desired navigation geometry, so larger-start-position generalization needs a separate check. Fixed initial arm poses, zero initial velocity, missing low targets after filtering and absent recovery states are additional coverage restrictions; do not equate a larger world-coordinate box with solving them.

## Sources and local contracts

- [Paired dataset subsets](clean_dataset_subsets.md)
- [Clean collection report](clean_collection_report.md)
- [Expert dataset specification](expert_datasets.md)
- [Expert operating limits](expert_operating_limits.md)
- [Research reference guide](references.md)
- [PyTorch Huber loss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
- [PyTorch AdamW documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
