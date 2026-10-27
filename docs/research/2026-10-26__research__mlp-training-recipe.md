# MLP Baseline Training: Research Review and Practical Recipe

Reviewed: 26 October 2026. Companion to the [MLP baseline training plan](../plans/2026-10-26__plan__mlp-baseline-training.md). Status: research and local measurements only. No policy has been trained.

This note checks the plan against published work and against measurements of the recorded data and teachers. It recommends a small number of changes. Most of the plan stands: whole-episode splits, fitting-only normalisers, separate students per teacher and closed-loop model selection all agree with the literature below. The changes concern four things: what the network must get exactly right, how to set the training budget and learning rate fairly, two cheap offline diagnostics, and the order of the first improvements after the baseline.

The baseline remains ordinary MLP behaviour cloning with action loss only. After architecture and seed selection, add the damping-Jacobian loss as a separately reported experiment for both teachers. Offline perturbation relabelling is excluded from this plan. Later DART and DAgger experiments retain a plain-BC control so that their contributions can be measured separately.

The [MLP training structure design](../specs/2026-10-26__design__mlp-training-structure.md) defines the unified implementation target: separate baseline and Jacobian training entry points, separate later DART/DAgger workflows, and shared low-level models, data processing, persistence and evaluation. It also defines configuration resolution, unique run IDs and selection/refit artifacts. This research note owns the numerical training recipe; the baseline plan owns the exact scenario split and evaluation predicate; source dataset manifests own action order and execution limits. The current training files remain skeletons.

## 1. Summary of recommended changes

| Plan item | Recommendation | Reason (section) |
| --- | --- | --- |
| Hardware assumption: CPU only | `nvidia-smi` now reports an RTX 5070 Ti Laptop GPU (12 GiB). Install a stable PyTorch ≥ 2.7 CUDA 12.8 wheel, check that `sm_120` is in `torch.cuda.get_arch_list()`, and keep CPU as the fallback | §2.6 |
| Fixed 10,000 updates | Run one constant-LR pilot curve first and read the plateau from it. 10k updates × 1024 is about 3 passes over 3 M highly correlated steps, which is probably too few for the precision required | §2.2, §4.2 |
| One learning rate (3e-4) for all architectures | Tune LR per architecture on the offline proxy over {1e-3, 3e-4, 1e-4}, then run closed-loop rollouts only for each architecture's best LR | §3.5 |
| Huber δ = 1 by default | Compare it against MSE once, in the M2 pilot. Teacher A's labels are exact, deterministic function values, and its "outliers" (saturation, chatter) are correct labels | §2.1, §4.3 |
| Four wheel outputs | Predict two (right, left) and duplicate them as `[r, l, l, r]`. In all 120 checked recordings the labels satisfy FR = BR and FL = BL exactly | §2.4 |
| Offline metric: RMSE/MAE | Also add a **signed hold-phase torque bias** per joint, with a gate of roughly 0.5 N·m, and the **learned damping** ∂τ̂/∂qd compared with the teacher's exact −30 N·m·s/rad | §2.1–2.2, §5 |
| Inference timing | Not a selection criterion between M1–M6. All six run in 21–40 µs in NumPy, and FK takes 6 µs | §2.5 |
| Simulation and evaluation workers | Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` (and `torch.set_num_threads(1)`) in every rollout process. The Markovian expert takes 10 ms per call with default threading and 51 µs pinned | §2.5 |
| After the baseline | Add a damping-Jacobian (Sobolev) loss for A and B as a paired ablation against plain BC. Keep the planned DART/DAgger comparisons separate | §3.3–3.4, §4.4, §6 |

## 2. What the teachers and data imply (local evidence)

### 2.1 Torque labels have a known structure

Both teachers compute the arm command as a gravity-compensated joint PD law, clipped to ±30 N·m ([expert.py](../../final/robotnik_arm_task/data_collection/expert.py), [robotnik_arm_control.py](../../final/robotnik_arm_task/robotnik_arm_control.py)):

```text
tau = clip( kp (q_des - q) + kd (qd_des - qd) + g(q),  -30, 30 ),   kp = 100, kd = 30
```

For A, `q_des` is `q + lead * qd_des` or `Q_HOLD`. `qd_des` comes from differential IK on positions only. For B, `q_des` is the integrated internal reference. In both cases nothing except the `-kd qd` term depends on `qd_mes`. This gives three consequences:

1. **The teacher's damping Jacobian is known exactly.** Wherever the torque is unclipped, ∂τ/∂qd_mes = −30·I. Clipped joints have a zero row. For B this holds at a fixed internal state. These derivative labels cost nothing to compute (§3.3).
2. **Holding gives the value loss almost no information about damping.** Settled holds have joint speeds of about 0.005–0.008 rad/s ([clean collection report](../reports/2026-10-26__report__clean-expert-data-collection.md)). Across that range, the damping term changes torque by only about 0.2 N·m. A network can fit hold torques well with the wrong slope in `qd`, and closed-loop damping depends on that slope.
3. **Gravity dominates hold torques.** The network must reproduce g(q) accurately as a function of seven angles, at the 0.1 N·m scale (§2.2).

The joint-3 chatter is a period-2 (500 Hz) oscillation. It fits a simple stability argument: an explicit velocity feedback of gain kd at step dt becomes oscillatory-unstable when kd·dt/I_eff > 2, i.e. effective inertia below about 0.015 kg·m² here. The collection report already hypothesises this cause, and it has not been tested. If it holds, two things follow. First, the chatter labels are not label noise: they are correct values of the teacher function at large |qd|. Second, a student whose learned |∂τ/∂qd| exceeds 30 has less stability margin than the teacher. That is a concrete failure mode to measure, not a reason to discard data.

### 2.2 The precision a student needs

The DART stress test (`results/expert_noise_robustness.csv`) found that the Markovian teacher **stops 16–60 mm short under constant arm torque offsets above 1 N·m**. A student's systematic error near the target acts like such an offset. As a working gate before rollouts, retain the signed hold-phase torque bias for diagnosis, but require its **absolute magnitude** per joint to be ≲ 0.5 N·m. RMSE alone does not reveal systematic signed bias; zero-mean error can still affect closed-loop stability.

| Quantity (Clean_253 train) | Markovian (A) | Stateful (B) |
| --- | --- | --- |
| Episodes / control steps | 253 / 3,045,616 | 253 / 3,356,970 |
| Episode length (s), mean [min, max] | 12.0 [11.0, 18.4] | 13.3 [11.0, 20.3] |
| Torque label std per joint (N·m) | 3.6, 3.0, 6.5, 3.0, 5.7, 2.0, 5.6 | 4.4, 2.3, 6.1, 2.7, 5.7, 2.1, 5.6 |
| Wheel label std (rad/s) | 1.43 | 1.51 |
| Steps with saturated torque, joints 2 / 3 | 2.1 % / 2.7 % | 0.9 % / 3.0 % |
| Steps with reach active (A teacher flag) | 51 % | — |

A 0.5 N·m bias is 8–25 % of one torque standard deviation. This is a precise regression problem in the region that decides success. That is why the training budget and learning rate deserve a pilot rather than a fixed guess.

### 2.3 Teacher A is a deterministic target

Re-querying `MarkovianExpert(lead_time=0.50)` on recorded observations reproduced all recorded A labels with zero difference (1,572 steps of `train-0000`). This supports the controller's structural property that A's label is a deterministic function of the deployable observation. Student errors should therefore be investigated as approximation or optimisation errors, with feature and execution checks first. Larger models are a reasonable thing to test for A.

B's label also depends on internal memory the student does not see, as the plan already notes. Matching its damping Jacobian does not remove that information limitation.

### 2.4 Wheel labels have two degrees of freedom

Both teachers output `[rw, lw, lw, rw]`. Checked recordings have |FR − BR| = |FL − BL| = 0 exactly. Predicting four free values lets the student command front and back wheels on one side differently, which makes the skid-steer base fight itself. A 2-output wheel head mirrored to four commands removes that error mode, and it is an architectural prior that is easy to justify.

The wheel commands are also discontinuous in the teacher: the `min_lin_vel` clip, the `pos_tol`/`ang_tol` switches and the reach gate at 1.5× those tolerances. A smooth MLP will round these edges. If closed-loop failures concentrate at the base-to-reach transition, consider adding explicit features: distance error, heading error as sin/cos, and standoff distance. These are computed from the same deployable quantities and would be tested as a feature ablation.

### 2.5 Compute measurements

Measured on this machine (24 threads), single sample, float32 NumPy, LayerNorm + SiLU, 26 inputs:

| Model | M1 | M2 | M3 | M4 | M5 | M6 |
| --- | --- | --- | --- | --- | --- | --- |
| Median forward (µs) | 21.7 | 22.6 | 26.0 | 34.0 | 39.6 | 34.8 |
| p99 (µs) | 24.2 | 25.3 | 29.0 | 37.3 | 43.2 | 38.0 |

`DeployableFK.ee_pos` takes 5.8 µs per call. Feature construction plus inference is therefore below 5 % of the 1 ms budget for every candidate. Per-layer Python overhead dominates, not FLOPs. Report latency for the efficiency question, but do not expect it to separate these architectures. PyTorch eager inference on one sample typically has more fixed overhead than NumPy. Export weights to NumPy for rollouts, or measure the torch path explicitly.

**Thread pinning.** `MarkovianExpert.act` took 9,983 µs per call with default OpenBLAS threading and 51 µs with `OMP_NUM_THREADS=1`. That is about 200× slower, consistent with thread start-up and contention on tiny LAPACK calls. No script in the repository sets thread limits. Set them at the top of every simulation, relabelling and evaluation worker, especially when running parallel processes.

### 2.6 GPU

The plan recorded that `nvidia-smi` could not reach the driver. On 26 October it reports an **RTX 5070 Ti Laptop GPU, 12 GiB, driver 595.99, CUDA 13.2**. Blackwell (`sm_120`) needs PyTorch ≥ 2.7 built with CUDA 12.8 or later. Older wheels install but cannot run kernels on it ([PyTorch forum](https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621)).

The full float32 feature and label cache is small (3.4 M × 37 × 4 B ≈ 0.5 GB per teacher), so it can sit on the GPU. Episode-then-timestep sampling can then be done with on-device index arithmetic, without a `DataLoader`. Training runs should then take minutes, not hours, but measure this in the pilot before freezing the grid. If CUDA setup costs more than about 30 minutes, use the CPU: these networks are small enough.

## 3. Relevant research and what it implies here

### 3.1 Compounding error is the expected failure mode, even with low loss

- Ross, Gordon and Bagnell show that offline BC error can compound quadratically with horizon, and that DAgger reduces this to linear ([AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)).
- Simchowitz, Pfrommer and Jadbabaie prove a stronger negative result for **continuous actions**. With stable dynamics and a smooth deterministic expert, a smooth deterministic imitator can still have execution error that grows **exponentially** in horizon relative to its training error. Exceptions are when the expert's trajectory distribution is sufficiently spread out, or when richer policy classes such as action chunking or diffusion are used ([COLT 2025](https://proceedings.mlr.press/v291/simchowitz25a.html); [arXiv](https://arxiv.org/abs/2503.09722)).

**Implication.** A 1 kHz rollout of 12–40 s has 12k–40k steps, all from clean demonstrations that cluster tightly around each teacher trajectory. That is exactly the regime these results warn about. A clean-only MLP that has low loss and still fails, or drifts in holds, is a plausible outcome, not necessarily a bug. The plan's debugging checklist still comes first. The remedies retained here are derivative matching (§3.3), disturbance-driven collection with DART, and on-policy relabelling with DAgger (§3.4).

### 3.2 Learned approximations of known controllers

Work on approximating model-predictive controllers with networks treats the network as a replacement controller with an error budget. Hertneck et al. train a network to imitate a robust MPC. They verify statistically, with Hoeffding bounds over closed-loop samples, that its error stays within the margin the robust design tolerates ([IEEE L-CSS 2018](https://arxiv.org/abs/1806.04167)). Karg and Lucia show that deep ReLU networks can represent piecewise-affine MPC laws efficiently, and they evaluate closed-loop performance, not only fitting error ([IEEE T-Cyb 2020](https://arxiv.org/abs/1806.10644)).

**Implication.** The same logic applies here, with an empirical margin in place of a formal one. The teacher tolerates constant torque offsets of about 1 N·m (§2.2), so the student's hold-phase bias must sit well inside that. This links an offline number to a closed-loop requirement, which is the kind of specific justification the marking criteria reward.

### 3.3 Matching derivatives: Sobolev training and TaSIL

- Czarnecki et al. train networks on target derivatives as well as values. They report better generalisation, especially with little data, including in policy distillation ([NeurIPS 2017](https://arxiv.org/abs/1706.04859)).
- TaSIL adds penalties on differences between the learner's and expert's higher-order Taylor terms to the BC loss. For incrementally stable experts, a small TaSIL loss on expert trajectories guarantees small imitation error on the learner's own trajectories. TaSIL outperformed BC, DART and DAgger baselines on MuJoCo tasks ([Pfrommer et al., NeurIPS 2022](https://arxiv.org/abs/2205.14812)).

**Implication.** Full TaSIL needs expert Jacobians with respect to all inputs. Here, the part that matters most for stability is available for free and is exact (§2.1): ∂τ/∂qd = −30·I where unclipped. A practical, well-motivated addition is:

```text
L = L_action + lambda_J * mean_over_subbatch || d tau_hat / d qd  -  J_teacher ||_F^2
J_teacher = -kd * diag(unclipped mask)    (convert to standardised units: multiply by sigma_qd_j / sigma_tau_i)
```

Compute it on a sub-batch of about 256 samples with `torch.func.jacrev` or 7 forward-mode JVPs, retaining differentiation through the Jacobian for training. Start with λ_J chosen so that the weighted term is about 10 % of the action loss at initialisation. Compare against the plain baseline on the same seed using the protocol in §4.4. The experiment matches only the 7×7 torque-versus-velocity Jacobian; full position Jacobians and higher-order TaSIL terms are outside this experiment. The qd-Jacobian label is exact for both teachers, for B conditional on its current internal state.

Even without the penalty, **measure** the learned diagonal ∂τ̂_j/∂qd_j on validation hold states for every model. Compare with −30 on clearly unclipped rows and zero on verified strictly saturated rows, using the valid-entry mask in §4.4; report omitted rows separately. Large derivative discrepancies flag possible damping or chatter problems for closed-loop checks, rather than proving that a rollout will fail.

### 3.4 Compatibility with later DART and DAgger experiments

DART injects noise during expert collection, so demonstrations include states reached under disturbed execution while retaining the supervisor's corrective labels ([Laskey et al., CoRL 2017](https://proceedings.mlr.press/v78/laskey17a.html)). Fixed noise injection is DART-style collection; reproducing full DART also requires its noise-selection procedure. DAgger collects states visited by the current learner or learner/expert mixture, queries the expert there, aggregates the data and retrains ([Ross et al., AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)). Both remain in the later data-collection plan.

The damping-Jacobian loss changes the training objective and can be used with either dataset. It may also change the learner's errors and the trajectories collected by DAgger, so any extra benefit must be measured. Use these plain-BC comparisons first:

| Variant | Data | Jacobian loss | Question |
| --- | --- | --- | --- |
| BC | Clean demonstrations | Off | How well does the tuned MLP baseline work? |
| BC+J | Same clean demonstrations | On | What does damping supervision add? |
| BC+DART | Clean plus DART-style collection | Off | What does disturbed collection add over BC? |
| BC+DART+DAgger | Previous data plus this learner's DAgger rounds | Off | What does on-policy aggregation add over BC+DART? |

Combining +J with DART or DAgger is a later paired experiment against the corresponding no-J variant. Hold architecture, validation scenarios, seed set and declared training/collection budgets fixed. Each full DAgger variant must collect with its own current policy; training variants on one frozen aggregate dataset compares training objectives only. Record data sources and sampling proportions, and retain an equal-data expert-collection control for DAgger as required by the existing plan.

If +J is carried into these later runs, determine the valid torque rows using the internal-saturation rules in §4.4. Execution `clip_flags` and noisy executed actions are not Jacobian saturation labels. Recompute standardised derivative targets using the run's saved fitting-only normalisers. For B, the expert memory must evolve along the actual collection trajectory at every step, as specified in [the dataset contract](../specs/2026-10-26__spec__expert-datasets.md#6-clean-noise-augmented-and-dagger-collection).

### 3.5 Fair architecture comparison and model selection

- The Google *Deep Learning Tuning Playbook* treats the learning rate as a **nuisance hyperparameter**. It must be tuned separately for each architecture before comparing architectures, because the best learning rate depends on the architecture ([Godbole et al.](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific)). With one fixed LR, a 3×256 network might lose to 2×128 only because 3e-4 suits the smaller model.
- Mandlekar et al. (robomimic) found that offline robot-imitation results are sensitive to design choices and stopping criteria, because training and evaluation objectives differ ([CoRL 2021](https://arxiv.org/abs/2108.03298)). This supports the plan's closed-loop selection.

**Implication.** Keep closed-loop selection. Add a cheap per-architecture LR sweep on the offline proxy, i.e. validation hold-torque bias and physical-unit RMSE, which takes minutes per run. Only the best LR per architecture goes to rollouts. This adds offline runs only. Rollouts, the expensive part, stay at one candidate per architecture.

### 3.6 Data quality: Clean_253 versus Clean_223

Belkhale, Cui and Sadigh frame imitation data quality as distribution shift. In their analysis, low **action divergence** (consistent, learnable actions at each state) matters, and more **state diversity** is not automatically better ([NeurIPS 2023](https://arxiv.org/abs/2306.02437)).

**Implication.** For A, the chatter episodes carry no action inconsistency: the labels are an exact deterministic function (§2.3). Dropping them buys no label consistency and removes the entire low-target interval. This supports the plan's decision to use Clean_253 as primary. For B, partial observability already creates action divergence everywhere, so the 223/253 comparison is more open there.

### 3.7 Optional: a physics-informed residual (a decision for the group)

Residual policy learning trains a network on top of an existing controller ([Silver et al., 2018](https://arxiv.org/abs/1812.06298)). The analogue here would compute g(q) with the deployable Pinocchio model, which FK already uses, and train the MLP on `tau - g(q)`. This removes the hardest part of the precision requirement in §2.2. However, the brief asks for a regressor that replicates the controller, and part of the controller would then be hand-coded. If used, it should be a clearly labelled ablation against the pure MLP, not the baseline.

## 4. Recommended recipe

### 4.1 Baseline settings (changes from the plan in bold)

| Setting | Value |
| --- | --- |
| Data | Clean_253, 221 fitting / 32 validation scenarios as in the plan; 64 test untouched |
| Inputs | 21 (A) / 26 (B), standardised with fitting-only statistics |
| Trunk | `[Linear → LayerNorm → SiLU] × depth`, widths per M1–M6 |
| Heads | Linear 7 (torque) + **linear 2 (right, left) mirrored to 4 wheels** |
| Output scaling | Per-channel standardisation with fitting-only statistics; execution-only clipping to the source manifest limits: currently ±30 N·m and **±6 rad/s** for both A and B; save these limits in the checkpoint |
| Loss | 0.5·arm + 0.5·wheel, **Huber δ = 1 or MSE, chosen once in the M2 pilot** |
| Jacobian loss | **Off for the pilot and baseline architecture screen; on only in the separate BC+J experiment (§4.4)** |
| Optimiser | AdamW, weight decay 1e-4, batch 1024 |
| LR | **Per architecture, best of {1e-3, 3e-4, 1e-4} on offline proxy**; short linear warm-up (e.g. 500 updates) then cosine |
| Update budget | **Set from the pilot plateau** (§4.2); same for all architectures |
| Sampling | Episode-uniform then timestep-uniform, full rate, on-device indices |
| Threads | 1 per rollout worker; parallel workers for the 8- and 32-scenario rollouts |

### 4.2 Setting the budget with one pilot curve

Train M2 on A with a **constant** LR after warm-up, at 1e-3 and at 3e-4, for 50k updates. Log validation metrics every 1k updates. With a constant LR, the curve shows directly where validation hold-torque bias and RMSE plateau. Choose the common budget at about 1.5× the plateau point, then switch to cosine decay over that budget for all screening runs. With a cosine schedule, every candidate budget would need its own run. Record the pilot as a separate experiment, as the plan requires.

### 4.3 Huber or MSE

After choosing the budget from the constant-LR Huber pilots, run a matched pair of M2-A fits with Huber and MSE: same seed, selected pilot LR, frozen update budget, warm-up/cosine schedule, fitting data and initial weights. Compare hold-phase bias, errors on saturated and high-|qd| steps, and the 8-scenario rollouts. Do not compare the shorter cosine-schedule MSE run directly against a 50k constant-LR Huber run. A fit can be reused as the first baseline only when its complete resolved configuration matches.

Huber with δ = 1 standardised unit behaves like L1 for residuals beyond about 2–6 N·m. It reduces the influence of large prediction residuals relative to MSE; saturation or chatter alone does not determine whether a sample lies in that regime. Inspect residuals on saturated and high-|qd| samples, which may be harder to fit. The effect on damping and stability must be measured. Freeze the loss choice before the architecture screen.

### 4.4 Damping-Jacobian loss experiment (A and B)

After selecting the plain-BC architecture per teacher, run BC+J on the same clean fitting episodes. Reuse its features, normalisers, action loss, optimiser, LR schedule, update budget and seeds 0–2. Start each paired run from the same seeded initial weights as its plain-BC counterpart and train for the same number of updates; do not compare a baseline with a BC+J model that received extra fine-tuning. Save separate checkpoints with `objective=bc` or `objective=bcj` and globally unique run IDs following the structure design, including cohort, LR, fit stage and config hash.

Use the following implementation rules:

1. Differentiate the seven standardised torque predictions with respect to the seven standardised measured joint velocities, holding all other inputs fixed. Include both diagonal and off-diagonal entries in the loss. Apply the loss before the execution-time action clip; wheel outputs receive action loss only.
2. Set the physical-unit expert target to `J_ij = -30 * mask_i * delta_ij`, where `mask_i` is one for an unclipped expert joint and zero for a verified strictly saturated one. Prefer teacher pre-clip torque or explicit internal saturation metadata and exclude boundary rows with a documented positive tolerance `epsilon_tau` in N·m. The existing recorded `clip_flags` means execution clipping (`a_executed != a_expert`), not internal teacher saturation, and must not be used for this mask. If only clipped labels are available, supervise rows with `abs(tau_expert) < 30 - epsilon_tau` and omit near-limit rows, since clipped labels alone cannot distinguish strict saturation from the boundary. Convert each entry with `J_standardised_ij = J_physical_ij * sigma_qd_j / sigma_tau_i`, using the actual saved scales, including scale floors.
3. Use a sub-batch of up to 256 samples from each action batch. Define `L_J` as the mean squared entry error over valid Jacobian entries; skip the term if there are no valid entries. Backpropagate through the student Jacobian to its weights. Derivative targets come from the controller structure, with valid rows determined from recorded clean expert labels or pre-clip metadata; this experiment adds no synthetic states or action labels.
4. Calibrate one λ_J per teacher/selected-architecture condition on a fixed fitting-only batch at the reference seed-0 initialisation so that `lambda_J * L_J` is approximately 10 % of `L_action`. Record the batch, initialisation and value, then freeze λ_J across seeds 0–2 and any final refit of that condition. The no-J control uses λ_J = 0. Log both raw and weighted losses, valid-entry coverage and training time.
5. Compare physical-unit RMSE/MAE, hold torque bias, damping diagonals and cross-joint terms, saturation, oscillation and sustained-hold success on the same 8- and 32-scenario validation sets. Keep the 64 test scenarios untouched until the final freeze. Retain plain BC as the reference regardless of whether BC+J improves it.

BC+J remains an MLP at deployment: the derivative loss is training-only and requires no online expert or Jacobian evaluation. For B it adds local velocity-feedback supervision but cannot recover hidden teacher memory. Success and stability still require closed-loop validation.

## 5. Offline gates before spending rollouts

Run these checks on the 32 validation scenarios after each training run. They take seconds and catch most problems before any rollout:

1. **Signed hold-phase torque bias** per joint: compute prediction-minus-expert mean over each episode's final second; retain signed values and report the median and maximum of their absolute values over episodes. Gate the maximum absolute episode bias at roughly 0.5 N·m (§2.2), not the signed maximum.
2. **Learned damping**: on valid unclipped hold rows, report the median diagonal ∂τ̂_j/∂qd_j against −30 and the fraction below −33 (more aggressive than the teacher). Report verified saturated rows against zero, with omitted-row and valid-entry coverage as in §4.4.
3. **Phase-split errors**: navigation, approach and hold, in physical units, as the plan already lists.
4. **Same-side front/rear wheel mismatch**: FR−BR and FL−BL are zero by construction with the mirrored head. Left and right wheel speeds may differ during turns.

Validate the harness in dependency order:

- (a) Before fitting, run the expert from raw observations through the same action encoding, normalisation round trip, action ordering, limits and rollout evaluator used by the student. Build student features in parallel to verify dimensions and finiteness; do not attempt to reconstruct the expert's full input from the compressed student features. Use the unified evaluation predicate for the expert reference.
- (b) Once the baseline trainer exists, overfit one fitting episode, check its offline action replay, then run a diagnostic closed-loop replay. Keep this sanity run separate from architecture selection.
- (c) After the first fitted student supplies validation residuals, add documented residual-matched noise to an expert diagnostic rollout. Record the resulting degradation; matching an error marginal does not guarantee the same state-dependent student failure, so a different outcome is diagnostic evidence rather than an automatic harness failure.

## 6. Revised order of work

| Stage | Work | Exit evidence |
| --- | --- | --- |
| 0 | Install PyTorch ≥ 2.7 (cu128) in the sim venv; verify `sm_120` or fall back to CPU; add thread pinning to workers | Import and device check recorded |
| 1 | Feature cache, split manifest, normaliser and checkpoint round trips, NumPy inference export; harness check (a), then (b) when the trainer is ready | Data/interface checks pass; separate overfit/replay evidence |
| 2 | M2-A pilot: constant-LR curves at 1e-3 and 3e-4; budget chosen; Huber vs MSE | Saved pilot curves, frozen budget and loss |
| 3 | **First verified baseline**: M2-A, then M2-B, through the offline gates, with 8- and 32-scenario rollouts; residual-noise diagnostic (c) now has fitted-student residuals | Checkpoints with closed-loop results and separately recorded diagnostic |
| 4 | Screen M1–M6 × {A, B} with a per-architecture LR sweep (offline) and one rollout candidate each | 12 closed-loop results plus 36 offline runs |
| 5 | Seeds 1–2 for the top two per teacher | 3 seeds per finalist |
| 6 | Damping-Jacobian loss only: BC+J versus the selected plain-BC architecture for each teacher, following §4.4 | Separate checkpoints and paired validation results for seeds 0–2 |
| 7 | Optional 223 comparison and any planned DART/DAgger comparisons (§3.4); freeze; optionally refit the clean finalist on all training episodes at the frozen update budget; evaluate the test set once | Frozen selection `best.pt` or new refit `final.pt`, with distinct run IDs and recorded provenance |

Stages 0–3 give the deliverable the plan prioritises. Stages 0–5 use ordinary action-loss BC throughout. Stage 6 adds the single training-objective change of damping-Jacobian supervision, with no new collection or synthetic-state augmentation. Later DART/DAgger runs retain the no-J controls in §3.4, so their gains can be distinguished from those of derivative supervision. Complete any such model selection before the final freeze and test evaluation in stage 7. Report BC+J as an application of derivative-supervised learning to this controller; empirical improvement remains to be established.

Selection runs use independent validation and save `best.pt`. An optional final all-training-data refit uses a new `fit_stage=refit` run, starts from its declared seeded initialisation, refits training-only normalisers, recomputes standardised derivative targets, and trains to the previously frozen update count. It saves `final.pt` and performs no validation-based selection or λ_J recalibration. If refitting is skipped, use the frozen selection `best.pt`. Follow the structure design's run identity and artifact rules in both cases.

## 7. Limits of this review

The 0.5 N·m gate comes from a teacher stress test with constant offsets. A student's error is state-dependent, so treat the gate as a screening heuristic and confirm it in closed loop. The kd·dt/I stability argument is a single-joint simplification of the chatter, and the inertia threshold has not been measured. The theoretical results in §3.1 and §3.3 assume conditions (smoothness, incremental stability) that have not been verified for this robot. They motivate experiments, not predicted outcomes. Timings are single-sample measurements on this host and should be re-measured inside the actual rollout loop.

## References

- Belkhale, Cui, Sadigh. *Data Quality in Imitation Learning*. NeurIPS 2023. [arXiv:2306.02437](https://arxiv.org/abs/2306.02437)
- Block, Jadbabaie, Pfrommer, Simchowitz, Tedrake. *Provable Guarantees for Generative Behavior Cloning: Bridging Low-Level Stability and High-Level Behavior*. NeurIPS 2023. [arXiv:2307.14619](https://arxiv.org/abs/2307.14619)
- Czarnecki, Osindero, Jaderberg, Świrszcz, Pascanu. *Sobolev Training for Neural Networks*. NeurIPS 2017. [arXiv:1706.04859](https://arxiv.org/abs/1706.04859)
- Godbole, Dahl, Gilmer, Shallue, Nado. *Deep Learning Tuning Playbook*. Google, 2023. [Scientific approach](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific)
- Hertneck, Köhler, Trimpe, Allgöwer. *Learning an Approximate Model Predictive Controller with Guarantees*. IEEE Control Systems Letters 2(3), 2018. [arXiv:1806.04167](https://arxiv.org/abs/1806.04167)
- Karg, Lucia. *Efficient Representation and Approximation of Model Predictive Control Laws via Deep Learning*. IEEE Trans. Cybernetics, 2020. [arXiv:1806.10644](https://arxiv.org/abs/1806.10644)
- Laskey, Lee, Fox, Dragan, Goldberg. *DART: Noise Injection for Robust Imitation Learning*. CoRL 2017. [PMLR](https://proceedings.mlr.press/v78/laskey17a.html)
- Mandlekar et al. *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*. CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
- Pfrommer, Zhang, Tu, Matni. *TaSIL: Taylor Series Imitation Learning*. NeurIPS 2022. [arXiv:2205.14812](https://arxiv.org/abs/2205.14812)
- Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011. [PMLR](https://proceedings.mlr.press/v15/ross11a.html)
- Silver, Allen, Tenenbaum, Kaelbling. *Residual Policy Learning*. 2018. [arXiv:1812.06298](https://arxiv.org/abs/1812.06298)
- Simchowitz, Pfrommer, Jadbabaie. *The Pitfalls of Imitation Learning when Actions are Continuous*. COLT 2025. [PMLR](https://proceedings.mlr.press/v291/simchowitz25a.html)
- PyTorch forum. *When will sm120 support be available?* [Thread](https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621)
