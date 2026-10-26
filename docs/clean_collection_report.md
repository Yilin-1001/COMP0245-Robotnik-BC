# Clean Expert Data Collection Report

Collected: 26 October 2026. Scope: **clean collection only**: no action noise, sensor noise, physics randomization, external pushes, DART or DAgger.

Two datasets were collected on one frozen set of 320 scenarios (256 train, 64 test). Each expert ran every scenario once, giving **640 episode records**. Every attempt is stored, including failures. Field definitions follow the [dataset specification](expert_datasets.md). The disturbance limits of both experts are documented separately in [expert operating limits](expert_operating_limits.md).

## 1. What was collected

| | Markovian (dataset A) | Stateful (dataset B) |
| --- | --- | --- |
| Location (git-ignored) | `final/robotnik_arm_task/data/Markovian/Clean/` | `final/robotnik_arm_task/data/Stateful/Clean/` |
| Episodes: train / test | 256 / 64 | 256 / 64 |
| Sustained-hold pass, train | **256/256** (Wilson 95%: 98.5–100%) | **253/256** (96.6–99.6%) |
| Sustained-hold pass, test | **64/64** (94.3–100%) | **64/64** (94.3–100%) |
| Eligible for initial BC | 256 | 253 (3 excluded: navigation timeout) |
| Recorded control steps, train / test | 3,078,616 / 768,695 | 3,476,970 / 841,494 |
| Size on disk | 1.9 GB | 2.3 GB |
| Default student input | 21 dims | 26 dims |

Each dataset folder contains:
- `manifest.json`: schema, teacher parameters, feature and FK definitions, field roles, source hashes and the command.
- `scenarios_train.csv`, `scenarios_test.csv`: byte-identical copies of the frozen manifests.
- `episodes.csv`: one outcome row per episode.
- `training_manifest.csv`: the initial-BC selection.
- `train/` and `test/`: one compressed `.npz` per episode.
- `qa/`: checks, coverage, the attempt log and the template-equivalence result.

The frozen scenario manifests and the pre-rollout coverage audit are in `data/scenarios/`.

## 2. Expert definitions (frozen)

**Markovian:** [`data_collection/expert.py`](../final/robotnik_arm_task/data_collection/expert.py), instantiated with `lead_time = 0.50 s`.
- The control law and model calibration are unchanged: the 1 kg end-effector body and base pitch 0.0022 rad.
- The file's own default stays at 0.15 s; the collection value is set in `settings.yaml` (`clean_collection.markovian_lead_time`).
- Version label: `markovian-lt0.50-v1`.

**Stateful:** [`data_collection/template_teacher.py`](../final/robotnik_arm_task/data_collection/template_teacher.py) calls the original `base_controller` and `arm_controller` from [`robotnik_arm_control.py`](../final/robotnik_arm_task/robotnik_arm_control.py). It keeps:
- the world-origin goal rule;
- the 25 s navigation timeout and the latched `done`;
- the base-before-arm call order;
- `q_des = q_hold = measured initial arm angles`, phase `align` and `done = False` at every reset;
- one memory update per control step.

Version label: `template-origin-goal-v1`.

**The Stateful wrapper reproduces the template bit for bit.** [`verify_template_teacher.py`](../final/robotnik_arm_task/data_collection/verify_template_teacher.py) runs the unmodified `robotnik_arm_task.main()` headless with a recorder on its motor commands. It then compares the wrapper's 11 commands at every step on three targets: the template's own, one behind the robot and one high. All 8,001, 11,506 and 10,400 steps were identical (`qa/template_teacher_equivalence.json`).

An earlier wrapper version rebuilt the base quaternion from roll/pitch/yaw. That changed it by about 1e-16, and the closed loop amplified the difference to tens of N·m within seconds. The wrapper now uses the simulator quaternion whenever the observed yaw equals the true yaw. The noise-sensitivity study in `expert_operating_limits.md` used the earlier version, so its trajectories are not bit-identical to this one.

Neither expert was retuned. Source revision: git `e916e83` plus uncommitted changes to the collection code, as listed in `manifest.json` (`source.git_dirty_paths`). Every behaviour-relevant file is pinned by SHA-256 (`source.source_sha256`), giving code digest `c94ad845…0038`. That digest also covers the simulator wrapper files and URDFs. Environment: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, PyBullet 3.2.7, Pinocchio 4.1.0.

## 3. Scenarios

Each scenario is a scrambled Sobol point in six dimensions:

| Dimension | Range |
| --- | --- |
| Target horizontal distance from the initial base | 0.5–3.0 m |
| Target azimuth relative to initial heading | −π to π |
| Target world height | 0.1–1.25 m |
| Initial base x | −0.5 to 0.5 m |
| Initial base y | −0.5 to 0.5 m |
| Initial base yaw | −π to π |

- **Mapping:** `target_xy = start_xy + d·[cos(yaw + az), sin(yaw + az)]`.
- **Distance:** uniform in radius, so near targets are denser per unit floor area than far ones.
- **Engines:**
  - train: `qmc.Sobol(d=6, scramble=True, seed=1026001).random_base2(m=8)`;
  - test: a separate engine with seed 1026002 and `m=6`.

  The test set is therefore not a prefix of the training sequence.
- **Fixed conditions:**
  - arm and finger start poses, as in the template;
  - nominal masses and friction;
  - zero added sensor or action noise;
  - no pushes or delays.
- **Identity:** scenario IDs (`train-0000` … `test-0063`) and environment seeds (1,000,000 + i for train, 1,100,000 + i for test) are immutable.
- **Checksums (SHA-256):**
  - `scenarios_train.csv` = `4b1114ff…7770`
  - `scenarios_test.csv` = `4e27ba1d…7d2f`

  Regenerate them with `python -m data_collection.scenarios`, which refuses to overwrite a differing frozen file.

Checks before rollout:
- **Duplicates and overlap:** no duplicate tuples, and no shared IDs or seeds between train and test. The closest test point is 0.16 from any training point in the normalised cube.
- **Development scenarios:** the 107 development scenarios that can be regenerated from the repository scripts (noise sweep, T3 development, controller comparison) have no identical tuple in the manifests; the closest lies 0.18 away. The 25 October lead-time experiment's scenarios cannot be regenerated from the repository, so they were not checked.
- **Exact equality:** both datasets' scenario files are byte-identical to the frozen manifests.

## 4. Coverage evidence and gaps

The pre-rollout audit is in `data/scenarios/coverage/coverage_report.md`, with `marginals.png`, `pairs.png` and `world_targets.png`. A copy is in each `qa/scenario_coverage/`.

- **Marginals:** train uses 8 bins and test 4 bins per variable; each has 32 and 16 points per bin respectively.
- **Coarse joint cells (distance × azimuth sector × height):**
  - train: 36 cells, 4–9 scenarios each, none empty;
  - test: 16 cells, exactly 4 each.
- **Discrepancy:** the centred L2 discrepancy is about 15× lower than for i.i.d. uniform sets of the same size (train 0.00055 vs median 0.0080; test 0.0053 vs 0.034).
- **Front, side and rear:** 64/64/64/64 train and 16 each in test. Start quadrants are balanced the same way. 64 train and 16 test targets lie in the rear sector (|relative azimuth| > 135°).
- **Near/far × low/high × four directions (32 combinations):** train has at least one scenario in each. **Test is empty in 4**: near-low-front, near-high-left, far-low-rear and far-high-left. These cannot all be filled with 64 points.
- **Boundaries:** Sobol does not sample exact endpoints. The largest gap between the nearest sample and a range endpoint is 0.4% of the range in train and 1.5% in test. Exact corners of the 6-D box are not covered.

Marginal and pairwise plots do not prove full six-dimensional coverage; no such claim is made. A later supplementary boundary suite should include:
- exact range endpoints;
- targets near the world origin, where B's origin-based goal is most unusual;
- the near/far × low/high corners the test set misses.

That suite should be a separate, labelled set, not a replacement for any Sobol point.

### Trajectory coverage after collection

Counts are for the train split. The `qa/coverage_*.png` figures and `qa_report.json` → `coverage` give the full numbers.

| Segment | Markovian steps | Stateful steps |
| --- | ---: | ---: |
| Navigating (reaching not active) | 1.53 M | 1.94 M |
| Approach (reaching active, before first task reach) | 0.78 M | 0.77 M |
| Holding (after first task reach) | 0.77 M | 0.76 M |

Navigation behaviour differs by teacher:
- **Stateful phases before `done`:** align 0.43 M steps, drive 1.35 M and orient 0.16 M.
  - Phase sequences: 249 align→drive→orient, 3 drive→orient (already aimed), 1 orient-only (started 5 cm from the goal) and 3 align→drive that never finished (the failures).
  - 317 of 320 completions came from orient.
- **Markovian base modes:** turn in place 0.30 M steps, drive 1.31 M, orient 0.02 M and stopped 1.45 M.
  - Reaching was never switched off after it had started.

Gaps that matter for learning:
- **No recovery data.** Clean collection contains no off-path recovery states. The Markovian expert never re-navigates after reaching starts, and the Stateful base never moves after `done` except in the 3 failures.
- **Narrow arm configurations.** The arm stays in configurations used for reaching from the fixed home pose. Joint 7 spans only −0.06 to 0.09 rad.
- **Approach always starts far out.** Reaching activates with the hand at least 58 mm (A) or 9 mm (B) from the target, with a median of about 290 mm.

## 5. Episode protocol and outcome definitions

Control and logging run at 1 kHz with no decimation and no action hold. For T executed actions there are T+1 observations. Each step follows the spec's order:
1. Save the pre-action observation and the simulator-truth diagnostics of the same instant.
2. Stateful only: copy the teacher memory before the call.
3. Query the teacher once.
4. Save `a_expert = [torque_arm(7), wheel_cmds FR, FL, BL, BR (4)]` and the memory after the call.
5. Clip to the execution limits (±30 N·m, ±6 rad/s) to form `a_executed`.
6. Step 1 ms and save the next observation.

No command was ever clipped, so `a_executed == a_expert` everywhere. Noise is recorded as disabled, and `sampled_action_noise` is all zeros.

**Termination.** An episode ends 3.0 s after the first task reach, or at the 40 s cap. The first task reach is the first pre-action observation where all three hold:
- the true end-effector error is below 10 mm;
- the teacher's reaching is active on that step;
- t > 8 s.

This matches the original `check_stop` reach condition. Because the clock is an accumulated float, as in the template, t = 8.000000000001 at step 8000 already passes the gate.

**Stored outcomes**, kept separate:
- ever reached, and the time of the first task reach;
- whether the full post-reach window was recorded;
- final end-effector error;
- maximum and mean error over the final 1 s (observations T−1000…T);
- fraction of final-1 s samples below 10 mm;
- **sustained-hold pass**: a complete window with every final-1 s sample below 10 mm;
- navigation `done` time and cause (orient completed or timeout);
- numerical failure, termination and truncation reasons.

`done` is never treated as task success, and a timeout `done` is never counted as successful navigation.

**The original stall stop is recorded but not used.** `check_stop` tests for a stall before it tests for reaching. For any expert that reached the target before 8 s and stayed there, its "best error" stops improving, so at t = 8 s it reports "Target not reachable" while the hand is within tolerance. Examples: 0.64 mm in `markovian-clean-train-0000`. This happened in 70 of 320 Markovian episodes, all of which passed the sustained hold. Terminating on it would have discarded valid holds.

**The 8 s gate dominates time-to-reach for A.** The Markovian expert was first within 10 mm with reaching active before 8 s in 145 of 256 training episodes (median 7.4 s). The Stateful expert did so in 62 of 256 (median 9.6 s). `qa/episode_checks.csv` stores the ungated time as `t_first_within_tol_ungated`.

## 6. Failures and exclusions

Only three episodes failed, all Stateful and all in train: `train-0064`, `train-0131` and `train-0183`. Each ended in a navigation timeout without reaching, with final errors of 1.1–2.5 m. Traces show one mechanism that the controller produced with no noise:
1. Each starts 0.46–0.54 m from its goal but facing away from it, with an initial turn of 2.55–3.02 rad.
2. After aligning, the base drives to within 0.11–0.17 m of the goal, not inside the 0.08 m tolerance, with the goal off to one side.
3. The drive phase limits yaw rate to 0.3 rad/s and keeps a positive forward speed. The `passed_goal` exit needs a heading error above 90°, and there is no transition back to align.
4. So the base circles the goal, the 0.01 lateral wheel friction lets the spiral widen, and the 25 s timeout latches `done` with the base 1.4 m away.

This geometry is a known risk: 13 train scenarios have a start-to-goal distance under 0.6 m and an initial turn over 2.5 rad, and 3 of them failed. The 2 such test scenarios both succeeded. The failures were not resampled, replaced or relabelled. Their full trajectories remain in `train/`.

**Initial-BC selection rule:**
- train split;
- valid data (every QA check passed);
- post-reach window complete;
- sustained-hold pass.

Result: 256/256 Markovian and 253/256 Stateful, with the 3 timeouts excluded as `outcome not_reached_nav_timeout`. No episode prefixes were salvaged.

**Test set.** All 64 fixed test scenarios are kept for closed-loop evaluation, including any future expert failures. The column `offline_test_subset` marks test episodes that pass the same quality rule, 64/64 for each expert. Use it only for offline action error, reported separately from the full set. Do not use test data for tuning, model selection or normalisation statistics.

**Label-quality flag: chatter during holding.** The hold window of 35 Markovian episodes (28 train, 7 test) and 33 Stateful episodes (27 train, 6 test) contains a sustained period-2 oscillation:
- **Frequency:** the dominant frequency is 500 Hz, the 1 kHz Nyquist limit, so the command reverses every 1 ms step.
- **Joint:** joint 3 is dominant in every case.
- **Torque:** some arm torque sits at ±30 N·m on about half the steps.
- **Arm and hand motion:** joint speeds reach about 1–2 rad/s, while the hand stays within 10 mm (peak deviation up to 7.8 mm). Settled holds sit at about 0.005–0.008 rad/s.
- **Flag rule:** `training_manifest.csv` → `label_quality_flags = hold_chatter` when the median joint-speed norm after the first reach exceeds 0.5 rad/s.

These episodes pass the agreed rule and stay eligible. The flag lets a later selection or loss weighting exclude them explicitly. The likely cause, a joint PD that is too stiff for a light joint at a 1 ms step in some poses, is a hypothesis this collection does not test.

## 7. QA results

[`qa_clean.py`](../final/robotnik_arm_task/data_collection/qa_clean.py) ran on all 640 episodes. Every check passed for every episode. Results are in `qa/qa_report.json` (per dataset) and `qa/episode_checks.csv` (per episode).

| Check | Result |
| --- | --- |
| Inventory | 320/320 per expert. Exactly one recording per expert and scenario, no superseded attempts, one code digest, scenario files identical to the frozen manifests in both datasets, no train/test overlap |
| Structure | T+1 observations for T actions. Shapes and dtypes correct, all values finite. Timestamps within 1.6e-11 s of k·1 ms, step dt within 2.4e-15 s |
| Noise and execution | Noise all zeros and disabled, `a_executed == a_expert` on every step, 0 clipped steps |
| Outcome flags | All recomputed independently from the stored arrays and identical |
| Markovian re-query | Every stored action, reach flag and teacher EE estimate recomputed from its stored pre-action observation with the frozen expert. Maximum difference **0** |
| Stateful replay | Teacher rebuilt from the recorded `q_hold`, goal and initial phase, then run on the stored observations plus stored teacher-side measurements (true base quaternion, finger and wheel angles). True `q_des` was never an input. Actions, `q_des` before/after, `qd_des`, phase and `done` before/after, `reach_active` and EE estimate all identical (difference **0**, 0 mismatches) |
| Stateful memory | Reset (`q_des_before[0] = q_hold = q_init` = configured pose, `align`, not done), continuity (before[t+1] = after[t] for `q_des`, phase and `done`), latched `done`, forward-only phase order, `reach_active = done` and the hold reference before `done`: all pass for 320/320 |
| Alignment | Simulator-truth EE recomputed from each stored pre-action observation equals the stored value at the same index (difference 0). The same comparison shifted by one step differs by a median 0.17 mm, so the check would catch a misalignment |
| Features | 21 (A) and 26 (B) default inputs built from stored raw data with the deployable FK estimator `fk-v1` (observed yaw, fixed pitch 0.0022 rad, fixed fingers); all finite |
| Re-execution | 6 episodes per expert (3 train, 3 test) re-simulated in fresh processes. Every stored array bitwise identical |

Every replay and re-query used zero tolerance, which is possible because all values are stored in float64. The re-execution check confirms simulator determinism; it is separate from replaying the controller on stored measurements.

The deployable FK estimate differs from simulator truth by a median of 0.4 mm (A episodes) and 0.75 mm (B episodes), but by up to 7.7 mm at worst. The fixed-pitch assumption breaks down while the base is accelerating. That is close to the 10 mm tolerance and should be kept in mind when interpreting the student's EE-error input.

## 8. Reproduction

From `final/robotnik_arm_task/` in the project environment:

```bash
python -m data_collection.scenarios                 # frozen manifests (verifies, never overwrites a differing file)
python -m data_collection.scenario_coverage         # pre-rollout coverage audit
python -m data_collection.verify_template_teacher --out data/Stateful/Clean/qa
python -m data_collection.collect --teachers markovian stateful --split train --stop 16   # pilot (train-0000..0015)
python -m data_collection.collect --teachers markovian stateful --split train test       # remaining 608 episodes
python -m data_collection.qa_clean --expect full --reexec 6
```

This collection used exactly those two `collect` commands, recorded in `qa/attempts.jsonl`. The pilot recordings passed QA before scaling and remain in the final cohort. No attempt was superseded.

The collector skips existing files whose code digest matches the current code. It refuses to mix in recordings made with different code unless run with `--supersede`, which moves old files to `qa/superseded/` and logs them. Simulation is deterministic, so a single `--split train test` run on a clean data folder gives the same files. Collection took 17 s for the pilot and 237 s for the rest, on 22 worker processes.

## 9. Remaining limitations

- **Clean only.** There are no disturbance or recovery demonstrations, so the datasets alone cannot support claims about robustness or recovery.
- **Expert behaviour, not ground truth.** Stateful labels include the template's navigation weaknesses (3 timeouts in train). Both experts' hold labels include the joint-3 chatter in about 10% of episodes.
- **Reference ceilings from a finite sample.** 64/64 test passes do not guarantee success on the whole task distribution; the Wilson lower bound is 94.3%.
- **Information mismatch.** B's teacher uses the true base roll/pitch and the measured finger angles, which the 26-input student does not receive. Exact action matching is not expected.
- **Simulator-truth base velocities.** Base speed and yaw rate are recorded only as simulator truth and are labelled as such. They are not valid deployment inputs without an equivalent sensor model.
- **Uncommitted code.** The collection code was uncommitted when it ran; it is identified by its file hashes. Commit it, then compare the committed files' SHA-256 with `manifest.json` → `source.source_sha256`.
