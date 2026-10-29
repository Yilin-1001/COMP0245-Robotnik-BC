# Teacher A Operating Envelope: Motion and Task Limits

Date: 29 October 2026. Status: measured on development scenes; premise for the final report.

This document records what the teacher used for training, the Markovian expert A, can and cannot do in simulation, and how those limits relate to the task range used for data collection and evaluation. It sets the premise for the report: a student is only expected to succeed where the teacher succeeds, and the training range was chosen inside the teacher's feasible region.

All results are empirical, from finite development scene sets. They are not certified bounds. Each table states its sample size.

## 1. Scope: which teacher

| Item | Value |
|---|---|
| Teacher | `data_collection/expert.py`, `MarkovianExpert` |
| Version | **`lead_time = 0.50 s`** (teacher version `markovian-lt0.50-v1`), the setting used for every dataset (Clean, DART). The code default is 0.15 s; it behaves differently (Section 6) and must not be quoted for teacher A. |
| Physics | Nominal unless stated: floor friction 1.0, nominal link masses, no sensor noise, 1 kHz control and physics |
| Execution limits | Arm torque ±30 N·m per joint; wheel speed ±6 rad/s (teacher commands at most about 5.3 rad/s) |
| Comparison teacher | Stateful template controller B (`data_collection/template_teacher.py`, original `robotnik_arm_control.py` logic), shown where it clarifies whether a limit is physical or specific to A |

Teacher A computes its action from the current observation only. Its arm reference is `q_des = q + lead_time * qd_des`, rebuilt every step, with no integrator. Its base controller replaces the template's phases with functions of the current distance and heading error, and it has no navigation timeout.

## 2. Task definition and success criterion

The task and evaluator predicates used throughout (`experiments/configs/common.yaml`, `evaluation/evaluate.py`):

| Item | Value |
|---|---|
| Reach | True end-effector error below **10 mm** at some time **after 8 s** |
| Sustained hold (success) | After the first reach, the episode continues **3 s**; every sample of the **final 1 s** is below 10 mm |
| Episode cap | **40 s** |

The scenario ranges from which training, validation and test scenes are sampled (`settings.yaml`, `clean_collection.ranges`):

| Dimension | Range |
|---|---|
| Target horizontal distance from the start base | 0.5–3.0 m |
| Target azimuth relative to the start heading | −π to π |
| Target world height | 0.1–1.25 m |
| Start base x, y | each −0.5 to 0.5 m |
| Start base heading | −π to π |

Training (221 fitting scenes) and validation (32 scenes) come from one scrambled Sobol sequence; the 64 sealed test scenes come from a separate Sobol engine. The split is in `experiments/splits/clean253.json` and is shown in `results/figures/2026-10-28/scenario-split/2026-10-28__figure__scenario-split.png` (`python -m evaluation.plot_scenario_split`).

## 3. Robot and controller envelope

Kinematic values are computed with the teacher's Pinocchio model at the initial pose (`evaluation/expert_workspace.py`, `envelope()`).

| Quantity | Value | Source |
|---|---|---|
| Shoulder (link 2) height above ground | 0.495 m | Pinocchio FK |
| Shoulder ahead of base centre | 0.205 m | Pinocchio FK |
| Reach clamp (target distance from the shoulder) | 0.85 m | `ARM_REACH_LIMIT` |
| Highest point straight up (shoulder + clamp) | **≈ 1.345 m** | derived |
| Joint PD gains | kp 100 N·m/rad, kd 30 N·m·s/rad | `ArmControllerParams` |
| Max desired joint speed, Cartesian gain, max Cartesian speed | 1.0 rad/s; 1.5 s⁻¹; 0.25 m/s | `ArmControllerParams` |
| Base speed limits | linear 0.05–0.25 m/s, angular ≤ 1.0 rad/s | `BaseControllerParams` |
| Base stop tolerances | 0.08 m, 0.05 rad | `BaseControllerParams` |
| Reach gate (arm may reach) | distance error < 0.12 m and heading error < 0.075 rad (1.5 × stop tolerances) | `MarkovianExpertParams.reach_tol_scale` |
| Base standoff from the target | 0.80 m at z ≤ 0.1, 0.65 m at 0.5–1.1, 0.35 m at z ≥ 1.25 (linear in between) | `STANDOFF_POINTS` |
| Wheel lateral friction | 0.01 (anisotropic contact `[1.0, 0.01, 1.0]`) | `core/env.py` |

## 4. Geometric limits

Sweep: `python -m evaluation.expert_workspace [--teacher B]`. 8 random scenes per grid value (random start in ±0.5 m, heading and azimuth; development seeds 970000+). Results: `results/diagnostics/2026-10-29__expert-workspace{,-B}/`.

### 4.1 Target height (start-to-target distance 1.5 m)

| Target height (m) | 0.0 | 0.05 | **0.1–1.30** | 1.35 | 1.4 | 1.5 | 1.6 |
|---|---|---|---|---|---|---|---|
| A sustained hold | 0/8 | 0/8 | **8/8 at every value** (final error 0.4–0.9 mm) | 0/8 | 0/8 | 0/8 | 0/8 |
| A closest approach (median) | 19 mm | 24 mm | < 1 mm | 19 mm | 68 mm | 168 mm | 266 mm |
| B sustained hold | 2/8 | 4/8 | 8/8 (final error 0.1–0.8 mm) | 0/8 | 0/8 | 0/8 | 0/8 |

- **Upper limit ≈ 1.30 m is physical.** Both teachers fall short by the same ≈ 20 mm at 1.35 m, consistent with the 1.345 m kinematic bound. An earlier check with targets at 1.7–2.0 m gave 0/10 (Section 6).
- **Lower limit 0.1 m is partly specific to A.** A stops about 2 cm short below 0.1 m; B, whose integrated reference keeps pushing, still holds half the 0.05 m scenes. The physical floor is therefore lower than A's.

### 4.2 Start-to-target distance

| Distance (m) | 0.05 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| A sustained hold | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| A first reach (median) | 8.0 s | 8.0 s | 8.0 s | 8.0 s | 8.0 s | 8.0 s | 23.5 s | 32.7 s | 39.6 s |
| B sustained hold | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | **0/8** | **0/8** |

The far sweep raised the episode cap to 80 s, to separate capability from the task's 40 s cap.

- **Near targets are not a limit for A**: it reverses to the standoff distance and reaches even at 0.05 m.
- **Far targets are limited only by time.** A covers about 0.22 m/s on average, so each extra metre costs about 4.6 s. With the 40 s cap and the 3 s window, the task limit is **about 9 m**.
- B fails at 8 m and 10 m (stopping 1.2 m and 2.9 m short) because its 25 s navigation timeout latches "done" and stops the base. A has no timeout.

### 4.3 Start pose

A depends only on relative geometry. Shifting the start 1–3 m from the origin with in-range relative geometry leaves it unaffected (12/12, Section 7).

## 5. Disturbance limits

From `docs/reports/2026-10-26__report__expert-operating-limits.md` (A at `lead_time = 0.50 s`; 16 scenes per condition; pass = at least 15/16). "Completes" = reaches and recovers after the noise stops; "holds" = at least 90% of noisy hold samples within 10 mm.

| Disturbance (per channel) | A completes up to | A holds up to |
|---|---|---|
| Arm torque, white | ≥ 8 N·m | 4 N·m |
| Arm torque, OU τ = 0.2 s | ≥ 8 N·m | 0.5 N·m |
| Arm torque, OU τ = 1.0 s | 4 N·m | 0.5 N·m |
| **Arm torque, constant offset** | **1 N·m** | **0.5 N·m** |
| Wheel speed, OU τ = 0.2 s, no gate | 1 rad/s | 0.1 rad/s |
| Wheel speed, OU τ = 0.2 s, off once reaching | 1 rad/s | 1 rad/s |
| Wheel speed, constant offset | 0.3 rad/s | 0.3 rad/s |

A tolerates large zero-mean noise but not persistent torque errors: without an integrator it settles outside 10 mm (16–60 mm short) under constant offsets above about 1 N·m. B, with its integrated reference, completes up to 4 N·m.

The DART protocol used for training (OU, τ = 0.2 s, arm σ ≤ 0.75 N·m, wheel σ ≤ 0.15 rad/s and off once reaching, all noise off 1.5 s after the reach) stays inside these limits: **441/442** DART episodes were held (`data/Markovian/DART/2026-10-28__collection__d2-s075-x2-58af5db87875/`).

## 6. Physics and push limits

`python -m evaluation.expert_t3_dev --controllers expert --lead-time 0.5 --out results/expert_t3_dev_lead050.csv`. 20 scenes per condition (10 for unreachable; development seeds 800000+). This script's predicate differs from the evaluator's: reach = first time within 10 mm; held = final error below 10 mm at the episode end (2 s after the reach, or after the push recovery).

| Condition | Reach | Held | Final error (median) |
|---|---|---|---|
| Nominal | 20/20 | 20/20 | 0.8 mm |
| Floor friction × 0.5 / × 1.5 | 20/20 / 20/20 | 20/20 / 20/20 | 0.8 mm |
| Sensor noise × 2 / × 20 / × 100 (std 1e-4 × scale) | 20/20 each | 20/20 each | 0.8 / 1.1 / 1.5 mm |
| **Link masses × 0.75** | 6/20 | **0/20** (6 unstable) | 27.3 mm |
| **Link masses × 1.25** | 11/20 | **7/20** | 20.6 mm |
| Combined: friction × 0.5, mass × 1.25, noise × 2 | 9/20 | 7/20 | 18.5 mm |
| Base push while driving (120 N × 0.2 s = 24 N·s at t = 3 s, random horizontal direction) | 12/20 | 11/20 | 4.7 mm |
| Base push 1 s after reaching (same impulse) | 20/20 | 16/20 (median recovery 4.2 s) | 1.2 mm |
| Hand push 1 s after reaching (20 N × 0.2 s) | 20/20 | 20/20 (median recovery 0.6 s) | 0.8 mm |
| Far targets 3–3.5 m | 20/20 | 20/20 | 0.7 mm |
| Unreachable targets 1.7–2.0 m high | 0/10 | 0/10 | 543 mm |

- **Mass mismatch is A's main physical weakness.** Its gravity compensation uses the nominal model and nothing integrates the residual, so it stops 20–30 mm short. With masses × 0.75, 6/20 episodes also drift in the arm's redundant directions.
- **Sideways base pushes during driving** can start a long slide, because the wheels resist lateral motion very little (friction 0.01) and the observation contains no base velocity.
- The 0.15 s code default does worse under mass mismatch (× 1.25: held 1/20, against 7/20 for 0.50 s; `results/expert_t3_dev.csv`). Values from that file describe a different teacher.

## 7. Known behaviours of teacher A

- **Clean data quality**: 320/320 sustained holds on the 256 training and 64 test scenes. About 10% of hold windows contain a 500 Hz joint-3 torque chatter of up to ±30 N·m while the end effector stays within 10 mm; these episodes are flagged `hold_chatter` and kept (`docs/reports/2026-10-26__report__clean-expert-data-collection.md`).
- **Joint-6 snap**: the initial arm pose puts joint 6 at 1.0 rad, below its limit of 1.134 rad, so the clipped reference produces a ≈ 10 ms jump when reaching starts. The template behaves the same.
- **Unobserved quantities**: base roll, pitch and velocity are not observed. Pitch is a calibrated constant (0.0022 rad). The PyBullet end-effector link carries a default 1 kg mass, which the teacher's model includes explicitly.
- **Local feedback structure**: where unclipped, ∂τ/∂q̇ = −30 I exactly; stiffness is near zero in the arm's null space (7 joints, 3-D task). This is the feedback a student must learn (see the training-method ablation).

## 8. Implications for the report

1. **The training range lies inside the feasible region.** Heights 0.1–1.25 m are within A's 0.1–1.30 m band, and distances 0.5–3.0 m are well within its near and far limits. No training scene asks for something the teacher cannot do.
2. **Read student failures against the teacher.** A scene counts against a student only if teacher A completes it. Heights above ≈ 1.30 m are physically unreachable; below 0.1 m, A itself falls short.
3. **Where the student is weaker than the teacher** (out-of-range probe, `python -m evaluation.generalisation_probe`, R4-dart-gate seeds 0–2, 12 scenes per condition; `results/ablation/2026-10-28__training-ablation-v1/generalisation/`):

   | Condition | Teacher A | R4 student |
   |---|---|---|
   | Inside the training ranges | 12/12 | 33/36 |
   | Target 3–4 m / 4–6 m | 12/12 / 12/12 | 35/36 / 33/36 |
   | Start 1–3 m from the origin | 12/12 | 36/36 |
   | Target 0.3–0.5 m away | 12/12 | **21/36** |
   | Target 1.25–1.4 m high | 8/12 | 13/24 where the teacher holds |
   | Target 0.03–0.1 m high | 4/12 | 6/12 where the teacher holds |

   The near-target gap (teacher 100%, student 58%) is a learning and data gap, not a physical limit: training never sampled targets closer than 0.5 m.
4. **Robustness claims** for the student should be stated relative to these teacher limits. Under link-mass mismatch or constant torque offsets the teacher itself misses the 10 mm tolerance, so a student cloned from it is not expected to do better.
5. **The choice of A over B as teacher** trades precision and steady-bias rejection (B: final error 0.1–0.8 mm, constant offsets up to 4 N·m) for a stateless teacher that can label any visited state, which DART and DAgger require. A also navigates more robustly: it has no timeout and replans after base pushes.

## 9. Evidence and reproduction

Run from `final/robotnik_arm_task/`. Result files are git-ignored and regenerate from these commands.

| Section | Command | Output |
|---|---|---|
| 3, 4 | `python -m evaluation.expert_workspace` and `--teacher B` | `results/diagnostics/2026-10-29__expert-workspace{,-B}/` |
| 5 | `python -m evaluation.expert_noise_robustness --n 16 --workers 22` | `results/expert_noise_robustness.csv` |
| 6 | `python -m evaluation.expert_t3_dev --controllers expert --lead-time 0.5 --out results/expert_t3_dev_lead050.csv` | `results/expert_t3_dev_lead050.csv` |
| 8 | `python -m evaluation.generalisation_probe --models <R4 seed exports>` | `results/ablation/2026-10-28__training-ablation-v1/generalisation/` |
| 2 | `python -m evaluation.plot_scenario_split` | scenario split figure |

Limitations: 8–20 scenes per condition give wide intervals (8/8 has a Wilson 95% lower bound of 0.68). Grid points are tested values, not continuous boundaries. Disturbance types were tested one at a time, except for the single combined condition.
