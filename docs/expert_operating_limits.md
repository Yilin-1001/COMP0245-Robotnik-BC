# Expert Operating Limits

Reviewed: 26 October 2026.

This document records the measured operating limits of the Markovian expert (A) and the original stateful template expert (B). Both can solve nominal reaching tasks, but their disturbance responses differ: A is sensitive to persistent arm torque errors, while B is sensitive to wheel disturbances that disrupt navigation and to displacement after navigation completion.

The values below are empirical results on a finite development set. They are not certified safety limits, continuous stability boundaries or guarantees for every target. Current collection remains **clean only**; this document does not authorize or require noisy dataset collection.

## 1. Evidence and controller versions

| Item | Source or configuration |
| --- | --- |
| Stress-test implementation | [evaluation/expert_noise_robustness.py](../final/robotnik_arm_task/evaluation/expert_noise_robustness.py) |
| Per-episode results | [results/expert_noise_robustness.csv](../final/robotnik_arm_task/results/expert_noise_robustness.csv), generated and git-ignored |
| Expert A | [data_collection/expert.py](../final/robotnik_arm_task/data_collection/expert.py), explicitly instantiated with **`lead_time=0.50 s`** in this sweep |
| Expert B | [data_collection/template_teacher.py](../final/robotnik_arm_task/data_collection/template_teacher.py), calling the original [arm and base controllers](../final/robotnik_arm_task/robotnik_arm_control.py) |
| Simulation interface | [core/env.py](../final/robotnik_arm_task/core/env.py) |
| Task ranges and tolerance | [settings.yaml](../final/robotnik_arm_task/settings.yaml) |

The CSV contains **2,112 episodes: 66 conditions × 2 teachers × 16 development scenarios**. The conditions comprise one clean baseline, 25 arm-noise conditions and 40 wheel-noise conditions. Arm and wheel disturbances were tested separately, not simultaneously.

The same scenario and, within a condition, the same sampled noise sequence are used for both teachers. Development seeds are `870000` through `870015`. The script uses pseudorandom uniform sampling, not Sobol sampling. Its scenario ranges are:

| Parameter | Range |
| --- | --- |
| Target horizontal distance relative to the starting base | 0.5–3.0 m |
| Target azimuth relative to initial base heading | Approximately −pi to pi |
| Target world height | 0.1–1.25 m |
| Initial base x and y | Each −0.5 to 0.5 m |
| Initial base yaw | −pi to pi |

Physics is nominal, sensor noise and external pushes are disabled, and the initial arm and finger poses are fixed. B retains the original world-origin goal rule, 25 s navigation timeout, base-before-arm call order and full simulator orientation for teacher-side kinematics.

**Version distinction:** the repository's default A parameter is still `lead_time=0.15 s`; the stress-test script overrides it to `0.50 s`. These thresholds describe the overridden version. They must not be assigned to the default version without another test. Earlier model-accuracy and OU pilot reports also use different protocols; their numerical limits are not interchangeable.

## 2. Disturbance and outcome definitions

### 2.1 What is perturbed

Noise is added to the expert's command before execution:

```text
executed_arm_torque = clip(expert_arm_torque + arm_noise, -30, 30)  # Nm
executed_wheel_speed = clip(expert_wheel_speed + wheel_noise, -6, 6)  # rad/s
```

The experts then observe the resulting physical state and continue controlling. Teacher wheel commands normally reach about 5.3 rad/s; the sweep's execution cap is 6 rad/s. This cap differs from the approximately 5.273 rad/s cap used in the separate D2 pilot.

Sigma is a **per-channel standard deviation**, not a bound on individual noise samples. Arm channels all receive the same sigma in Nm, although their dynamics differ. Wheel channels are also sampled independently.

| Noise structure | Meaning in this implementation |
| --- | --- |
| White | Independent zero-mean Gaussian samples at every 1 ms step |
| OU | Zero-mean temporally correlated Gaussian noise with stationary standard deviation sigma; correlation times 0.05, 0.2 or 1.0 s |
| Constant offset | One independent Gaussian draw per channel at episode start, retained throughout the noisy phase |

For a constant-offset condition labelled `2 Nm`, each joint does **not** necessarily receive exactly +2 Nm. Its fixed offset is drawn from a zero-mean Gaussian with standard deviation 2 Nm. OU is initialized from its stationary distribution in this sweep; the earlier D2 pilot initialized OU at zero.

Arm sigma grid: **0.5, 1, 2, 4, 8 Nm**. Wheel sigma grid: **0.1, 0.3, 1, 3 rad/s**.

### 2.2 Timing and success criteria

1. Start applying the selected noise at episode start.
2. Record the first true end-effector error below **10 mm**. In this script, reach detection does not additionally require the original task's minimum time or early-stop predicate.
3. Continue noise until the earlier of first reach plus **3 s**, or the **30 s** noisy-phase cap.
4. Remove all noise. Allow up to **10 s** of clean recovery, stopping after the error has remained below 10 mm continuously for **1 s**.

The original task's early stall termination is not enforced by this sweep.

| Outcome | Per-episode definition | Condition-level pass |
| --- | --- | --- |
| Reached under noise | First reach occurs no later than 30 s | Report count out of 16 |
| Completes | The same episode reaches under noise **and** passes clean recovery | At least 15/16 episodes |
| Holds | At least 90% of the recorded noisy hold samples are below 10 mm | At least 15/16 episodes |
| Recovered | Passes the continuous 1 s clean-recovery check, including episodes that first reach only after noise removal | Report separately; not sufficient for completion |

For completion, intersect `reached` and `recovered` per episode; separate marginal counts are not enough. Four episodes in this CSV first reach after 27 s, so their noisy hold windows are shorter than 3 s because of the cap. The hold statistic is therefore not always a full three-second test.

### 2.3 Wheel gating is teacher-dependent

In gated conditions the script sets wheel noise to zero while the returned navigation/reaching flag is true:

- **B:** the flag is its latched `nav_done`, so wheel noise remains off after completion, including completion caused by timeout.
- **A:** the flag is the current `reach_active` condition. It can become false again, allowing wheel noise to resume. The sweep does **not** add a permanent first-activation latch for A.

These gated results differ from the proposed D2 scheduler, which permanently disables wheel noise on first activation for both experts. Gating also cannot repair a navigation failure that happens before the flag is set.

## 3. Measured threshold summary

Each cell is the largest **tested** sigma meeting the corresponding 15/16 criterion. `>=` means the largest grid value passed and a higher value was not tested. `None` means none of the tested nonzero values passed. A largest passing grid point does not establish that all smaller continuous values pass, or that the next grid point is an exact physical limit.

### 3.1 Arm torque disturbances, Nm per joint

| Structure | A completes | A holds | B completes | B holds |
| --- | ---: | ---: | ---: | ---: |
| White | >=8 | 4 | >=8 | >=8 |
| OU, tau = 0.05 s | >=8 | 1 | >=8 | 1 |
| OU, tau = 0.2 s | >=8 | 0.5 | >=8 | 0.5 |
| OU, tau = 1.0 s | 4 | 0.5 | >=8 | 0.5 |
| Constant offset | **1** | **0.5** | **4** | **2** |

### 3.2 Wheel disturbances without gating, rad/s per wheel

| Structure | A completes | A holds | B completes | B holds |
| --- | ---: | ---: | ---: | ---: |
| White | >=3 | >=3 | 0.3 | 0.3 |
| OU, tau = 0.05 s | >=3 | 0.3 | 0.3 | 0.1 |
| OU, tau = 0.2 s | 1 | 0.1 | 0.3 | 0.1 |
| OU, tau = 1.0 s | 1 | 0.1 | 0.3 | None |
| Constant offset | 0.3 | 0.3 | **0.1** | None |

### 3.3 Wheel disturbances with the sweep's gate, rad/s per wheel

| Structure | A completes | A holds | B completes | B holds |
| --- | ---: | ---: | ---: | ---: |
| White | >=3 | >=3 | 0.3 | 0.3 |
| OU, tau = 0.05 s | 1 | 1 | 0.3 | 0.3 |
| OU, tau = 0.2 s | **1** | **1** | **0.3** | **0.3** |
| OU, tau = 1.0 s | 1 | 1 | 0.1 | 0.1 |
| Constant offset | 0.3 | 0.3 | 0.1 | 0.1 |

Some gated/ungated thresholds do not order monotonically. The script seeds noise by condition index, and gated and ungated conditions have different indices. They therefore do not receive identical noise realizations, even though A and B are paired within each condition. Interpret differences as finite-sample results, not isolated causal estimates of gating alone.

## 4. Expert A: Markovian controller limits

### 4.1 Persistent torque errors expose missing accumulated compensation

A rebuilds its reference from current state:

```text
q_des = q + lead_time * qd_des
```

Its reaching branch has no accumulated reference offset to retain compensation for an unknown persistent torque disturbance. A nonzero Cartesian error can generate corrective desired velocity and torque, but the hand may settle outside the target tolerance. This interpretation is consistent with the current controller and the [earlier model-accuracy diagnosis](expert_model_accuracy.md); this sweep does not itself repeat mass-randomization tests.

| Constant arm sigma | A reaches under noise | A completes including recovery | A holds |
| --- | ---: | ---: | ---: |
| 0.5 Nm | 16/16 | 16/16 | 16/16 |
| 1 Nm | 16/16 | 16/16 | 14/16 |
| 2 Nm | 11/16 | 10/16 | 6/16 |
| 4 Nm | 2/16 | 2/16 | 1/16 |
| 8 Nm | 1/16 | 0/16 | 0/16 |

At 4 Nm, at least one episode is flagged unstable and the expert torque is at its limit on roughly half of noisy steps. The CSV counts saturation when **any** arm joint reaches its command limit; it does not identify the joint or preserve a full trajectory. Joint-specific oscillation and peak-error claims require the underlying trace, not this aggregate CSV alone.

This is a limitation of this reference law, observation set and model under unmodelled disturbances, not a universal statement that Markovian controllers cannot reject constant disturbances. Compensation is also constrained by torque limits, joint limits and reachability.

### 4.2 Moving the base compromises precise holding

At wheel OU sigma 0.3 rad/s and tau 0.2 s, A reaches in 16/16 episodes but holds in only 11/16. At 1 rad/s it still reaches in 16/16 but holds in 0/16. With the sweep's gate, the 1 rad/s condition passes both criteria in 16/16 episodes.

A can reactively adjust its base command, but this does not guarantee the hand remains within 10 mm while the base is continually disturbed. Its kinematics also use a fixed base pitch rather than the true changing roll/pitch, adding an observation/model limitation under larger base motion.

### 4.3 Large offsets enter a saturated regime

At wheel constant-offset sigma 3 rad/s, A reaches in 0/16 episodes. The mean execution-clipping fraction is about 26%. This is a severe disturbance relative to nominal wheel commands. It demonstrates failure of the tested closed loop with its action limits; it does **not** prove an actuator-only limitation independent of controller logic.

## 5. Expert B: stateful template controller limits

### 5.1 Navigation has limited recovery paths

B progresses through `align`, `drive` and `orient`, with completion stored separately in a latched `done` flag. The drive branch caps requested **base yaw rate** at +/-0.3 rad/s and does not transition back to `align` after a large heading deviation. That base-yaw command must not be confused with the wheel-noise sigma, although both are expressed in rad/s.

Noise can disrupt heading and translation sufficiently to prevent phase completion within 25 s. For wheel OU sigma 1 rad/s and tau 0.2 s without gating:

- only 5/16 episodes reach;
- 10/16 are flagged as navigation timeouts;
- median maximum displacement after the first completion flag is about 3.10 m;
- mean execution-clipping fraction is only about 0.1%.

These observations support a navigation/recovery limitation rather than a failure explained solely by wheel-command clipping. The environment's anisotropic wheel-contact setting `[1.0, 0.01, 1.0]` permits weak lateral resistance, consistent with reported sliding. The CSV does not contain heading or lateral-velocity traces, so detailed sliding mechanisms remain trace-based explanations rather than independently reconstructed measurements here.

### 5.2 Completion and timeout both disable future base corrections

Once `done` is true, B returns zero wheel commands indefinitely. It does not replan after displacement. The timeout also sets this flag; a completion flag is therefore not proof of successful navigation.

At wheel OU sigma 0.3 rad/s and tau 0.2 s:

| Condition | Reaches | Holds | Navigation timeouts |
| --- | ---: | ---: | ---: |
| No gate | 15/16 | 3/16 | 1/16 |
| Sweep gate | 15/16 | 15/16 | 1/16 |

The gated result is consistent with avoiding disturbance after the controller has stopped correcting the base. It does not show that B has acquired post-completion recovery, and the conditions use different noise realizations.

Zero wheel commands are not position feedback. Wheel velocity targets also do not imply instantaneous zero physical velocity. After a timeout the base can remain displaced or keep moving. Only the wheel part of the action becomes zero; the arm may continue producing nonzero torques.

### 5.3 Bias can prevent tight phase tolerances from being met

The final orientation tolerance is 0.05 rad. A persistent disturbance can leave a residual error that prevents the orient phase from completing; disturbances can also disrupt the earlier drive phase. At constant wheel-offset sigma 0.3 rad/s, B reaches only 3/16 episodes, with seven flagged navigation timeouts and no execution clipping. The CSV does not isolate one specific failure mechanism for every episode.

### 5.4 Accumulated arm reference improves steady-bias tolerance

B integrates desired joint velocity into `q_des`, with anti-windup and joint-limit clipping. The retained reference offset can provide compensation after the hand approaches its target. At constant arm sigma 2 Nm, B completes and holds in 15/16 episodes. At 4 Nm it completes in 15/16 and holds in 14/16, substantially better than A on the same conditions.

This memory is useful behavior for the stateful student to learn. It does not guarantee rejection of arbitrary offsets: saturation, reference clipping and reachability still bound compensation. Teacher `q_des` remains an auxiliary training label, not an available deployed student input.

## 6. Shared constraints and interpretation

- **Completion is easier than precision under continuing noise.** With arm OU tau 0.2 s and sigma 1 Nm, both complete in 16/16, but only A 9/16 and B 12/16 pass noisy holding. At 2 Nm, completion is A 15/16 and B 16/16, while holding is only 1/16 for each. A recommendation of 1–2 Nm must therefore specify a recovery objective, not reliable 10 mm holding under continuing disturbance.
- **Temporal structure matters.** White noise at 1 kHz produces much less persistent deviation in these tests than OU or a constant offset at the same sigma. Report distribution, sigma, correlation time, channel, gating and clipping together. High white-noise thresholds are not interchangeable with equally large biases.
- **Physical and task limits remain.** Both arm controllers project distant goals to a 0.85 m shoulder-centered reach limit and enforce joint, velocity and torque limits. Projection can prevent an unreachable target from being matched exactly. Bounded commands do not guarantee stable motion.
- **Information differs.** B uses full true orientation in teacher kinematics; A uses a fixed pitch assumption. A/B comparisons therefore change more than the presence of memory. Neither teacher result guarantees that a student with less information can match every action.
- **Nominal success is local evidence.** Both teachers reach, hold and recover in 16/16 clean development episodes. This does not establish success on all future starts, targets or changed physics.

## 7. Implications for collection and learning

### Current clean collection

Proceed with clean collection under nominal physics and the declared task distribution. Record the actual expert parameters, scenario identity, outcome and failure reason. Preserve failed attempts in raw storage; the first clean BC training manifest can select episodes that pass the agreed reaching/holding checks and data-integrity checks. Report the full attempted denominator and exclusions.

Use the [dataset specification](expert_datasets.md) for fields and sequence timing. The current discussion prioritizes clean data; the noisy modes described in that specification remain later extensions. Freeze A's `lead_time` explicitly before collection rather than treating the 0.15 and 0.50 s versions as the same expert.

### Future disturbance collection or DAgger

1. Select the objective first: reaching under disturbance, holding under continuing disturbance, or recovery after disturbance removal.
2. Treat the table as a candidate-selection aid. Independently test the chosen combined arm-and-wheel condition; separate channel passes do not prove a combined pass.
3. If comparing teachers under a common disturbance condition, use the same declared units, scales, noise-generation rules and seed policy. Different per-teacher scales are separate experimental conditions.
4. Gating can limit collection to behavior B can supervise, but does not improve its underlying controller. Define whether A's gate is instantaneous or permanently latched.
5. Keep clean expert labels separate from executed perturbed actions. Flag unrecoverable navigation failures; do not automatically interpret prolonged zero-wheel labels as useful recovery supervision.
6. DAgger can address the student's visited-state distribution, but cannot make an incapable teacher supply successful recovery actions. A stateful teacher must still be updated along the actual rollout.
7. Fixed or manually chosen OU noise is DART-style augmentation. Full DART also estimates and updates the disturbance distribution using student error. See the [research reference guide](references.md).

## 8. Statistical and reproducibility limits

- **Only 16 distinct scenarios per condition.** The 2,112 episodes reuse them across conditions. A 15/16 success count has a two-sided 95% Wilson lower bound of approximately **71.7%**, not evidence of a guaranteed 93.75% population success rate.
- **Coarse grid.** Intermediate sigma values and additional noise seeds were not exhaustively tested. Different conditions use different noise realizations, so a perfectly monotonic response is not expected from these counts.
- **No joint channel sweep.** The artifact cannot certify any simultaneous arm/wheel configuration.
- **Unequal joint effects.** Equal Nm standard deviations are not equal relative disturbances for shoulder and wrist joints.
- **Protocol dependence.** Reach, noisy hold and recovery definitions, OU initialization, execution limits and gating differ from some earlier pilots. Four hold windows are truncated by the noisy-phase cap.
- **Aggregate evidence.** Saturation and drift summaries do not preserve enough information to reproduce joint-level or sliding traces. The `unstable` flag counts more than 50 sampled steps with any arm speed above 5 rad/s; those steps need not be consecutive.
- **Development evidence only.** These results were reported as development runs and do not replace final held-out evaluation. The CSV stores seeds and outcomes but not a complete immutable runtime/configuration manifest; source links describe the files inspected for this note, not a fully archived execution environment.

The CSV checked for this document has SHA-256:

```text
9dcc66ecbc42ad8023171b6ddfdfc53a87900ac83b0c247af1e8504aed35c039
```

To reproduce the declared sweep, run from `final/robotnik_arm_task` in the simulation environment:

```bash
python -m evaluation.expert_noise_robustness --n 16 --workers 22
```

The default command writes the generated CSV path above. Archive an existing result or pass a separate `--out` path when comparing runs. Record source revision, expert parameters, environment versions and the exact outcome definitions with every future result.
