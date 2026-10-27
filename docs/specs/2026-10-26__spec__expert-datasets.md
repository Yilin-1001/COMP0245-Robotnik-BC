# Expert Dataset Specification

This document defines two datasets for learning the Robotnik mobile-base and arm reaching task. Dataset A uses the Markovian expert. Dataset B uses the original stateful controller. Both retain complete trajectories and raw observations so that feature and model ablations can reuse the same recordings.

This is a collection specification, not a report of completed collection. The current `collect.py` and `dagger.py` entry points are placeholders. Earlier planning documents contain different draft feature lists and sampling rules; the choices below describe the agreed design for these two datasets.

## 1. Purpose and agreed scope

| Property | Dataset A | Dataset B |
| --- | --- | --- |
| Teacher | `data_collection/expert.py`: Markovian expert | `robotnik_arm_control.py`, with the initialization and call order in `robotnik_arm_task.py` |
| Main purpose | Learn a reactive observation-to-action policy | Learn the original controller's history-dependent behavior |
| Default student | Single-step MLP | GRU policy; explicit reference-state policy as a separate experiment |
| Default input size per timestep | **21** | **26** |
| Action label size | **11** | **11** |
| Expert memory labels | Optional diagnostics | Required reference and navigation-state records |
| Temporal storage | Complete episodes | Complete episodes |
| Initial collection modes | Clean expert and action-noise rollouts | Clean expert and action-noise rollouts |
| Later extension | Student rollouts labelled by the same Markovian expert | Student rollouts labelled by the stateful expert maintained along those rollouts |

The body-frame unit gravity direction is deferred. It is not part of either default feature vector and is not a required derived field in this specification. Dataset B therefore has 26 inputs, rather than the previously proposed 29.

The teacher is available during collection and training-time relabelling. Deployment runs the student without the teacher. In particular, the true expert `q_des`, navigation phase and completion flag must never be supplied as ordinary student inputs.

## 2. Coordinate and timing conventions

- Positions are in metres, angles in radians, angular velocities in radians per second, torques in N m and time in seconds.
- `W` is the simulator world frame.
- `B_yaw` has its origin at the current base position, rotates with base yaw, and retains the world vertical axis. It is not the fully tilted body frame.
- For a world position `p_W`, `p_B = Rz(-base_yaw) @ (p_W - base_pos_W)`.
- For a world displacement `e_W`, `e_B = Rz(-base_yaw) @ e_W`; no translation is applied to a displacement.
- Timestep `t` denotes the observation before computing and executing action `a_t`. The resulting observation is `o_(t+1)`.
- Joint and wheel ordering follows `core/interfaces.py` and the controller: seven arm torques, followed by wheel speeds in front-right, front-left, back-left, back-right order.

Student end-effector features must come from a forward-kinematics estimator that can also run at deployment. With the current yaw-only observation interface, use observed arm angles, the recorded fixed finger configuration and a fixed, versioned roll/pitch assumption. Record the estimator configuration, including any fixed pitch calibration. Use the same estimator for training and inference.

Keep simulator-truth EE position and the teacher's own EE estimate separately. Neither may silently replace the student's estimate. This distinction matters for Dataset B because the original controller uses the full simulated base orientation, while the agreed student inputs omit measured roll/pitch and gravity direction.

## 3. Dataset A: Markovian expert

### Default student input

| Feature | Field or construction | Dimension | Purpose |
| --- | --- | --- | --- |
| Arm joint positions | `q_mes` | 7 | Arm configuration |
| Arm joint velocities | `qd_mes` | 7 | Velocity feedback and damping |
| Target position in `B_yaw` | Transform `target_ee_pos_W` using current base position and yaw | 3 | Target geometry relative to the robot |
| Estimated EE-to-target error in `B_yaw` | Target minus deployable FK estimate, expressed in `B_yaw` | 3 | Explicit reaching-error feature |
| Target world height | `target_ee_pos_W[2]` | 1 | Preserves the height used by the expert's standoff-distance rule |
| **Total** | | **21** | |

Target position and estimated EE error are intentionally related features: the error gives the network a useful geometric calculation explicitly. The default vector does not additionally include EE position and error magnitude, which can be reconstructed from the same records.

Record base forward speed and yaw rate for optional input ablations, but exclude them from the default input. The current expert does not use measured base velocity to calculate wheel commands. Adding both measurements produces a 23-dimensional variant; it is not automatically a better policy.

### Action labels

For each pre-action observation, save the clean expert command:

```text
a_expert[t] = [torque_arm[0:7], wheel_cmds[0:4]]  # 11 values
```

The label includes the expert's own output limits and is recorded before collection noise is added. Wheel commands are angular-velocity targets, not wheel torques. Arm commands are commanded torques, not necessarily the torques subsequently measured at the actuators.

Dataset A supports clean behavior cloning, noise-augmented behavior cloning, input ablations and later DAgger rounds. Complete temporal records also permit comparisons with history-based policies using the same teacher.

## 4. Dataset B: stateful expert

### Default student input

Start with the same 21-dimensional feature definition as Dataset A, then add:

| Additional feature | Field or construction | Dimension | Purpose |
| --- | --- | --- | --- |
| Fixed navigation goal relative to the current base | Rotate `base_goal_pos_W - base_pos_W[:2]` into `B_yaw` | 2 | Represents the goal used by the original base controller |
| Final heading error | `sin(base_goal_bearing - base_yaw)`, `cos(base_goal_bearing - base_yaw)` | 2 | Represents the required final base orientation |
| Elapsed time on a fixed scale | `elapsed_s / time_scale_s` | 1 | Preserves information needed for the navigation timeout |
| **Additional dimensions** | | **5** | |
| **Total per timestep** | | **26** | |

`base_goal_pos_W` and `base_goal_bearing` are fixed episode task references. Compute them once using the selected task-initialization rule, save their actual values, and make that same task information available at deployment. This does not require running the expert's feedback controller. The original initialization derives its goal bearing from the world target relative to the world origin; do not silently replace this with a start-relative rule during collection. If that rule is changed later, version it and evaluate the changed teacher separately.

Use a fixed, recorded positive `time_scale_s`, such as the configured finite navigation timeout. Do not normalize time by the observed final episode duration, because that duration is unknown during deployment. The original controller's reference is not a time-interpolated trajectory: elapsed time does not replace its reference memory.

The default initial arm and finger poses remain fixed as in the current task. Record them even when constant. If initial arm poses are randomized later, the policy must retain the initial observation or receive the known initial hold configuration as episode context; this changes the initialization requirements.

### Required expert-state records

Capture the state before and after each expert call. Store snapshots as copies so that later in-place updates cannot overwrite earlier records.

| Field | Dimension/type per step | Meaning and use |
| --- | --- | --- |
| `q_des_before` | 7 floats | Reference supplied to the arm controller at this step |
| `q_des_after` | 7 floats | Updated and limited reference used to compute this step's torque |
| `qd_des` | 7 floats | Desired joint velocity returned by the arm controller |
| `phase_before`, `phase_after` | Categorical values | Base phase before and after the base-controller call |
| `nav_done_before`, `nav_done_after` | Booleans | Navigation completion flag before and after the call |
| `reach_active` | Boolean | Flag passed to the arm controller after the base update |
| `q_hold` | 7 floats per episode | Initial arm configuration held during navigation |

The code stores `phase` as `align`, `drive` or `orient`, and stores completion separately in `done`. Preserve these separately rather than assuming the phase field itself becomes `done`. Also keep navigation completion separate from environment termination or reaching success.

Recommended auxiliary training targets are:

```text
reference_error_target[t] = q_des_after[t] - q_mes[t]  # 7 values
phase_target[t]           = phase_after[t]             # classification
nav_done_target[t]        = nav_done_after[t]          # binary classification
```

These are labels, not default inputs. A GRU can learn an internal representation using action loss plus auxiliary prediction losses. An explicit reference-state student can instead predict and carry its own reference estimate between steps. Its training must include sequences using its own predicted state; always feeding true expert references during training would create a deployment mismatch.

Do not replace `q_des` with an independently accumulated Cartesian tracking error. The original reference update contains differential IK, changing Jacobians, damping, velocity limits, phase-dependent holding and anti-windup clipping. A three-dimensional EE reference also cannot fully represent a seven-dimensional joint reference.

### Remaining observation limitation

Omitting gravity direction does not make Dataset B fully observable. The original expert still uses full base orientation in kinematics and gravity compensation. Record this teacher-side information for diagnostics where available, without exposing it through student features. Exact action matching is not guaranteed when the student lacks information used by the teacher. Evaluate the significance through closed-loop tracking and recovery tests.

## 5. Shared trajectory structure

Separate deployable observations, action labels, executed commands, expert state and diagnostics. Store raw data as the source of truth and derive feature vectors at load time. A file format can be selected during implementation; the logical layout is:

```text
dataset_manifest
  schema_version, dataset_id, teacher_id, teacher_version
  feature_definition_version, FK_configuration
  scenario_split_definition, collection_configuration

episode
  metadata
    episode_id, scenario_id, split, source, random_seeds
    target_ee_pos_W, base_start_pose, q_init, fixed_finger_angles
    teacher_parameters, physics_parameters, sensor_configuration
    navigation_goal_and_time_scale  # required for B
    control_hz, noise_configuration, stopping_configuration
    learner_checkpoint_and_beta     # for DAgger episodes
  observations[0:T+1]
    timestamp, base_pos_W, base_yaw, q_mes, qd_mes
    base_forward_speed, base_yaw_rate
  transitions[0:T]
    dt, a_expert, a_executed, sampled_action_noise
    execution_source, command_clipping_flags
    terminated, truncated, termination_reason
  expert_state[0:T]                 # required for B
    q_des_before, q_des_after, qd_des
    phase_before, phase_after, nav_done_before, nav_done_after
    reach_active
  diagnostics
    true_EE_position, teacher_EE_estimate, true_base_orientation
    tracking_error, perturbation_status, success_and_failure_flags
```

An episode with `T` executed actions has `T + 1` observations. Constant target and task information can be stored once per episode. The previous executed action is recovered by shifting `a_executed` within an episode; the first step needs an explicit initialization convention and validity mask.

Record the source of base-speed measurements. Simulator-truth velocities must be marked as such and must not be presented as encoder or odometry estimates. They only become valid deployment inputs if an equivalent sensor estimate is available. If teacher and student receive different noisy or privileged observations, preserve the teacher-side values or their reproducible generation procedure separately.

Fit normalization statistics only on the training split. Record separate action scales for arm torque and wheel speed. Keep A and B teacher labels identifiable; do not merge their potentially different actions into one unlabeled regression target.

## 6. Clean, noise-augmented and DAgger collection

### Exact order within a control step

1. Read and save the pre-action observation `o_t`.
2. For B, snapshot the teacher's pre-call state.
3. Query the teacher once. For B, update the base controller first, then the arm controller, matching the original call order.
4. Save `a_expert[t]` and the post-call teacher state.
5. Choose the behavior command, add any configured collection noise, apply execution limits and save the final `a_executed[t]`.
6. Execute that command for one control period and save `o_(t+1)` and post-action diagnostics.

The current `env.step()` returns EE information after execution. Do not attach that EE position to the preceding observation when building student features.

For clean collection, `a_executed` equals the expert command, subject to the documented execution interface. For action-noise collection, store the clean expert label and the actual perturbed command separately. Save noise parameters and samples, including temporal correlation or hold duration if used. Fixed noise injection is a DART-style collection method; reproducing full DART additionally requires its noise-selection procedure.

For DAgger, the student or expert mixture drives the environment, while the teacher supplies labels on the resulting trajectory. In B, initialize teacher memory at the episode start and update it at every control step using that same trajectory, even when its proposed action is not executed. Do not import reference states from a separate clean expert rollout or reset the expert at each query. Preserve the sequence so that student memory can be recomputed under updated network weights during retraining.

### Scenario coverage and quality

- Use the same scenario list and declared disturbance rules for A and B. Their resulting trajectories need not be identical.
- As an initial budget example, 256 training scenarios with one clean and one action-noise rollout each produce 512 episodes per dataset. Validate the teacher and noise settings in a smaller pilot before scaling. Validation and test scenarios are additional, separate sets.
- Split by scenario before training. Clean/noisy variants and repeats of one scenario must stay in the same split, consistently across both datasets.
- Cover alignment, driving, orientation, reach activation, approach, settling and sustained holding. Continue recording after first reaching the tolerance, using a fixed recorded hold duration such as three seconds.
- Start with nominal physics and controlled action disturbances. Record mass, friction, sensor-noise and external-push variations as separate, identifiable conditions and verify expert performance in each.
- Retain failed episodes with failure and label-quality flags. Apply an explicit, reported rule when selecting them for training; never silently discard them from collection statistics.
- The original base controller latches `done` and then outputs zero wheel speeds. Test its recovery behavior before treating post-navigation base-push labels as corrective demonstrations.

### Sampling and temporal integrity

Collect and retain every control step at the current 1 kHz control frequency. Do not hold an action for ten physics steps merely because a training loader uses a stride of ten.

Independent MLP samples may be subsampled. A recurrent model deployed at 1 kHz must also learn the corresponding memory-update cadence. Options include full-rate forward unrolling with losses on selected steps, or a separately validated lower-rate recurrent design. Blindly decimating a sequence changes the dynamics of its memory updates.

Training chunks must preserve episode boundaries and temporal order. Carry hidden state across ordered chunks or use a suitable history warm-up. Do not initialize every middle-of-episode chunk as if it were a fresh episode. Two to five frames at 1 kHz cover only milliseconds and are not an adequate default assumption for navigation memory.

## 7. What the datasets let us test

| Question | Controlled comparison |
| --- | --- |
| Does action-noise collection improve recovery? | Clean versus clean-plus-noise data from the same teacher, with sample budget controlled |
| Does history help imitate the original controller? | Single-step MLP versus GRU on the same B episodes and per-step observations |
| Does expert-state supervision teach useful memory? | GRU action loss versus GRU action loss plus reference/phase supervision on B |
| Can explicit reference memory work? | Explicit reference-state policy versus GRU, using the same B data and deployment information |
| Does the previous action help or create a shortcut? | With versus without previous executed action, evaluated on transitions and disturbances |
| Does DAgger help beyond adding more data? | Before/after DAgger with an equal-data expert-collection control |

Report closed-loop reaching success, sustained-hold error, recovery time, saturation and transition behavior alongside action-prediction error. For B, also report reference and phase prediction quality when auxiliary heads are used. A previous-action-copy baseline is useful for interpreting a low offline action loss.

Comparing an A-trained MLP with a B-trained GRU changes the teacher, observation features and architecture together. It compares two complete approaches; it does not isolate the value of recurrence. Use within-dataset comparisons for that claim.

## 8. Research basis

See the [research reference guide](../references/2026-10-26__reference__research-bibliography.md) for the categorized bibliography, official projects, theoretical limitations and the distinction between published findings and project-specific design choices.

- [Swamy et al., *Sequence Model Imitation Learning with Unobserved Contexts*, NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/708e58b0b99e3e62d42022b4564bad7a-Paper-Conference.pdf): motivates learning from history together with supervision on student-induced trajectories; adding history to offline BC alone can encourage repetition of past mistakes.
- [Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild*, 2022](https://arxiv.org/abs/2201.08117), with [the authors' method description](https://leggedrobots.org/lrws_2022/assets/pdfs/paper7.pdf): provides precedent for recurrent students trained with action imitation and privileged-information reconstruction. Using `q_des` and navigation state as reconstruction targets is a project-specific adaptation.
- [Wen et al., *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf): motivates explicit tests for copying previous actions rather than treating history as an automatic improvement.
- [Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, CoRL 2017](https://proceedings.mlr.press/v78/laskey17a.html): motivates disturbance-driven data collection while retaining the supervisor's corrective action labels.

Relevant local implementation references: [expert](../../final/robotnik_arm_task/data_collection/expert.py), [original controllers](../../final/robotnik_arm_task/robotnik_arm_control.py), [task initialization](../../final/robotnik_arm_task/robotnik_arm_task.py), [observation/action interface](../../final/robotnik_arm_task/core/interfaces.py), [environment](../../final/robotnik_arm_task/core/env.py), and [expert model accuracy](../reports/2026-10-24__report__markovian-expert-model-accuracy.md).
