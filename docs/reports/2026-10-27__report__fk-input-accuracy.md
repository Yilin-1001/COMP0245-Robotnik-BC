# FK End-Effector Position in the Student Inputs: Definition and Direct Accuracy Check

Date: 2026-10-27. This is a post-training diagnostic only. No inputs, models, physics, scenarios or model selection were changed, and no new test data was accessed.

## What the FK computes

FK (forward kinematics) computes the position of the end-effector frame origin. It uses the URDF link geometry, the joint axes and the current joint angles. The implementation is `core.features.DeployableFK.ee_pos()`, which uses Pinocchio to return the world position of the `robot_j2s7s300_end_effector` frame.

Its inputs are the observed base xyz, the base yaw and the 7 arm joint angles. Base roll is fixed at 0, base pitch at 0.0022 rad, and the 6 finger angles at 0.1 rad. It is a deterministic geometric computation. The end-effector error input is `Rz(-yaw) × (target_world − ee_FK_world)`.

The fixed pitch is a flat-ground calibration. The observation contains no true roll or pitch, so changes in base attitude cause position errors. The URDF mass correction mainly affects gravity torques; it does not change the FK link geometry.

## Direct measurement

The first 4 fitting scenarios of the 8 frozen recovery probes were selected in advance: `train-0001`, `train-0035`, `train-0067`, `train-0108`. Each was run with expert A and with the then-selected R1 student (step 5000). The expert used lead_time 0.50; the student used the original 21 inputs and frozen normalisation. Control ran at 1 kHz, and positions were compared every 10 steps (10 ms).

The reference position is read directly from PyBullet `getLinkState(...)[4]`, the world origin of the same EE link frame. This avoids comparing a centre-of-mass position with a link origin. `RobotEnv.ee_pos()`, the FK with the true full base attitude, was checked as well. Simulator truth went only into the diagnostic log, never into a policy.

| Euclidean FK error vs PyBullet position | Expert, 4 trajectories | R1 student, 4 trajectories |
|---|---:|---:|
| Sampled observations | 4604 | 16000 |
| Median | 0.507 mm | 0.607 mm |
| 95th percentile | 1.539 mm | 1.694 mm |
| Sampled maximum | 3.918 mm | 5.435 mm |
| Fraction of samples above 10 mm | 0 | 0 |
| Median over the final 1 s of each trajectory | 0.643 mm | 0.734 mm |

The maximum refers only to the 10 ms samples. It is not an upper bound over every 1 ms physics state. The check covers only 4 nominal scenarios, so it does not generalise to arbitrary disturbances, tilt or a physical robot.

With the true full base attitude, `RobotEnv.ee_pos()` differs from the PyBullet position by less than 0.000129 mm in every record. The millimetre-level errors in the table therefore come from the deployed FK's fixed roll/pitch approximation, not from a mismatch in the URDF geometry.

All 4 expert trajectories held the target. None of the 4 student trajectories reached it: their final true errors were about 2100, 1962, 1575 and 1090 mm. The FK error is far smaller than these task errors, so this evidence does not attribute those failures mainly to the FK. The millimetre-level approximation still uses part of the 10 mm margin, which is a separate question from precise holding.

## Difference from the earlier accuracy report

The 0.2–3.2 mm in the [earlier expert model accuracy report](2026-10-24__report__markovian-expert-model-accuracy.md) is the expert's final control error at the target, not a direct FK position error. The table above compares the FK-predicted position with the true link position, which is a different quantity.

## Results and reproduction

- [Raw per-sample diagnostics and summary](../../Archive/final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/fk_accuracy.json).
- [Frozen probes, model and code hashes](../../Archive/final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/fk_accuracy_contract.json).
- [Reproduction script](../../Archive/final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/2026-10-27__analysis__fk-input-accuracy.py). The script matches its recorded source SHA. To rerun it, use a new result directory so the original frozen evidence is kept.
- [FK implementation](../../final/robotnik_arm_task/core/features.py).

No production code was changed. Verification consisted of 8 full closed-loop PyBullet replays and an independent geometric check of the PyBullet link frame against the full-attitude FK.
