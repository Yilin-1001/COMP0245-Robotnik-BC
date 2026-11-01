# Robotnik task entry point

Run every project command from this directory. Follow the [project README](../../README.md) to install the environment, collect data, train R4, evaluate it and run the DAgger rows.
Test-set commands are in [step 5: Test on all 64 test scenarios](../../README.md#5-test-on-all-64-test-scenarios). The method and results are in the [project pipeline](../../docs/2026-10-30__guide__project-pipeline.md).

| Folder | Contents |
|---|---|
| `core/` | Interfaces, environment, features and the reach gate |
| `data_collection/` | Teacher A (`expert.py`), teacher B wrapper, scenarios, clean, DART and DAgger collection, QA |
| `training/` | Model, engine, losses, dataset build and the training entry point |
| `evaluation/` | Closed-loop evaluation, test diagnosis, probes, hold test and live comparison |
| `experiments/` | Configs, frozen split, ablation v1 runner and manifest, DAgger runner (v2), demo manifests |
| `tools/` | One-off data tools |
| `data/`, `checkpoints/`, `results/` | Git-ignored outputs, except `results/results.csv` and the experiment summaries |

## Original course-template notes

The notes below come with the course template and describe its expert controller demonstration.
`models/` and `configs/` are already in this folder, and the environment comes from the project README, so the template's first two setup steps are not needed.

Run `robotnik_arm_task.py` (or `ROBOENV_HEADLESS=1 python robotnik_arm_task.py` without a display).

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.

 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)
