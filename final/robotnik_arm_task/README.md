# Robotnik task entry point

Final upload weights: [`checkpoints/2026-11-05final/`](checkpoints/2026-11-05final/README.md).
The [model naming guide](../../docs/model-naming.md) maps every Report name to its canonical ID and historical alias.

**Run every project module from this directory.** Start with the [assessor's run guide](../../README.md) for installation and the project framework.

- [Read what model seeds mean](../../README.md#3-before-choosing-a-model-what-do-the-seeds-mean) before choosing weights.
- [Choose among the 27 submitted models](../../README.md#4-submitted-model-choices-and-filenames).
- [Run live comparison](../../README.md#5-live-comparison-expert-versus-your-chosen-model), with a copyable model-name list, model/seed selection, scene overrides, keyboard controls and a headless check.
- [Apply mouse-drag disturbances](../../README.md#mouse-drag-disturbances-inspect-robustness-interactively) in the PyBullet GUI and observe recovery after release.
- [Run validation and unified tests](../../README.md#6-validate-and-test-the-submitted-models).
- [Reproduce collection, feature building and training](../../README.md#7-reproduce-the-data-and-training-pipeline).

Live comparison and validation need the released weights/configs and frozen validation split; training datasets are not needed. Pass `--protocol checkpoints/2026-11-05final/demo_models.yaml` explicitly to select the submitted weights. The method and results are in the [project pipeline](../../docs/2026-10-30__guide__project-pipeline.md) and [report evidence](report/README.md).

| Folder | Contents |
|---|---|
| `core/` | Interfaces, environment, features and the reach gate |
| `data_collection/` | Teacher A (`expert.py`), teacher B wrapper, scenarios, clean, OU, full DART and DAgger collection, QA |
| `training/` | Model, engine, losses, dataset build and the training entry point |
| `evaluation/` | Closed-loop evaluation, test diagnosis, probes, hold test and live comparison |
| `experiments/` | Configs, frozen split, ablation v1 runner and manifest, DAgger runner (v2/v3), demo manifests |
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
