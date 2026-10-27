# Making the Markovian expert's model accurate

The Markovian expert (`final/robotnik_arm_task/data_collection/expert.py`) first stopped 34–67 mm short of its targets. The template controller reached the same targets to within about 1 mm. This note explains the cause, the two fixes and how each was checked. With both fixes, the expert's final error is 0.2–3.2 mm, against a 10 mm tolerance.

## Why model errors matter more for the Markovian expert

Both controllers use the same arm torque law:

```
torque = kp·(q_des − q) + kd·(qd_des − qd) + predicted_gravity
```

If `predicted_gravity` is wrong, the PD terms have to supply the missing torque.

- **Template:** `q_des` is added up step by step. While the hand is off target, `q_des` keeps moving, so the arm can only come to rest at the target. The offset that builds up in `q_des` ends up holding the missing torque. This is integral action, and it hides model errors.
- **Markovian expert:** `q_des = q + qd_des·0.15 s` is rebuilt every step and nothing adds up. The missing torque can only come from a nonzero hand error, so the arm stops short of the target.

A Markovian controller has no memory to absorb a model error, so the model has to be right.

## Fix 1: add the missing 1 kg end-effector mass

**Finding.** The URDF gives the `robot_j2s7s300_end_effector` link no mass. PyBullet replaces a missing mass with 1 kg at the link origin (it prints "No inertial data for link, using mass=1" at load). Pinocchio, which computes `predicted_gravity`, keeps the link massless. The simulated arm therefore carries 1 kg that the expert's model leaves out: up to 7.6 Nm of missing torque on one joint.

A rough estimate of the effect: 7.6 Nm divided by the expert's velocity gain (kp·0.15 + kd = 45) gives 0.17 rad/s. On a ~0.5 m arm with Cartesian gain 1.5, that needs a hand error of about 55 mm. This matches the 34–67 mm the expert stopped at.

**Fix.** The expert adds the same 1 kg body to its own Pinocchio model, at the end-effector frame on joint 7 (`MarkovianExpert.__init__`). The URDF and the template are unchanged.

**Check: gravity-only hold test.** The arm was set to 6 poses. For 0.3 s each, it received only the model's gravity torque, with no PD feedback. If the model is right, the arm stays where it is.

| Gravity model | Largest joint sag | End-effector drift (median / max) |
|---|---|---|
| URDF only | 2.03 rad | 120.5 mm / 155.3 mm |
| With the 1 kg body | 0.032 rad | 0.6 mm / 4.0 mm |

This test checks the model without the controller's help, so a good reaching result can't hide a model error.

## Fix 2: calibrate the base pitch

**Finding.** The expert computes kinematics and gravity from the Observation, which has the base yaw but not its roll or pitch. Assuming a level base made the expert's end-effector estimate 1.4–4.3 mm off the true position. Measured over 20 episodes, the base sits about 0.0022 rad nose-up on flat ground, with roll close to 0.

**Fix.** The expert uses a fixed pitch of 0.0022 rad (`MarkovianExpertParams.base_pitch`).

**Check.** The pitch was measured on targets 0–19. It was then tested on 40 new random targets (20–59) and 12 targets at the corners of the sampling ranges, so it was not tuned on the targets it was scored on.

## Results

| Expert model | Final error |
|---|---|
| Original URDF | 34–67 mm, failed 2 of 3 targets |
| + 1 kg end-effector body | 2–6 mm, 20 of 20 targets |
| + base pitch | 0.2–3.2 mm, 52 of 52 targets |

Each target was held for 3 s after the end-effector first came within 10 mm. The expert also reached every target with action noise of up to 3 Nm (torque) and 1 rad/s (wheels).

## What is left

- **Remaining 1–3 mm.** The real pitch varies a little around 0.0022 rad, and a fixed value cannot follow that. On the same targets, 2 s after reaching, the template's median error is 0.6 mm and the expert's is 1.2 mm.
- **Changed physics.** Test cases that change masses or friction (T3) create a new model error that the expert cannot absorb. Expect it to stop short of the target in those cases, and use the hold test to measure by how much.
- **Debugging rule.** If the expert stops short of a target with nothing moving, check the model before the gains. Raising gains shrinks the error but never removes it, and brings the 30 Nm torque limit and 1 kHz stability closer.
