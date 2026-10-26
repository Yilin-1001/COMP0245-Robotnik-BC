"""Check that TemplateTeacher + RobotEnv reproduce robotnik_arm_task.py's own commands.

Runs the unmodified robotnik_arm_task.main() headless with its MotorCommands class replaced by a
recorder, so the code under test is the template script itself. The same target is then run with
RobotEnv and TemplateTeacher from the template's start pose (origin, bearing 0), and the 11 wheel and
arm commands are compared at every step the template executed. The template's target is a module
constant, so other targets are tested by setting robotnik_arm_task.TARGET_EE_POS before main().

Output: <out>/template_teacher_equivalence.json

Usage: python -m data_collection.verify_template_teacher --out data/Stateful/Clean/qa
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np

TARGETS = [[0.0, 1.1, 0.5],      # the template's own TARGET_EE_POS
           [-1.6, -0.9, 0.25],   # behind the robot: long align turn
           [2.2, 0.4, 1.15]]     # high target, short standoff


def run_template(target, max_steps):
    os.environ["ROBOENV_HEADLESS"] = "1"
    os.environ["ROBOENV_MAX_STEPS"] = str(max_steps)
    import robotnik_arm_task as task
    from simulation_and_control import MotorCommands

    commands = []

    class Recorder(MotorCommands):
        def SetControlCmd(self, cmd, modes):
            commands.append(np.array(cmd[:11], dtype=float))
            return super().SetControlCmd(cmd, modes)

    task.MotorCommands = Recorder
    task.TARGET_EE_POS = np.array(target, dtype=float)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        task.main()
    last = [l for l in out.getvalue().splitlines() if l.startswith(("Target reached", "Target not reachable", "Reached step cap"))]
    return np.array(commands), last[-1] if last else ""


def run_wrapper(target, n_steps):
    from core.env import RobotEnv
    from data_collection.scenarios import Scenario
    from data_collection.template_teacher import TemplateTeacher
    env = RobotEnv()
    obs = env.reset(Scenario(seed=0, target_ee_pos=np.array(target, dtype=float)))
    teacher = TemplateTeacher(env)
    teacher.reset(q_hold=obs.q_mes)
    commands = []
    for _ in range(n_steps):
        action, _ = teacher.act(obs)
        commands.append(np.concatenate((action.wheel_cmds, action.torque_arm)))
        obs, _ = env.step(action)
    env.close()
    return np.array(commands)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-steps", type=int, default=40000)
    args = parser.parse_args()
    results = []
    for target in TARGETS:
        ref, stop_msg = run_template(target, args.max_steps)
        mine = run_wrapper(target, len(ref))
        diff = np.abs(ref - mine)
        first = int(np.argmax(diff.max(1) > 0)) if diff.max() > 0 else None
        results.append({"target": target, "steps_compared": len(ref), "template_stop": stop_msg,
                        "max_abs_diff_wheels_rad_s": float(diff[:, :4].max()), "max_abs_diff_torque_Nm": float(diff[:, 4:].max()),
                        "first_differing_step": first, "bitwise_identical": bool(diff.max() == 0)})
        print(json.dumps(results[-1]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "template_teacher_equivalence.json"), "w") as f:
        json.dump({"description": __doc__.strip().splitlines()[0], "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
