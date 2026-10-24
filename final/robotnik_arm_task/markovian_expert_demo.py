"""Demo of the Markovian expert (data_collection/expert.py), laid out like robotnik_arm_task.py.

Run from final/robotnik_arm_task/:
    python markovian_expert_demo.py                                  # GUI, default target
    python markovian_expert_demo.py --target 1.5 -1.0 0.3            # another target
    python markovian_expert_demo.py --start 0.5 0.5 --bearing 2.0    # another start pose
    ROBOENV_HEADLESS=1 python markovian_expert_demo.py               # headless
"""
import argparse
import os
import time

import numpy as np

# Headless mode and step cap for CI / direct runs, as in robotnik_arm_task.py
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.env import RobotEnv
from core.interfaces import Action
from data_collection.expert import MarkovianExpert
from data_collection.scenarios import Scenario
from robotnik_arm_control import StopCriteria, check_stop


# ==============================================================================
# TARGET SPECIFICATION (defaults; override with --target / --start / --bearing)
# Same target as robotnik_arm_task.py: [X (forward/m), Y (lateral/m), Z (height/m)], world frame
TARGET_EE_POS = np.array([0.0, 1.1, 0.5])
# The template always starts at the origin facing +X; the Markovian expert works from any start pose
BASE_START_POS = np.array([0.0, 0.0, 0.0])
BASE_START_BEARING = 0.0
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
REACH_TOLERANCE = 0.01
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, nargs=3, default=TARGET_EE_POS, metavar=("X", "Y", "Z"))
    parser.add_argument("--start", type=float, nargs=2, default=BASE_START_POS[:2], metavar=("X", "Y"))
    parser.add_argument("--bearing", type=float, default=BASE_START_BEARING, help="start yaw (rad)")
    args = parser.parse_args()
    target = np.array(args.target, dtype=float)

    # --------------------------------------------------------------------------
    # Simulation setup (same physics as robotnik_arm_task.py, see core/env.py) and expert
    # --------------------------------------------------------------------------
    env = RobotEnv(gui=not headless, reach_tolerance=REACH_TOLERANCE)
    expert = MarkovianExpert()
    scenario = Scenario(seed=0, target_ee_pos=target,
                        base_start_pos=np.array([args.start[0], args.start[1], 0.0]),
                        base_start_bearing=args.bearing)
    obs = env.reset(scenario)
    if not headless:
        env.client.addUserDebugPoints(pointPositions=[target.tolist()], pointColorsRGB=[[1.0, 0.0, 0.0]], pointSize=10)

    init_ee_pos = env.ee_pos()
    print("==================================================================")
    print(f"Target Cartesian EE Position : {target}")
    print(f"Initial Base Pose            : {np.round(obs.base_pos[:2], 4)}, bearing {obs.base_bearing:.3f} rad")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(target - init_ee_pos):.4f}")
    print("==================================================================")

    time_step = 0.001
    current_time = 0.0
    # Same stop conditions as the template; "base navigation done" is the expert's reach-active flag
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)

    # Data logging for plotting
    time_all, ee_pos_all, ee_error_all, base_pos_all = [], [], [], []
    q_mes_all, reach_active_all = [], []

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    while True:
        # Base phase and arm/wheel action from the current observation only (no state kept between steps)
        wheel_cmds, reach_active = expert.base_command(obs)
        action = Action(torque_arm=expert.arm_torque(obs, reach_active), wheel_cmds=wheel_cmds)
        ee_pos = env.ee_pos()
        ee_error = target - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # Exit logic with 'q' key in PyBullet GUI window
        if not headless:
            keys = env.client.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & env.client.KEY_WAS_TRIGGERED:
                break

        # Stop criteria: maximum steps, target reached within fine tolerance, or unreachable target.
        # Success is tested first: check_stop tests the stall before success, so a target reached
        # more than stall_window before min_time would otherwise be reported as unreachable.
        if reach_active and ee_dist_error < REACH_TOLERANCE and current_time > stop_criteria.min_time:
            stop_message = f"Target reached within {REACH_TOLERANCE*1000:.0f}mm tolerance at t={current_time:.2f}s!"
        else:
            stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, reach_active)
        if stop_message:
            print(stop_message)
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(obs.base_pos.copy())
        q_mes_all.append(obs.q_mes.copy())
        reach_active_all.append(reach_active)

        obs, info = env.step(action)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(round(current_time / time_step)) % 500 == 0:
            dist_err, heading_err = expert.base_errors(obs)
            print(f"t={current_time:.2f}s | Base pos: {np.round(obs.base_pos[:2], 3)} | EE error: {info['ee_err']:.4f}m | "
                  f"base dist err {dist_err:+.3f}m heading err {heading_err:+.3f}rad | reach {reach_active}")

    # Final summary report
    final_ee = env.ee_pos()
    print("==================================================================")
    print(f"Final Base Position          : {np.round(obs.base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(final_ee, 4)}")
    print(f"Expert's EE Estimate         : {np.round(expert.ee_pos(obs), 4)}  (from the observation only)")
    print(f"Target End-Effector Position : {target}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(target - final_ee):.4f}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(env.sim.GetMotorAngles(0)[11:17] - 0.1)):.6f}")
    print("==================================================================")

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        env.close()
        return

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    time_all = np.array(time_all)
    ee_pos_all = np.array(ee_pos_all)
    ee_dist_norm = np.linalg.norm(np.array(ee_error_all), axis=1)
    base_pos_all = np.array(base_pos_all)
    q_mes_all = np.array(q_mes_all)
    reach_active_all = np.array(reach_active_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=target[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching (Markovian expert)')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time, with the arm's reach-active phase shaded
    plt.figure(figsize=(8, 4))
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.fill_between(time_all, 0, ee_dist_norm.max(), where=reach_active_all, color='orange', alpha=0.15,
                     label='Arm reaching (base within tolerance)')
    plt.axhline(y=REACH_TOLERANCE, color='r', linestyle='--', label='10 mm tolerance')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Base path (top view)
    plt.figure(figsize=(6, 6))
    plt.plot(base_pos_all[:, 0], base_pos_all[:, 1], label='Base path', color='b')
    plt.plot(*base_pos_all[0, :2], 'bo', label='Start')
    plt.plot(*target[:2], 'r*', markersize=14, label='Target (XY)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.title('Mobile Base Path')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Arm joint positions
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, q_mes_all[:, i], label=f'Measured J{i+1}')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == '__main__':
    main()
