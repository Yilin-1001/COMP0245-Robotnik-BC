"""Two independent, synchronized PyBullet windows: frozen expert versus learned policy.

Run from final/robotnik_arm_task/: python -m evaluation.live_compare
SPACE pauses both windows, R resets both, Q closes both. Physics/control remain
at 1 kHz; a barrier synchronizes the two processes every 20 simulation steps.
At the duration limit the scene freezes for inspection until R or Q is pressed.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time

TASK_DIR = Path(__file__).resolve().parents[1]
# Terminal models of the training-method ablation; the default demo is its best row, R4-dart-gate, seed 0.
DEFAULT_PROTOCOL = TASK_DIR / "experiments/demo_models/2026-10-28__training-ablation-v1.yaml"
DEFAULT_ROW = "R4-dart-gate"


def resolve_demo(args):
    from dataclasses import replace
    import numpy as np
    import yaml
    from evaluation.test_sets import get_split
    protocol_path = args.protocol or DEFAULT_PROTOCOL
    archived = "the pre-ablation models and demo manifests are archived under Archive/ (see Archive/README.md); " \
               "pass --protocol <model manifest>"
    if not protocol_path.exists():
        raise FileNotFoundError(f"{protocol_path.name} not found: {archived}")
    protocol = yaml.safe_load(protocol_path.read_text())
    # Ablation manifests list several rows per (teacher, objective, seed); older manifests have no row field.
    artifact = next(a for a in protocol["decisions"]["final"]["artifacts"]
                    if (a["teacher"], a["objective"], a["seed"]) == (args.teacher, args.objective, args.seed)
                    and a.get("row", args.row) == args.row)
    path = TASK_DIR / artifact["inference"]
    if not path.exists():
        raise FileNotFoundError(f"{artifact['inference']} not found: {archived}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["inference_sha256"]:
        raise ValueError(f"artifact hash differs from the frozen protocol: {path}")
    scenarios = get_split("validation", TASK_DIR / "experiments/splits/clean253.json", args.teacher)
    scenario = next((s for s in scenarios if s.scenario_id == args.scenario), None)
    if scenario is None:
        raise ValueError(f"{args.scenario} is not in the validation set")
    custom = []
    if args.target is not None:
        scenario = replace(scenario, target_ee_pos=np.asarray(args.target, dtype=float))
        custom.append("target")
    if args.start is not None:
        scenario = replace(scenario, base_start_pos=np.array([args.start[0], args.start[1], 0.0]))
        custom.append("start")
    if args.bearing is not None:
        scenario = replace(scenario, base_start_bearing=float(args.bearing))
        custom.append("bearing")
    if custom:
        scenario = replace(scenario, scenario_id=f"{scenario.scenario_id}/custom-{'-'.join(custom)}", split="custom")
    for warning in outside_training_range(scenario):
        print(f"WARNING: {warning}; the student never saw such scenes in training", flush=True)
    return artifact, scenario, str(path)


def outside_training_range(scenario) -> list[str]:
    """Where a (custom) scene leaves the ranges the clean and DART training scenarios were sampled from."""
    import numpy as np
    start, target = np.asarray(scenario.base_start_pos, float), np.asarray(scenario.target_ee_pos, float)
    dist = float(np.linalg.norm(target[:2] - start[:2]))
    out = []
    if not 0.5 <= dist <= 3.0:
        out.append(f"target is {dist:.2f} m from the start (training: 0.5-3.0 m)")
    if not 0.1 <= target[2] <= 1.25:
        out.append(f"target height {target[2]:.2f} m (training: 0.1-1.25 m)")
    if np.any(np.abs(start[:2]) > 0.5):
        out.append(f"start ({start[0]:.2f}, {start[1]:.2f}) m (training: x and y within +-0.5 m)")
    return out


def _worker(role, args, artifact, scenario, path, barrier, stopped, paused, restart, results):
    import numpy as np
    from core.env import RobotEnv
    from core.interfaces import Action
    from evaluation.evaluate import expert_factory
    from training.common.policies import MLPPolicy

    env = None
    steps, failure = 0, None
    initial, obs = None, None
    label = f"EXPERT {args.teacher}" if role == "expert" else \
        f"{artifact.get('row', args.objective.upper())} {args.teacher} seed {args.seed}"
    color = [0.0, 0.55, 0.15] if role == "expert" else [0.05, 0.3, 0.9]
    try:
        env = RobotEnv(gui=not args.headless)
        policy = (expert_factory(args.teacher) if role == "expert" else
                  MLPPolicy.load(path, expected={"teacher": args.teacher, "run_id": artifact["run_id"]}))
        if hasattr(policy, "bind_env"):
            policy.bind_env(env)

        def reset():
            observation = env.reset(scenario)
            policy.reset()
            if not args.headless:
                env.client.configureDebugVisualizer(env.client.COV_ENABLE_GUI, 0)
                env.client.resetDebugVisualizerCamera(2.8, 45, -25,
                                                      [*observation.base_pos[:2], .65])
            return observation

        obs = reset()
        initial = {"initial_base_pos": obs.base_pos.tolist(), "initial_arm_q": obs.q_mes.tolist()}
        generation = restart.value
        follow_camera = True
        text_ids = [-1] * 5
        line_id = -1
        error = float(np.linalg.norm(scenario.target_ee_pos - env.ee_pos()))
        torque_max = wheel_right = wheel_left = 0.0
        cap = round(args.seconds * 1000)
        print(f"[{label}] ready: {scenario.scenario_id}; target={scenario.target_ee_pos}", flush=True)
        while not stopped.is_set():
            tick = time.perf_counter()
            if not args.headless:
                if not env.client.isConnected():
                    stopped.set()
                    break
                keys = env.client.getKeyboardEvents()
                pressed = lambda key: keys.get(ord(key), 0) & env.client.KEY_WAS_TRIGGERED
                if pressed('q'):
                    stopped.set()
                    break
                if pressed(' '):
                    with paused.get_lock():
                        paused.value = not paused.value
                if pressed('r'):
                    with restart.get_lock():
                        restart.value += 1
                if pressed('f'):
                    follow_camera = not follow_camera
            barrier.wait(timeout=120)
            if stopped.is_set():
                break
            if generation != restart.value:
                obs, steps, generation = reset(), 0, restart.value
                error = float(np.linalg.norm(scenario.target_ee_pos - env.ee_pos()))
                torque_max = wheel_right = wheel_left = 0.0
            if not paused.value and steps < cap:
                for _ in range(min(20, cap - steps)):
                    action = policy.act(obs)
                    if not (np.isfinite(action.torque_arm).all() and np.isfinite(action.wheel_cmds).all()):
                        raise FloatingPointError("non-finite policy action")
                    action = Action(np.clip(action.torque_arm, -30, 30), np.clip(action.wheel_cmds, -6, 6))
                    obs, info = env.step(action)
                    steps += 1
                error = info["ee_err"]
                if not (np.isfinite(obs.q_mes).all() and np.isfinite(obs.base_pos).all() and np.isfinite(error)):
                    raise FloatingPointError("non-finite simulated state")
                torque_max = float(np.abs(action.torque_arm).max())
                wheel_right, wheel_left = action.wheel_cmds[:2]
            if not args.headless:
                state = "FINISHED - R to replay" if steps >= cap else "PAUSED" if paused.value else "RUNNING"
                texts = [f"{label} | {scenario.scenario_id}", f"t={env.t:.2f}s | {state}",
                         f"EE error={error * 1000:.1f} mm | tolerance=10 mm",
                         f"max |torque|={torque_max:.2f} Nm | wheels R/L={wheel_right:.2f}/{wheel_left:.2f}",
                         "SPACE pause | R replay | F follow camera | Q quit both"]
                for i, text in enumerate(texts):
                    text_ids[i] = env.client.addUserDebugText(
                        text, [*obs.base_pos[:2], 1.9 - i * .14], textColorRGB=color,
                        textSize=1.2, replaceItemUniqueId=text_ids[i])
                line_id = env.client.addUserDebugLine(env.ee_pos(), scenario.target_ee_pos,
                                                      [1, 0, 0], 2, replaceItemUniqueId=line_id)
                if follow_camera:
                    camera = env.client.getDebugVisualizerCamera()
                    env.client.resetDebugVisualizerCamera(camera[10], camera[8], camera[9],
                                                          [*obs.base_pos[:2], .65])
                time.sleep(max(0.0, .02 / args.speed - (time.perf_counter() - tick)))
            elif steps >= cap:
                break
    except threading.BrokenBarrierError:
        if not stopped.is_set():
            failure = "other simulation failed to synchronize within 120 seconds"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"[{label}] {failure}", flush=True)
    finally:
        stopped.set()
        barrier.abort()
        results.put({"role": role, "pid": os.getpid(), "scenario_id": scenario.scenario_id,
                     "target_ee_pos": scenario.target_ee_pos.tolist(),
                     "steps": steps, "simulation_time_s": env.t if env else 0.0,
                     "final_error_mm": error * 1000 if initial else None,
                     "final_arm_q": obs.q_mes.tolist() if obs is not None else None,
                     "failure": failure, **(initial or {})})
        if env is not None and env.client.isConnected():
            env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", choices=["A", "B"], default="A")
    p.add_argument("--row", default=DEFAULT_ROW, help="ablation row in the manifest, e.g. R1-baseline or R4-dart-gate")
    p.add_argument("--objective", choices=["bc", "bcj"], default="bcj")
    p.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    p.add_argument("--scenario", default="train-0186",
                   help="one of the 32 frozen validation scenarios (default: target 2.7 m away, behind the robot)")
    p.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="override the world-frame EE target in meters")
    p.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"),
                   help="override the base start position in meters (world frame)")
    p.add_argument("--bearing", type=float, metavar="B", help="override the base start yaw in radians (0 = facing +X)")
    p.add_argument("--seconds", type=float, default=40.0, help="simulation duration before freezing")
    p.add_argument("--speed", type=float, default=1.0, help="wall-clock playback speed, physics stays at 1 kHz")
    p.add_argument("--protocol", type=Path,
                   help=f"model manifest (default {DEFAULT_PROTOCOL.relative_to(TASK_DIR)}: teacher-A ablation models)")
    p.add_argument("--headless", action="store_true", help="run both simulators without windows, then exit")
    p.add_argument("--output", type=Path, default=TASK_DIR / "results/live_compare/comparison.json")
    args = p.parse_args(argv)
    import math
    if not (math.isfinite(args.seconds) and args.seconds >= .001 and
            math.isfinite(args.speed) and args.speed > 0):
        p.error("seconds must be finite and >= 0.001; speed must be finite and positive")
    if args.target is not None and not all(math.isfinite(v) for v in args.target):
        p.error("target coordinates must be finite")
    if (args.start is not None and not all(math.isfinite(v) for v in args.start)) or \
            (args.bearing is not None and not math.isfinite(args.bearing)):
        p.error("start coordinates and bearing must be finite")
    try:
        artifact, scenario, path = resolve_demo(args)
    except (ValueError, StopIteration, OSError) as exc:
        p.error(str(exc) or "no matching artifact in the final protocol")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    ctx = mp.get_context("spawn")
    barrier, stopped, paused = ctx.Barrier(2), ctx.Event(), ctx.Value('b', False)
    restart, results = ctx.Value('i', 0), ctx.Queue()
    processes = [ctx.Process(target=_worker, name=f"robot-{role}", args=(
        role, args, artifact, scenario, path, barrier, stopped, paused, restart, results))
                 for role in ("expert", "student")]
    print(f"Two independent simulators | teacher {args.teacher} | {artifact['label']} | {args.scenario}", flush=True)
    print("SPACE pause/resume, R replay, Q close both. Each window follows its robot.", flush=True)
    try:
        for process in processes:
            process.start()
        while any(process.is_alive() for process in processes):
            if any(process.exitcode is not None for process in processes):
                stopped.set()
                barrier.abort()
            for process in processes:
                process.join(timeout=.1)
    except KeyboardInterrupt:
        stopped.set()
        barrier.abort()
    finally:
        stopped.set()
        barrier.abort()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join()
    rows = []
    for _ in processes:
        try:
            rows.append(results.get(timeout=2))
        except queue.Empty:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"artifact": artifact, "runs": rows}, indent=2) + "\n")
    print(f"Saved summary: {args.output}", flush=True)
    return int(len(rows) != 2 or any(r["failure"] for r in rows) or any(p.exitcode for p in processes))


if __name__ == "__main__":
    raise SystemExit(main())
