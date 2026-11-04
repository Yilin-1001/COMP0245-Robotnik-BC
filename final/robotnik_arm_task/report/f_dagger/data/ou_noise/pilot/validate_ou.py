"""Standalone D2 pilot; does not modify controllers or construct training datasets."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = "/tmp/robotnik_ou_matplotlib"
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import numpy as np
from scipy.stats import qmc

TASK = Path('/media/yilin/data/ML_Final/COMP_0245_2026_Final/final/robotnik_arm_task')
sys.path.insert(0, str(TASK))
DT, TAU, MAX_TIME, HOLD_TIME = .001, .2, 40., 3.
OUT = None


def initialize(out):
    global OUT
    OUT = Path(out)
    stream = open(OUT / f'worker_{os.getpid()}.log', 'a', buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)


def stateful_controller(env, target):
    import pinocchio as pin
    from core.interfaces import Action
    from robotnik_arm_control import (ArmControllerParams, BaseControllerParams,
                                     BaseNavState, arm_controller, base_controller)
    dm = env.dyn_model
    shoulder = dm.pin_model.getFrameId('robot_j2s7s300_link_2')
    lower, upper = env.sim.GetBotJointsLimit()
    lower, upper = np.array(lower[4:11]), np.array(upper[4:11])
    standoff = float(np.interp(target[2], [.1, .5, 1.1, 1.25], [.80, .65, .65, .35]))
    bearing = float(np.arctan2(target[1], target[0])) if np.linalg.norm(target[:2]) > 1e-4 else 0.
    goal = target[:2] - standoff * np.array([np.cos(bearing), np.sin(bearing)])
    nav, bp, ap = BaseNavState(), BaseControllerParams(nav_timeout=25.), ArmControllerParams(reach_limit=.85)
    hold = env.sim.GetMotorAngles(0)[4:11].copy()
    state = {'q_des': hold.copy(), 'phase': 'align', 'done': False, 'timeout': False}

    def step(obs):
        full = np.concatenate((env.sim.GetBasePosition(), env.sim.GetBaseOrientation(), env.sim.GetMotorAngles(0)))
        q_pin = dm.ReoderJoints2PinVec(full, 'pos')
        pin.forwardKinematics(dm.pin_model, dm.pin_data, q_pin)
        pin.updateFramePlacements(dm.pin_model, dm.pin_data)
        ee = dm.pin_data.oMf[env.ee_frame_id].translation.copy()
        was_done = nav.done
        wheels = base_controller(nav, obs.base_pos, obs.base_bearing, goal, bearing, env.t, bp)
        if nav.done and not was_done and env.t > bp.nav_timeout:
            state['timeout'] = True
        torque, state['q_des'], qd_des = arm_controller(
            dm, q_pin, obs.q_mes, obs.qd_mes, state['q_des'], hold, target, ee,
            env.ee_frame_id, shoulder, lower, upper, DT, ap, reach_active=nav.done)
        state.update(phase=nav.phase, done=bool(nav.done))
        return Action(torque, wheels), bool(nav.done)

    return step, state


def make_scenarios(count, seed, anchors=False):
    values = qmc.Sobol(6, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(max(1, count)))))[:count]
    out = []
    for i, u in enumerate(values):
        dist, azimuth, height = .5 + 2.5*u[0], -np.pi + 2*np.pi*u[1], .1 + 1.15*u[2]
        start = np.array([u[3]-.5, u[4]-.5, 0.])
        yaw = -np.pi + 2*np.pi*u[5]
        target = start + np.array([dist*np.cos(yaw+azimuth), dist*np.sin(yaw+azimuth), height])
        out.append(dict(id=f'sobol_{seed}_{i:03d}', target=target.tolist(), start=start.tolist(), yaw=float(yaw)))
    if anchors:
        targets = [[0., 1.1, .5], [1.5, 1., .6], [-1.2, 1.8, .3], [.4, -2.2, 1.1]]
        out = [dict(id=f'anchor_{i}', target=t, start=[0., 0., 0.], yaw=0.) for i,t in enumerate(targets)] + out
    return out


def run(job):
    from core.env import RobotEnv
    from core.interfaces import Action
    from data_collection.expert import MarkovianExpert, MarkovianExpertParams
    from data_collection.scenarios import Scenario
    from robotnik_arm_control import BaseControllerParams, StopCriteria, check_stop
    teacher, lead, sigma_t, sigma_w, noise_seed, s = job
    name = f'{teacher}_lead{lead:g}_T{sigma_t:g}_W{sigma_w:g}_n{noise_seed}_{s["id"]}'
    start_wall = time.monotonic()
    env = RobotEnv(control_hz=1000, reach_tolerance=.01)
    try:
        target = np.array(s['target'])
        obs = env.reset(Scenario(seed=noise_seed, target_ee_pos=target,
                                base_start_pos=np.array(s['start']), base_start_bearing=s['yaw'],
                                physics={'mass_scale': 1., 'friction_scale': 1., 'sensor_noise_scale': 0.}))
        assert env.noise_std == 0 and env.push is None
        if teacher == 'Stateful':
            controller, teacher_state = stateful_controller(env, target)
        else:
            expert = MarkovianExpert(MarkovianExpertParams(lead_time=lead))
            teacher_state = {'timeout': False, 'phase': 'reactive'}
            def controller(obs):
                wheels, active = expert.base_command(obs)
                return Action(expert.arm_torque(obs, active), wheels), bool(active)
        bp = BaseControllerParams(nav_timeout=25.)
        wheel_limit = (bp.max_lin_vel + bp.max_ang_vel*bp.wheel_base_width/2) / bp.wheel_radius
        rng = np.random.default_rng(noise_seed)
        sig = np.r_[np.full(7, sigma_t), np.full(4, sigma_w)]
        rho, noise = np.exp(-DT/TAU), np.zeros(11)
        traces, errors, reached, gate_t = [], [], None, None
        first_stop, original_stop_reason, settled_t = None, None, None
        criteria = StopCriteria(reach_tolerance=.01, max_steps=40000)
        torque_clips = wheel_clips = torque_on_steps = wheel_on_steps = 0
        torque_energy = wheel_energy = 0.
        base_at_gate, max_base_drift = None, 0.
        err = float(np.linalg.norm(env.ee_pos()-target))
        for k in range(40000):
            action, active = controller(obs)
            if active and gate_t is None:
                gate_t, base_at_gate = env.t, obs.base_pos[:2].copy()
            msg = check_stop(criteria, env.t, DT, err, active)
            if msg and first_stop is None:
                first_stop = env.t
                original_stop_reason = 'reached' if msg.startswith('Target reached') else 'stalled_or_cap'
            if active and env.t > 8. and err < .01 and reached is None:
                reached = env.t
            noise = rho*noise + np.sqrt(1-rho*rho)*rng.normal(0., sig)
            applied = noise.copy()
            if gate_t is not None:
                applied[7:] = 0.
            if reached is not None and env.t >= reached+1.5-1e-9:
                applied[:] = 0.
                if settled_t is None:
                    settled_t = env.t
            if sigma_t and settled_t is None:
                torque_on_steps += 1
                torque_energy += float(np.sum(applied[:7]**2))
            if sigma_w and gate_t is None:
                wheel_on_steps += 1
                wheel_energy += float(np.sum(applied[7:]**2))
            raw_torque, raw_wheels = action.torque_arm+applied[:7], action.wheel_cmds+applied[7:]
            torque = np.clip(raw_torque, -30., 30.)
            wheels = np.clip(raw_wheels, -wheel_limit, wheel_limit)
            torque_clips += int(np.any(torque != raw_torque))
            wheel_clips += int(np.any(wheels != raw_wheels))
            assert np.isfinite(torque).all() and np.isfinite(wheels).all()
            obs, info = env.step(Action(torque, wheels))
            err = info['ee_err']
            errors.append(err)
            if base_at_gate is not None:
                max_base_drift = max(max_base_drift, float(np.linalg.norm(obs.base_pos[:2]-base_at_gate)))
            if k % 10 == 0:
                traces.append([env.t, err, *obs.base_pos, float(active), float(gate_t is not None),
                               float(settled_t is not None), *applied])
            if reached is not None and env.t >= reached+HOLD_TIME-1e-9:
                break
        last = np.array(errors[-1000:])
        row = dict(name=name, teacher=teacher, lead_time=lead, sigma_t=sigma_t, sigma_w=sigma_w,
                   noise_seed=noise_seed, scenario_id=s['id'], **{f'target_{axis}': float(target[i]) for i,axis in enumerate('xyz')},
                   start_x=s['start'][0], start_y=s['start'][1], start_yaw=s['yaw'],
                   reached=reached is not None, first_reach_s=reached, wheel_gate_s=gate_t,
                   noise_off_s=settled_t, end_s=env.t,
                   last1s_max_mm=float(last.max()*1000), last1s_mean_mm=float(last.mean()*1000),
                   last1s_within10mm_fraction=float(np.mean(last<.01)), final_error_mm=err*1000,
                   held_last1s=bool(reached is not None and len(last)==1000 and last.max()<.01),
                   original_stop_reason=original_stop_reason, original_stop_s=first_stop,
                   original_stop_reached=original_stop_reason=='reached',
                   nav_timeout=bool(teacher_state['timeout']),
                   max_postgate_base_drift_mm=max_base_drift*1000,
                   torque_clip_fraction=torque_clips/(k+1), wheel_clip_fraction=wheel_clips/(k+1),
                   applied_torque_rms=float(np.sqrt(torque_energy/(7*torque_on_steps))) if torque_on_steps else 0.,
                   applied_wheel_rms=float(np.sqrt(wheel_energy/(4*wheel_on_steps))) if wheel_on_steps else 0.,
                   wall_s=time.monotonic()-start_wall)
        np.savez_compressed(OUT / 'traces' / f'{name}.npz', trace=np.array(traces, dtype=np.float32))
        return row
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--seed', type=int, default=427601)
    ap.add_argument('--anchors', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--leads', type=float, nargs='+', default=[.15,.5])
    ap.add_argument('--noise-scales', type=float, nargs='+', default=[0.,1.])
    ap.add_argument('--noise-repeats', type=int, default=1)
    ap.add_argument('--alpha-random', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'traces').mkdir(exist_ok=True)
    scenarios = make_scenarios(args.n, args.seed, args.anchors)
    jobs = []
    for ti,(teacher,lead) in enumerate([('Stateful',0.)]+[('Markovian',lead) for lead in args.leads]):
        for i,s in enumerate(scenarios):
            for scale in args.noise_scales:
                for rep in range(1 if scale==0 else args.noise_repeats):
                    seed = args.seed*1000+i*10+rep
                    alpha = float(np.random.default_rng(seed+917).uniform(.25,1.)) if args.alpha_random and scale else 1.
                    jobs.append((teacher,lead,1.5*scale*alpha,.3*scale*alpha,seed,s))
    source_names = ['core/env.py','data_collection/expert.py','robotnik_arm_control.py','robotnik_arm_task.py','configs/robotnik_arm.json']
    meta = dict(arguments=vars(args), scenarios=scenarios,
                protocol=dict(dt=DT,tau=TAU,max_time=MAX_TIME,hold_time=HOLD_TIME,tail_noise_off_after=1.5,
                              reach_tolerance=.01,min_reach_time=8.,sensor_noise=0.,mass_scale=1.,friction_scale=1.,push=None,
                              source_stop_recorded_not_enforced=True,trace_stride=10,
                              stationary_sigma=True,ou_initial_state=0,channels_independent=True),
                source_sha256={name:hashlib.sha256((TASK/name).read_bytes()).hexdigest() for name in source_names})
    (out/'manifest.json').write_text(json.dumps(meta,indent=2))
    began=time.monotonic()
    rows=[]
    with mp.get_context('spawn').Pool(args.workers, initializer=initialize, initargs=(str(out),)) as pool:
        with (out/'results.jsonl').open('w') as stream:
            for row in pool.imap_unordered(run,jobs):
                rows.append(row)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(f'[{len(rows)}/{len(jobs)}] {row["teacher"]} lead={row["lead_time"]} '
                      f'sigma={row["sigma_t"]:.3f}/{row["sigma_w"]:.3f} {row["scenario_id"]} '
                      f'reach={row["reached"]} hold={row["held_last1s"]} '
                      f'last1s_max={row["last1s_max_mm"]:.2f}mm',flush=True)
    rows.sort(key=lambda r:(r['teacher'],r['lead_time'],r['sigma_t'],r['scenario_id'],r['noise_seed']))
    with (out/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'COMPLETED {len(rows)} episodes in {time.monotonic()-began:.1f}s',flush=True)


if __name__=='__main__':
    main()
