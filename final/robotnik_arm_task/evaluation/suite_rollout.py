"""Fixed-duration closed-loop protocol; independent of historical rollout-v1 scoring."""
import math
import time

import numpy as np

from core.interfaces import Action

PROTOCOL = {'id': 'continuous-hold-v1', 'tolerance_m': .01, 'terminal_window_s': 1.,
            'grade_boundaries_mm': [5., 10., 20., 50.], 'failure_overrides_success': True,
            'history_hz': 10}


def score_errors(errors, control_hz, hold_s, failure=None):
    """Score observations at t=k/hz: a 3 s streak has 3001 inclusive samples at 1 kHz."""
    err = np.asarray(errors, dtype=float)
    if err.ndim != 1 or not len(err) or control_hz <= 0 or hold_s <= 0:
        raise ValueError('nonempty 1-D errors, positive control_hz and hold_s required')
    if not np.isfinite(err).all() or np.any(err < 0):
        failure = failure or 'FloatingPointError: invalid error observations'
    good = np.isfinite(err) & (err >= 0) & (err < PROTOCOL['tolerance_m'])
    edges = np.diff(np.r_[False, good, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    needed = int(math.ceil(hold_s * control_hz))
    qualifies = np.flatnonzero(ends - starts >= needed + 1)
    completion = int(starts[qualifies[0]] + needed) if len(qualifies) else None
    finite = err[np.isfinite(err) & (err >= 0)]
    window = err[max(0, len(err) - 1 - int(round(control_hz))):]
    terminal = None if failure else float(window.max() * 1000)
    grade = 'F' if failure else 'ABCDE'[int(np.searchsorted(PROTOCOL['grade_boundaries_mm'], terminal, side='right'))]
    first = np.flatnonzero(good)
    return {'success': bool(completion is not None and not failure), 'grade': grade,
            'completion_time_s': completion / control_hz if completion is not None and not failure else None,
            'qualifying_hold_start_s': int(starts[qualifies[0]]) / control_hz if len(qualifies) and not failure else None,
            'first_arrival_s': int(first[0]) / control_hz if len(first) else None,
            'longest_hold_s': float(max(ends-starts-1, default=0) / control_hz),
            'final_err_mm': float(err[-1] * 1000) if np.isfinite(err[-1]) else None,
            'final1s_max_err_mm': terminal,
            'final1s_mean_err_mm': None if failure else float(window.mean() * 1000),
            'min_err_mm': float(finite.min() * 1000) if len(finite) else None,
            'failure': failure, 'failure_type': failure.split(':', 1)[0] if failure else None}


def _check_observation(obs):
    values = np.r_[obs.base_pos, obs.base_bearing, obs.q_mes, obs.qd_mes, obs.target_ee_pos]
    if not np.isfinite(values).all():
        raise FloatingPointError('non-finite observation')


def rollout(policy, scenario, duration_s, hold_s, interface, env_factory=None):
    """Execute every scheduled step unless an exception occurs; always close the environment."""
    if env_factory is None:
        from evaluation.evaluate import make_robot_env
        env_factory = make_robot_env
    hz = int(interface['control_hz'])
    cap = int(round(duration_s * hz))
    if cap <= 0 or not np.isclose(cap/hz, duration_s) or hold_s <= 0:
        raise ValueError('duration must be positive and aligned with the control period')
    torque_limit, wheel_limit = float(interface['torque_limit']), float(interface['wheel_limit'])
    errors, latencies = [], []
    history, prev, delta_sum, saturated, steps = [], None, 0., 0, 0
    env, failure, final_pos = None, None, None
    stride = max(1, hz // PROTOCOL['history_hz'])
    started = time.perf_counter()
    try:
        env = env_factory()
        obs = env.reset(scenario)
        if hasattr(policy, 'bind_env'):
            policy.bind_env(env)
        policy.reset()
        _check_observation(obs)
        final_pos = np.asarray(env.ee_pos(), dtype=float)
        error = float(np.linalg.norm(scenario.target_ee_pos - final_pos))
        if not np.isfinite(error):
            raise FloatingPointError('non-finite initial end-effector error')
        errors.append(error)
        history.append([0., error * 1000])
        for k in range(cap):
            start = time.perf_counter_ns()
            action = policy.act(obs)
            latencies.append((time.perf_counter_ns() - start) / 1000.)
            torque, wheels = np.asarray(action.torque_arm), np.asarray(action.wheel_cmds)
            raw = getattr(policy, 'last_raw', None)
            if torque.shape != (7,) or wheels.shape != (4,) or not np.isfinite(np.r_[torque, wheels]).all():
                raise FloatingPointError('invalid policy action')
            if raw is not None and not np.isfinite(raw).all():
                raise FloatingPointError('non-finite unclipped policy action')
            saturated += int(np.any(np.abs(raw[:7] if raw is not None else torque) >= torque_limit))
            executed = Action(np.clip(torque, -torque_limit, torque_limit), np.clip(wheels, -wheel_limit, wheel_limit))
            if prev is not None:
                delta_sum += float(np.abs(executed.torque_arm - prev).mean())
            prev = executed.torque_arm
            obs, info = env.step(executed)
            steps = k + 1
            _check_observation(obs)
            error = float(info['ee_err'])
            final_pos = np.asarray(info['ee_pos'], dtype=float)
            if not np.isfinite(error) or error < 0 or not np.isfinite(final_pos).all():
                raise FloatingPointError('invalid end-effector observation')
            errors.append(error)
            if steps % stride == 0 or steps == cap:
                history.append([steps/hz, error * 1000])
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failure = failure or f'{type(exc).__name__}: close: {exc}'
    if not errors:
        errors = [float('inf')]
    result = score_errors(errors, hz, hold_s, failure)
    if steps != cap and result['failure'] is None:
        result = score_errors(errors, hz, hold_s, 'RuntimeError: incomplete rollout')
    return {**result, 'scenario_id': scenario.scenario_id, 'steps': steps, 'duration_s': steps/hz,
            'wall_time_s': time.perf_counter()-started,
            'final_ee_pos_W': final_pos.tolist() if final_pos is not None and np.isfinite(final_pos).all() else None,
            'saturation_frac': saturated/len(latencies) if latencies else None,
            'torque_variation_nm': delta_sum/(steps-1) if steps > 1 else None,
            **{f'latency_p{q}_us': float(np.percentile(latencies,q)) if latencies else None for q in (50,95,99)},
            'history': history}
