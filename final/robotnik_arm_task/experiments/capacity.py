"""Single manifest-defined capacity fit through the existing BC engine, with matched probes."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from training.common.config import TASK_DIR, finalise_identity, resolve_config, run_dir_for


def experiment_config(manifest: dict) -> dict:
    hidden = manifest['hidden']
    if not isinstance(hidden, list) or not hidden or any(type(w) is not int or w <= 0 for w in hidden):
        raise ValueError('hidden must be a nonempty list of positive integer widths')
    architecture = manifest['architecture']
    if not re.fullmatch(r'd\d{2}w\d{4}', architecture) or architecture != f'd{len(hidden):02d}w{hidden[0]:04d}' or len(set(hidden)) != 1:
        raise ValueError('architecture ID must match the equal-width hidden layers')
    # Register only in this run's configuration; preserve the historical M1–M6 templates and matrix.
    config = resolve_config(TASK_DIR/'experiments/configs/baseline.yaml', manifest['teacher'], manifest['overrides'])
    config['model']['architectures'][architecture] = hidden
    config['identity']['architecture'] = architecture
    return config


def probe_indices(cache, rows_per_episode: int = 64) -> list:
    return [np.linspace(lo, hi-1, min(rows_per_episode, hi-lo), dtype=np.int64)
            for lo, hi in zip(cache.offsets[:-1], cache.offsets[1:])]


def probe_metrics(model, cache, normaliser, arm_weight: float, wheel_weight: float) -> dict:
    """Per-episode means, then equal episode averaging; both splits have the same units and reduction."""
    indices = probe_indices(cache)
    rows = np.concatenate(indices)
    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(normaliser.transform_x(cache.x[rows]), dtype=torch.float32, device=device)
            pred = torch.cat([model(chunk).cpu() for chunk in x.split(2048)]).numpy().astype(np.float64)
    finally:
        model.train(was_training)
    error_std = pred-normaliser.transform_y(cache.y[rows])
    error = normaliser.inverse_y(pred)-cache.y[rows]
    a = np.abs(error_std)
    huber = np.where(a <= 1, .5*a*a, a-.5)
    episodes = np.repeat(np.arange(len(indices)), [len(i) for i in indices])

    def reduce(mask):
        per_episode = []
        for e in range(len(indices)):
            chosen = mask & (episodes == e)
            if chosen.any():
                per_episode.append([np.mean(error[chosen, :7]**2), np.mean(error[chosen, 7:]**2),
                                    np.mean(huber[chosen, :7]), np.mean(huber[chosen, 7:])])
        if not per_episode:
            return None
        arm_mse, wheel_mse, arm, wheel = np.mean(per_episode, axis=0)
        return {'arm_rmse_nm': float(np.sqrt(arm_mse)), 'wheel_rmse_rad_s': float(np.sqrt(wheel_mse)),
                'arm_huber': float(arm), 'wheel_huber': float(wheel),
                'action_huber': float(arm_weight*arm+wheel_weight*wheel), 'episodes': len(per_episode)}

    phase = cache.diagnostics.get('phase')
    return {'sampled_rows': len(rows), 'overall': reduce(np.ones(len(rows), dtype=bool)),
            'phase': {name: reduce(phase[rows] == i) if phase is not None else None
                      for i, name in enumerate(('navigation', 'approach', 'hold'))}}


def fitting_diagnostic_ids(rows: list, fitting_ids: list) -> list:
    allowed = set(fitting_ids)
    cells = {}
    for r in rows:
        if r['scenario_id'] not in allowed:
            continue
        cell = tuple(min(int(float(r[k])*n), n-1) for k, n in
                     [('u_target_dist', 2), ('u_target_z', 4), ('u_rel_azimuth', 4)])
        cells.setdefault(cell, []).append(r['scenario_id'])
    selected = []
    for cell in itertools.product(range(2), range(4), range(4)):
        if not cells.get(cell):
            raise ValueError(f'empty fitting cell {cell}')
        selected.append(min(cells[cell]))
    return selected


def prepare(manifest_path: Path, workers: int = 16) -> dict:
    from training.common.engine import prepare_run
    manifest = yaml.safe_load(manifest_path.read_text())
    prep = prepare_run(experiment_config(manifest), workers=workers, finalise=False)
    probes = {name: {'episode_ids': cache.episode_ids,
                     'indices': [i.tolist() for i in probe_indices(cache)]}
              for name, cache in [('fitting', prep['train']), ('validation', prep['validation'])]}
    with open(TASK_DIR/prep['config']['data']['primary_root']/'scenarios_train.csv') as f:
        fitting_ids = fitting_diagnostic_ids(list(csv.DictReader(f)), prep['train'].episode_ids)
    binding = {'probes': probes, 'fitting_diagnostic_ids': fitting_ids}
    digest = lambda data: hashlib.sha256(data).hexdigest()
    prep['provenance'].update(capacity_manifest_sha256=digest(manifest_path.read_bytes()),
                              capacity_driver_sha256=digest(Path(__file__).read_bytes()),
                              capacity_probe_sha256=digest(json.dumps(binding, sort_keys=True).encode()))
    prep['config'] = finalise_identity(prep['config'], prep['provenance'])
    return dict(prep, capacity_binding=binding)


def probe_recorder(prep: dict, metrics: Path):
    recorded = {json.loads(line)['update'] for line in metrics.read_text().splitlines()} if metrics.exists() else set()
    arm, wheel = prep['config']['loss']['arm_weight'], prep['config']['loss']['wheel_weight']

    def on_checkpoint(step, checkpoint, export, model):
        if step in recorded:
            return
        row = {'update': step, 'fitting': probe_metrics(model, prep['train'], prep['normaliser'], arm, wheel),
               'validation': probe_metrics(model, prep['validation'], prep['normaliser'], arm, wheel)}
        with metrics.open('a') as f:
            f.write(json.dumps(row)+'\n')
        recorded.add(step)
        print(json.dumps({'checkpoint': step, 'probe': row}), flush=True)

    return on_checkpoint


def main(argv=None):
    from training.baseline.train import objective_from, summary
    from training.common.config import configure_runtime, pin_rollout_threads
    from training.common.engine import fit, init_model
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    pin_rollout_threads(); torch.set_num_threads(1)
    prep = prepare(args.manifest)
    config = prep['config']
    directory = run_dir_for(config)
    output = TASK_DIR/config['outputs']['results_root']
    if args.dry_run:
        print(json.dumps(summary(prep), indent=2))
        return 0
    runtime = configure_runtime('cuda')
    output.mkdir(parents=True, exist_ok=True)
    # Same initialisation and objective as the real run; throw away these independent preflight updates.
    model = init_model(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    x = torch.as_tensor(prep['normaliser'].transform_x(prep['train'].x[:1024]), dtype=torch.float32, device='cuda')
    y = torch.as_tensor(prep['normaliser'].transform_y(prep['train'].y[:1024]), dtype=torch.float32, device='cuda')
    objective = objective_from(config)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, x, y, prep['normaliser'], None).total
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite preflight loss')
        loss.backward(); optimizer.step()
    torch.cuda.synchronize()
    preflight = {'runtime': runtime, 'updates': 200, 'elapsed_s': time.perf_counter()-start,
                 'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'parameter_count': sum(p.numel() for p in model.parameters())}
    (output/'preflight.json').write_text(json.dumps(preflight, indent=2))
    print(json.dumps({'preflight': preflight}), flush=True)
    del model, optimizer, loss, x, y
    torch.cuda.empty_cache()
    (output/'probe_indices.json').write_text(json.dumps(prep['capacity_binding']))
    contract = {**summary(prep), 'manifest': str(args.manifest), 'config': config}
    (output/'run_contract.json').write_text(json.dumps(contract, indent=2))
    print(json.dumps({'training': summary(prep)}), flush=True)
    metrics = output/'probe_metrics.jsonl'
    result = fit(config, prep['train'], prep['validation'], init_model(config), prep['normaliser'], objective,
                 on_checkpoint=probe_recorder(prep, metrics), resume=directory/'latest.pt' if args.resume else None)
    (output/'training_complete.json').write_text(json.dumps({'status': result.status, 'terminal_update': result.terminal_update,
        'run_id': config['identity']['run_id'], 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}, indent=2))
    print(json.dumps({'completed': result.status, 'step': result.terminal_update}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
