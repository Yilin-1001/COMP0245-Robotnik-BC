"""Package the report's terminal models under canonical names, without training.

Run from the task directory: python -m tools.export_final_checkpoints --date 2026-11-04
Existing destinations are refused. Source checkpoints are read only.
"""
import argparse
import copy
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from core.model_names import MODEL_NAMES, canonical_id, report_label, row_id
from training.common.checkpoint import load_checkpoint, save_checkpoint, sha256_file
from training.common.config import canonical_json, scientific_view
from training.common.policies import MLPPolicy

TASK = Path(__file__).resolve().parents[1]


def report_sources(task):
    """24 hash-pinned final R1–R8 models, then report round histories and full DART α=6."""
    protocol = yaml.safe_load((task / 'experiments/demo_models/training-ablation-v1-v2.yaml').read_text())
    selected = protocol['decisions']['final']['artifacts']
    if len(selected) != 24:
        raise ValueError('expected the 24 report-tested R1–R8 models')
    sources = {}
    for entry in selected:
        source = task / entry['inference']
        if sha256_file(source) != entry['inference_sha256']:
            raise ValueError(f'changed report model: {source}')
        sources[source.parent] = ('final', source)
    root = task / 'checkpoints/ablation'
    for date in ('2026-11-01', '2026-11-03', '2026-11-04'):
        for directory in sorted((root / f'{date}__training-ablation-v2').glob('*')):
            stem = directory.name
            row = row_id(stem)
            if row in ('R5', 'R6', 'R7', 'R8') and date != '2026-11-04':
                sources.setdefault(directory, ('round_history', directory / f'{stem}.npz'))
            elif row == 'R9' and stem.startswith('R9-dart-full-a6-'):
                sources[directory] = ('final' if '-r3__' in stem else 'round_history', directory / f'{stem}.npz')
    counts = {group: sum(g == group for g, _ in sources.values()) for group in ('final', 'round_history')}
    if counts != {'final': 27, 'round_history': 22}:
        raise ValueError(f'incomplete report checkpoint inventory: {counts}')
    return sorted(sources.values(), key=lambda item: item[1].as_posix())


def export_one(source, output, group, task, date):
    source_pt = source.parent / 'step_42000.pt'
    payload = load_checkpoint(source_pt)
    old_meta = copy.deepcopy(payload['meta'])
    old_run = old_meta['run_id']
    row = row_id(old_run)
    name, model_id, _ = MODEL_NAMES[row]
    run = canonical_id(old_run)
    seed = payload['config']['identity']['seed']
    variant = run.split('__')[0]
    rnd = int(variant.rsplit('-r', 1)[1]) if '-r' in variant else 0
    directory = output / ('models' if group == 'final' else 'round_history') / model_id
    if group != 'final':
        directory /= f'round-{rnd}'
    directory /= f'seed-{seed}'
    directory.mkdir(parents=True, exist_ok=False)
    target_pt, target_npz = directory / f'{run}.pt', directory / f'{run}.npz'
    release = {'report_name': name, 'model_id': model_id, 'run_id': run, 'legacy_run_id': old_run,
               'row': row, 'seed': seed, 'round': rnd, 'date': date, 'group': group,
               'source_checkpoint': str(source_pt.relative_to(task)), 'source_checkpoint_sha256': sha256_file(source_pt),
               'source_inference': str(source.relative_to(task)), 'source_inference_sha256': sha256_file(source),
               'source_config_hash': payload['config']['identity']['config_hash'],
               'purpose': 'Inference release; original training configuration is source_config.yaml.'}
    # Keep the training record separately; the released config and both loaders agree on the new identity.
    shutil.copy2(source.parent / 'config.yaml', directory / 'source_config.yaml')
    config = copy.deepcopy(payload['config'])
    config['variant'] = variant
    config['identity']['run_id'] = run
    config['release'] = release
    config_hash = hashlib.sha256(canonical_json(scientific_view(config)).encode()).hexdigest()
    config['identity'].update(config_hash=config_hash, config_id=config_hash[:12])
    payload['config'] = config
    payload['meta'].update(run_id=run, report_name=name, model_id=model_id, legacy_run_id=old_run,
                           release=release)
    save_checkpoint(target_pt, payload)
    (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for filename in ('normaliser.npz', 'provenance.json', 'metrics.csv'):
        shutil.copy2(source.parent / filename, directory / filename)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
        meta = json.loads(str(archive['metadata_json']))
    if meta['run_id'] != old_run or meta['source_step'] != 42000 or payload['step'] != 42000:
        raise ValueError(f'wrong source identity/step: {source}')
    if meta['source_sha256'] != sha256_file(source_pt):
        raise ValueError(f'NumPy export is not from its terminal checkpoint: {source}')
    meta.update(run_id=run, report_name=name, model_id=model_id, legacy_run_id=old_run, release=release,
                source_checkpoint=target_pt.name, source_sha256=sha256_file(target_pt))
    np.savez(target_npz, metadata_json=np.array(json.dumps(meta)), **arrays)
    result = {**release, 'checkpoint': str(target_pt.relative_to(output)),
              'inference': str(target_npz.relative_to(output)), 'input_dim': meta['input_dim'],
              'teacher': meta['teacher'], 'objective': meta['objective'],
              'feature_definition': meta['feature_definition'], 'step': 42000,
              'checkpoint_sha256': sha256_file(target_pt), 'inference_sha256': sha256_file(target_npz)}
    result['verification'] = verify_one(source, source_pt, target_npz, target_pt)
    return result


def verify_one(source_npz, source_pt, target_npz, target_pt):
    """Exact comparisons catch metadata edits accidentally changing a weight or a scale."""
    original, released = load_checkpoint(source_pt), load_checkpoint(target_pt)
    if original['model_state'].keys() != released['model_state'].keys():
        raise ValueError('checkpoint tensor names changed')
    for key, value in original['model_state'].items():
        if not torch.equal(value, released['model_state'][key]):
            raise ValueError(f'checkpoint tensor changed: {key}')
    if original['normaliser'] != released['normaliser']:
        raise ValueError('checkpoint normaliser changed')
    with np.load(source_npz) as before, np.load(target_npz) as after:
        if before.files != after.files:
            raise ValueError('inference array names/order changed')
        for key in before.files:
            if key != 'metadata_json':
                np.testing.assert_array_equal(before[key], after[key])
    errors = {}
    for backend, before, after in [('numpy', source_npz, target_npz), ('torch', source_pt, target_pt)]:
        first = MLPPolicy.load(str(before), backend=backend)
        second = MLPPolicy.load(str(after), backend=backend)
        rng = np.random.default_rng(20261104)
        x = first.x_mean + first.x_scale * rng.normal(size=(128, len(first.x_mean)))
        expected, actual = first.predict(x), second.predict(x)
        np.testing.assert_array_equal(expected, actual)
        if not np.isfinite(actual).all():
            raise ValueError('non-finite verification prediction')
        errors[backend + '_max_abs_prediction_difference'] = float(np.max(np.abs(expected - actual)))
    return {'weights_and_normaliser_exact': True, 'probe_rows_per_backend': 128, **errors}


def export_final(task, output, date):
    sources = report_sources(task)  # Validate inventory before creating a destination.
    output.mkdir(parents=True, exist_ok=False)
    records = [export_one(source, output, group, task, date) for group, source in sources]
    records.sort(key=lambda r: (r['row'], r['round'], r['seed']))
    manifest = {'date': date, 'n_final_models': 27, 'n_round_history_models': 22,
                'naming': {row: {'report_name': v[0], 'model_id': v[1], 'legacy_row_id': v[2]}
                           for row, v in MODEL_NAMES.items()}, 'models': records}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (output / 'model_index.csv').open('w', newline='') as handle:
        columns = ['row', 'report_name', 'model_id', 'seed', 'round', 'group', 'run_id', 'legacy_run_id',
                   'input_dim', 'step', 'checkpoint', 'inference', 'source_inference', 'inference_sha256']
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    artifacts = [{'label': f"{report_label(r['model_id'])} | round {r['round']} | seed {r['seed']}",
                  'row': r['model_id'], 'teacher': r['teacher'], 'objective': r['objective'], 'seed': r['seed'],
                  'run_id': r['run_id'], 'inference': str((output / r['inference']).relative_to(task)),
                  'inference_sha256': r['inference_sha256']}
                 for r in records if r['group'] == 'final']
    (output / 'demo_models.yaml').write_text(yaml.safe_dump(
        {'kind': 'demo-models', 'experiment': output.name, 'decisions': {'final': {'artifacts': artifacts}}}, sort_keys=False))
    lines = ['# Final report checkpoints', '', f'Date: {date}. 27 final models + 22 round-history models.', '',
             '| Row | Report name | Directory |', '|---|---|---|']
    lines += [f'| {row} | {v[0]} | `models/{v[1]}/` |' for row, v in MODEL_NAMES.items()]
    lines += ['', 'Each final row has seeds 0, 1, 2. R1–R4 use round 0; R5–R9 use round 3.',
              'R9 is full DART with alpha=6. The alpha=1 exploratory run is excluded.',
              '`round_history/` contains the other report training runs, with their original available seeds.',
              'Gate means the 22nd reach_gate input; rows without Gate have 21 inputs.',
              'OU and f-DAgger use fixed OU noise. Only R9 is named DART.', '',
              'Every model directory contains a .pt checkpoint, .npz NumPy export, normaliser, metrics,',
              'released config, original source_config.yaml, and source provenance. Use .npz for NumPy inference',
              'or .pt for PyTorch inference. These are terminal step-42000 inference releases, not resume states.', '',
              '`manifest.json` and `model_index.csv` map canonical names to source run IDs and SHA-256 hashes.',
              'NPZ source_checkpoint is relative to its own directory. Original paths inside source records',
              'are provenance and are not needed for inference. No symlinks or external weights are required.',
              'All network tensors, exported arrays and normalisers were checked for exact equality;',
              '128 input probes per backend/model produced exactly equal predictions before/after naming.', '',
              'From final/robotnik_arm_task:', '```bash',
              f'python -m evaluation.live_compare --protocol checkpoints/{output.name}/demo_models.yaml --row R8-f-DAgger-Gate --seed 0',
              '```', 'The demo protocol paths are relative to the task directory. If relocating this folder,',
              'load the chosen model directly with training.common.policies.MLPPolicy.load(path).']
    (output / 'README.md').write_text('\n'.join(lines) + '\n')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'SHA256SUMS').write_text(''.join(f'{sha256_file(p)}  {p.relative_to(output).as_posix()}\n' for p in files))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, type=datetime.date.fromisoformat)
    args = parser.parse_args()
    date = args.date.isoformat()
    output = TASK / 'checkpoints' / f'{date}final'
    result = export_final(TASK, output, date)
    print(json.dumps({'output': str(output), 'final': result['n_final_models'],
                      'round_history': result['n_round_history_models'], 'verified': len(result['models'])}, indent=2))


if __name__ == '__main__':
    main()
