"""Build paired Clean_253/Clean_223 data artifacts without modifying source data.

Run after data_collection.collect and data_collection.qa_clean have produced <data>/<Teacher>/Clean/:
    python -m data_collection.build_clean_subsets
Refuses to overwrite existing Clean_253/Clean_223 folders (docs/reports/2026-10-26__report__paired-clean-dataset-subsets.md).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from copy import deepcopy
import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from data_collection.scenarios import SETTINGS, TASK_DIR

DATA = Path(TASK_DIR) / SETTINGS['paths']['data']
TEACHERS = ('Markovian', 'Stateful')
SCRIPT = Path(__file__).resolve()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def has_chatter(row):
    return 'hold_chatter' in row['label_quality_flags']


def coverage(rows):
    names = ['u_target_dist', 'u_rel_azimuth', 'u_target_z', 'u_start_x', 'u_start_y', 'u_start_yaw']
    u = np.array([[float(r[n]) for n in names] for r in rows])
    # 2 distance x 2 height x 8 direction = 32 predeclared parameter cells.
    cells = np.minimum((u[:, [0, 2, 1]] * np.array([2, 2, 8])).astype(int), [1, 1, 7])
    count = Counter(tuple(x) for x in cells)
    all_cells = [(d, h, a) for d in range(2) for h in range(2) for a in range(8)]
    return {
        'n': len(rows),
        'marginal_8bin_counts': {n: np.histogram(u[:, i], bins=np.linspace(0, 1, 9))[0].tolist() for i, n in enumerate(names)},
        'distance2_height2_direction8': {
            'counts': {f'{d},{h},{a}': count[(d, h, a)] for d, h, a in all_cells},
            'empty_cells': [list(x) for x in all_cells if count[x] == 0],
            'interpretation': 'Parameter coverage only; not a guarantee of state-space or boundary coverage.'},
    }


def copy_verified(job):
    source, target, expected = job
    shutil.copy2(source, target)
    assert not source.samefile(target), f'Copy aliases source: {target}'
    actual = sha(target)
    assert actual == expected, f'Copy checksum mismatch: {target}'
    return target, actual


def main():
    src = {}
    for teacher in TEACHERS:
        folder = DATA / teacher / 'Clean'
        fields, rows = read_csv(folder / 'training_manifest.csv')
        ep_fields, episodes = read_csv(folder / 'episodes.csv')
        scenario_fields, train = read_csv(folder / 'scenarios_train.csv')
        test_fields, test = read_csv(folder / 'scenarios_test.csv')
        assert len(rows) == len(episodes) == 320
        assert len(train) == 256 and len(test) == 64
        by_id = {r['scenario_id']: r for r in rows}
        assert len(by_id) == 320
        src[teacher] = dict(folder=folder, fields=fields, rows=rows, by_id=by_id,
                            ep_fields=ep_fields, episodes=episodes, scenario_fields=scenario_fields,
                            train=train, test_fields=test_fields, test=test,
                            manifest=json.loads((folder / 'manifest.json').read_text()))
    for split in ('train', 'test'):
        assert src['Markovian'][split] == src['Stateful'][split]
    train_ids = [r['scenario_id'] for r in src['Markovian']['train']]
    test_ids = [r['scenario_id'] for r in src['Markovian']['test']]
    assert not set(train_ids) & set(test_ids)
    common = {s for s in train_ids if all(src[t]['by_id'][s]['eligible_initial_bc'] == 'True' for t in TEACHERS)}
    quiet = {s for s in common if all(not has_chatter(src[t]['by_id'][s]) for t in TEACHERS)}
    assert len(common) == 253 and len(quiet) == 223
    assert set(train_ids) - common == {'train-0064', 'train-0131', 'train-0183'}
    selections = {253: common, 223: quiet}
    targets = [(DATA / t / f'Clean_{n}') for n in selections for t in TEACHERS]
    for target in targets:
        assert not target.exists(), f'Refusing to overwrite: {target}'
        assert not target.with_name('.' + target.name + '.building').exists(), f'Existing build: {target}'

    # Snapshot all source files, including QA and excluded episodes. Compare again at completion.
    source_files = [p for t in TEACHERS for p in sorted(src[t]['folder'].rglob('*')) if p.is_file()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        source_hashes = dict(zip(source_files, pool.map(sha, source_files)))
    for teacher in TEACHERS:
        folder = src[teacher]['folder']
        attempts = [json.loads(x) for x in (folder / 'qa/attempts.jsonl').read_text().splitlines()]
        written = {r['episode_id']: r for r in attempts if r['event'] == 'written'}
        for row in src[teacher]['rows']:
            assert source_hashes[folder / row['file']] == written[row['episode_id']]['sha256'], row['episode_id']
    print(f'Verified {len(source_files)} source files; 640 episode hashes match original attempt logs.', flush=True)

    builds = []
    jobs = []
    for n, selected in selections.items():
        train = [r for r in src['Markovian']['train'] if r['scenario_id'] in selected]
        for teacher in TEACHERS:
            source = src[teacher]
            target = DATA / teacher / f'Clean_{n}'
            stage = target.with_name('.' + target.name + '.building')
            for directory in ('train', 'test', 'qa'):
                (stage / directory).mkdir(parents=True, exist_ok=True)
            rows = [r for r in source['rows'] if r['split'] == 'test' or r['scenario_id'] in selected]
            episodes = [r for r in source['episodes'] if r['split'] == 'test' or r['scenario_id'] in selected]
            assert len(rows) == len(episodes) == n + 64
            write_csv(stage / 'training_manifest.csv', source['fields'], rows)
            write_csv(stage / 'episodes.csv', source['ep_fields'], episodes)
            write_csv(stage / 'scenarios_train.csv', source['scenario_fields'], train)
            shutil.copy2(source['folder'] / 'scenarios_test.csv', stage / 'scenarios_test.csv')

            audit = []
            for scenario_id in train_ids:
                reasons = []
                for t in TEACHERS:
                    r = src[t]['by_id'][scenario_id]
                    if r['eligible_initial_bc'] != 'True':
                        reasons.append(f'{t}: {r["reason"]}')
                    if n == 223 and has_chatter(r):
                        reasons.append(f'{t}: hold_chatter')
                audit.append(dict(scenario_id=scenario_id, included=scenario_id in selected,
                                  reason='; '.join(reasons) if reasons else 'paired inclusion criteria passed'))
            write_csv(stage / 'selection_audit.csv', ['scenario_id', 'included', 'reason'], audit)
            provenance = []
            for row in rows:
                original = source['folder'] / row['file']
                digest = source_hashes[original]
                jobs.append((original, stage / row['file'], digest))
                provenance.append(dict(episode_id=row['episode_id'], scenario_id=row['scenario_id'], split=row['split'],
                                       file=row['file'], source_file=str(Path('..') / 'Clean' / row['file']),
                                       sha256=digest, bytes=original.stat().st_size))
            write_csv(stage / 'episode_provenance.csv', list(provenance[0]), provenance)

            manifest = deepcopy(source['manifest'])
            manifest['dataset_id'] = f'{teacher}/Clean_{n}'
            manifest['dataset_kind'] = 'paired clean trajectory subset; byte-identical episode copies'
            manifest['counts'] = {'train_episodes': n, 'test_episodes': 64, 'total_episodes': n + 64}
            manifest['scenario_split_definition'] = {
                'train': f'scenarios_train.csv ({n} paired selected training scenarios)',
                'test': 'scenarios_test.csv (all 64 original test scenarios, unfiltered)',
                'files_sha256': {f: sha(stage / f) for f in ('scenarios_train.csv', 'scenarios_test.csv')},
                'source_sobol': source['manifest']['scenario_split_definition']['sobol'],
                'train_sampling_note': 'Filtered subset of the frozen 256-point Sobol cohort; not a newly generated power-of-two net.',
                'test_use': 'Held-out evaluation only; never used for subset selection, normalisation or tuning.',
                'no_validation_split': True,
            }
            manifest['derivation'] = {
                'version': 'paired-clean-subsets-v1',
                'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'suffix_meaning': 'Number of training trajectories PER expert; test remains 64.',
                'selection': 'Both teachers initial-BC eligible' + (' and neither teacher has hold_chatter' if n == 223 else '; chatter flags retained'),
                'selection_scope': 'Training only. All original test trajectories retained, including any test chatter flags.',
                'paired_teacher': next(t for t in TEACHERS if t != teacher),
                'source_dataset': '../Clean',
                'source_metadata_sha256': {f: source_hashes[source['folder'] / f] for f in ('manifest.json', 'episodes.csv', 'training_manifest.csv', 'scenarios_train.csv', 'scenarios_test.csv')},
                'source_qa_report_sha256': source_hashes[source['folder'] / 'qa/qa_report.json'],
                'builder_sha256': sha(SCRIPT),
                'raw_episode_metadata': 'Unmodified: episode identity, teacher version, schema and code digest remain those of source collection.',
                'collection_configuration': 'The inherited collection_configuration describes the ORIGINAL acquisition, not the filtered inventory; use counts above for this subset.',
                'changes_not_made': ['no resimulation', 'no relabelling', 'no array modification or decimation', 'no normalisation fitting', 'no teacher changes'],
            }
            manifest['written'] = manifest['derivation']['created_utc']
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            # Preserve source QA as explicitly inherited evidence, not a purported rerun on this inventory.
            shutil.copy2(source['folder'] / 'qa/qa_report.json', stage / 'qa/source_qa_report.json')
            qa_fields, qa_rows = read_csv(source['folder'] / 'qa/episode_checks.csv')
            episode_ids = {r['episode_id'] for r in rows}
            qa_rows = [r for r in qa_rows if r['episode_id'] in episode_ids]
            assert len(qa_rows) == n + 64
            write_csv(stage / 'qa/inherited_episode_checks.csv', qa_fields, qa_rows)
            coverage_report = {'train': coverage(train), 'test': coverage(source['test']),
                               'source_train': coverage(source['train']),
                               'note': 'Filtering changes train distribution. Test is unchanged. Exact boundaries and six-dimensional coverage are not guaranteed.'}
            (stage / 'qa/coverage.json').write_text(json.dumps(coverage_report, indent=2))
            (stage / 'README.md').write_text(
                f'# {teacher} Clean_{n}\n\n'
                f'{n} paired training episodes and all 64 original test episodes.\n\n'
                'See manifest.json for selection and provenance, selection_audit.csv for all 256 original training decisions, '
                'and qa/subset_qa_report.json for copy verification. Original trajectory arrays are unchanged.\n\n'
                + ('Training excludes every scenario where either teacher has hold_chatter. Test chatter remains unfiltered.\n' if n == 223 else 'Successful training episodes retain original chatter flags; this is not a chatter-free dataset.\n')
            )
            builds.append(dict(teacher=teacher, n=n, target=target, stage=stage, rows=rows,
                               provenance=provenance, coverage=coverage_report, source=source))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(copy_verified, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 200 == 0 or i == len(jobs):
                print(f'Copied and SHA-256 verified {i}/{len(jobs)} episode files.', flush=True)

    # Verify inventories, pairing, selection and unchanged sources before publishing directories.
    for b in builds:
        teacher, n, stage = b['teacher'], b['n'], b['stage']
        source = b['source']
        assert len(list((stage / 'train').glob('*.npz'))) == n
        assert len(list((stage / 'test').glob('*.npz'))) == 64
        assert sha(stage / 'scenarios_test.csv') == source_hashes[source['folder'] / 'scenarios_test.csv']
        assert {r['scenario_id'] for r in b['rows'] if r['split'] == 'train'} == selections[n]
        assert all(r['eligible_initial_bc'] == 'True' for r in b['rows'] if r['split'] == 'train')
        if n == 223:
            assert all(not has_chatter(r) for r in b['rows'] if r['split'] == 'train')
    for n in selections:
        a, b = [x for x in builds if x['n'] == n]
        for filename in ('scenarios_train.csv', 'scenarios_test.csv', 'selection_audit.csv'):
            assert sha(a['stage'] / filename) == sha(b['stage'] / filename), filename
    current_files = [p for t in TEACHERS for p in sorted(src[t]['folder'].rglob('*')) if p.is_file()]
    assert set(current_files) == set(source_files)
    with ThreadPoolExecutor(max_workers=4) as pool:
        after_hashes = dict(zip(source_files, pool.map(sha, source_files)))
    assert source_hashes == after_hashes, 'Original data changed during subset build'

    summary = []
    for b in builds:
        rs = b['rows']
        n = b['n']
        episode_rows = [r for r in b['source']['episodes'] if r['split'] == 'train' and r['scenario_id'] in selections[n]]
        stats = {
            'dataset': f'{b["teacher"]}/Clean_{n}', 'train': n, 'test': 64,
            'train_steps': sum(int(r['T_actions']) for r in episode_rows),
            'train_chatter': sum(has_chatter(r) for r in rs if r['split'] == 'train'),
            'test_chatter': sum(has_chatter(r) for r in rs if r['split'] == 'test'),
            'bytes': sum(int(r['bytes']) for r in b['provenance']),
            'empty_train_cells_of_32': len(b['coverage']['train']['distance2_height2_direction8']['empty_cells']),
        }
        qa = dict(status='PASS', checks={
            'expected_inventory': True, 'paired_scenarios_identical': True,
            'train_test_disjoint': True, 'training_selection_rule': True,
            'test_scenarios_unfiltered': True, 'all_episode_copies_sha256_identical_to_source': True,
            'copies_do_not_alias_source_files': True, 'all_original_files_unchanged': True,
            'source_episode_hashes_match_collection_attempt_logs': True,
        }, episode_files_verified=n+64, statistics=stats,
                  inherited_qa='Original full trajectory QA/replay results are inherited via byte-identical files; no new simulation or expert replay was run for this derivative.')
        (b['stage'] / 'qa/subset_qa_report.json').write_text(json.dumps(qa, indent=2))
        b['stage'].rename(b['target'])
        summary.append(stats)

    report_dir = DATA / 'clean_subsets'
    report_dir.mkdir(exist_ok=True)
    shutil.copy2(SCRIPT, report_dir / SCRIPT.name)
    (report_dir / 'summary.json').write_text(json.dumps(dict(status='PASS',datasets=summary,
        selection_counts={'common_success':253,'common_success_without_either_chatter':223},
        original_files_verified_unchanged=len(source_files), episode_copies_verified=len(jobs)),indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('DONE: four paired dataset folders published; original Clean folders unchanged.', flush=True)


if __name__ == '__main__':
    main()
