"""Data-only aggregation and artifact generation; does not import physics or model adapters."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from core.model_names import report_label
from evaluation.suite_manifest import digest

BOOTSTRAP_SEED = 1101031


def summarise(rows,models=None,scenes=None):
    if models is None:
        models=list({r['model_id']:{'id':r['model_id'],'row':r['row'],'seed':r['seed']} for r in rows}.values())
    if scenes is None:
        scenes=list({r['scenario_id']:{k:r[k] for k in ['scenario_id','suite','band']} for r in rows}.values())
    groups=sorted({(r['suite'],'all') for r in scenes}|{(r['suite'],r['band']) for r in scenes if r['suite'] in ('T1','T3')})
    policies=[('row',r) for r in sorted({m['row'] for m in models})]+[('model',m['id']) for m in models]
    expert={r['scenario_id']:r['success'] for r in rows if r['row']=='expert'}
    output=[]
    for suite,band in groups:
        group_ids={r['scenario_id'] for r in scenes if r['suite']==suite and (band=='all' or r['band']==band)}
        for level,policy in policies:
            matched=[m for m in models if m['row' if level=='row' else 'id']==policy]
            model_ids={m['id'] for m in matched}
            selected=[r for r in rows if r['scenario_id'] in group_ids and r['model_id'] in model_ids]
            for subset in ('all','expert_success'):
                eligible=group_ids if subset=='all' else {sid for sid in group_ids if expert.get(sid) is True}
                chosen=[r for r in selected if r['scenario_id'] in eligible]
                unknown_expert=len(group_ids-expert.keys()) if subset=='expert_success' else 0
                expected=len(eligible)*len(matched);missing=expected-len(chosen)
                clusters=defaultdict(list)
                for r in chosen:clusters[r['scenario_id']].append(float(r['success']))
                values=np.array([np.mean(v) for v in clusters.values()])
                interval=[None,None]
                if len(values) and missing==0 and unknown_expert==0:
                    rng=np.random.default_rng(BOOTSTRAP_SEED)
                    means=values[rng.integers(0,len(values),(1000,len(values)))].mean(axis=1)
                    interval=list(map(float,np.quantile(means,[.025,.975])))
                times=[r['completion_time_s'] for r in chosen if r['success'] and r['completion_time_s'] is not None]
                errors=[r['final1s_max_err_mm'] for r in chosen if r['grade']!='F' and r['final1s_max_err_mm'] is not None]
                grades=Counter(r['grade'] for r in chosen)
                output.append({'level':level,'policy':policy,'row':matched[0]['row'],
                    'report_name':report_label(matched[0]['row']),
                    'seed':matched[0]['seed'] if level=='model' else None,'suite':suite,'band':band,'subset':subset,
                    'n':len(chosen),'n_expected':expected,'n_missing':missing,'expert_pending_scenes':unknown_expert,
                    'coverage_status':'partial' if missing or unknown_expert else 'complete',
                    'scene_n':len(clusters),'models':len(matched),
                    'successes':sum(r['success'] for r in chosen),
                    'success_rate':float(np.mean([r['success'] for r in chosen])) if chosen else None,
                    'ci_low':interval[0],'ci_high':interval[1],
                    'completion_median_s':float(np.median(times)) if times else None,
                    'terminal_error_median_mm':float(np.median(errors)) if errors else None,
                    **{f'grade_{g}':grades[g] for g in 'ABCDEF'}})
    return output


def write_csv(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row if k!='history'))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def load_records(directory):
    from evaluation.suite_manifest import load_manifest
    from evaluation.test_suite import validate_outcome,validate_run
    directory=Path(directory)
    run=json.loads((directory/'run.json').read_text())
    manifest=load_manifest(directory/'scene_manifest.json')
    validate_run(run,manifest)
    expected={(m['id'],sid) for m in run['models'] for sid in run['scenario_ids']}
    if len(expected)!=run['expected_jobs']:raise ValueError('report expected coverage mismatch')
    records=[];seen=set()
    model_map={m['id']:m for m in run['models']}
    scene_map={r['scenario_id']:r for r in manifest['scenarios']}
    for p in sorted((directory/'jobs').glob('*.json')):
        item=json.loads(p.read_text());ident=item['identity']
        if ident['run_hash']!=run['run_hash'] or digest(item['record'])!=item['record_sha256']:
            raise ValueError(f'{p}: result identity/integrity mismatch')
        pair=(ident['policy_id'],ident['scene_id'])
        if pair in seen:raise ValueError(f'{p}: duplicate model/scenario result')
        if pair not in expected:raise ValueError(f'{p}: unexpected model/scenario result')
        record=item['record'];validate_outcome(record,ident)
        for key in ['row','seed']:
            if key in model_map[pair[0]] and record.get(key)!=model_map[pair[0]][key]:raise ValueError('result model metadata differs from pin')
        for key in ['suite','band']:
            if key in scene_map[pair[1]] and record.get(key)!=scene_map[pair[1]][key]:raise ValueError('result scenario metadata differs from pin')
        seen.add(pair);records.append(item['record'])
    return run,records


def generate_report(directory):
    directory=Path(directory)
    run,rows=load_records(directory)
    scene_manifest=json.loads((directory/'scene_manifest.json').read_text())
    selected_scenes=[s for s in scene_manifest['scenarios'] if s['scenario_id'] in set(run['scenario_ids'])]
    summary=summarise(rows,run['models'],selected_scenes)
    write_csv(directory/'rollouts.csv',rows)
    write_csv(directory/'summary.csv',summary)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    expected=run['expected_jobs']
    complete={(r['model_id'],r['scenario_id']) for r in rows}=={(m['id'],sid) for m in run['models'] for sid in run['scenario_ids']} and not run.get('missing_models')
    show=lambda v,fmt='.2f':'N/A' if v is None else format(v,fmt)
    lines=['# Unified R1–R8 test results','',f"Started: {run['started_local']}.",
           f"Status: {'COMPLETE' if complete else 'INCOMPLETE'}; {len(rows)}/{expected} requested rollouts.",
           f"Scene manifest SHA-256: `{run['scene_manifest_sha256']}`.",
           f"Run identity: `{run['run_hash']}`.",'',
           'Only continuous error <10 mm counts as success (3 s; T5 20 s). All rollouts run to their fixed cap. '
           'Completion time includes the qualifying hold. Any exception overrides success with F. '
           'Terminal A–F precision uses the final inclusive second maximum and is independent of success.',
           '', 'T0 was previously viewed during development. This new scoring protocol is not numerically equivalent to the historical rollout-v1 results. '
           'New extension scenes were frozen before this run. T4 deliberately pairs translations with T0; it is not an independent relative-task sample.',
           '', 'Intervals: descriptive 95% scenario-cluster bootstrap (1000 resamples, fixed seed '+str(BOOTSTRAP_SEED)+'). '
           'Training seeds stay together when resampling a scene; intervals do not establish uncertainty over all possible training runs.',
           '', '## Model coverage','', '| Row | Available training seeds | DAgger round |','|---|---|---|']
    byrow=defaultdict(list)
    for m in run['models']:byrow[m['row']].append(m)
    for row,ms in sorted(byrow.items()):
        lines.append(f"| {report_label(row)} | {', '.join(str(m['seed']) for m in ms)} | {ms[0].get('round','—')} |")
    if run.get('missing_models'):lines.extend(['','Missing models: `'+json.dumps(run['missing_models'])+'`.'])
    for subset,title in [('all','All frozen scenes'),('expert_success','Expert-success subset')]:
        lines.extend(['','## '+title,'','| Row | Suite/band | Success | Rate [95% interval] | Completion median (successes only), s | Terminal median, mm |',
                      '|---|---|---|---|---|---|'])
        for r in summary:
            if r['level']!='row' or r['subset']!=subset:continue
            rate='N/A' if r['success_rate'] is None else f"{100*r['success_rate']:.1f}%"
            if r['ci_low'] is not None:rate+=f" [{100*r['ci_low']:.1f}, {100*r['ci_high']:.1f}]"
            if r['coverage_status']=='partial':rate+=f" PARTIAL ({r['n']}/{r['n_expected']} evaluated; expert pending {r['expert_pending_scenes']})"
            lines.append(f"| {report_label(r['policy'])} | {r['suite']}/{r['band']} | {r['successes']}/{r['n']} | {rate} | {show(r['completion_median_s'])} | {show(r['terminal_error_median_mm'])} |")
    lines.extend(['','## Efficiency and runtime','', '| Model | Parameters | Warmed latency p50 / p95 / p99, µs |', '|---|---:|---|'])
    for m in run['models']:
        bench=run.get('latency_benchmark',{}).get(m['id'],{})
        lines.append(f"| {m['id']} | {m.get('parameters','—')} | "+' / '.join(show(bench.get(f'p{q}_us')) for q in (50,95,99))+' |')
    lines.extend(['','Coverage by model/suite: `summary.csv` includes expected/evaluated/missing outcomes and pending expert verdicts. Partial estimates omit confidence intervals; missing jobs are not counted as failures.'])
    failures=Counter(r.get('failure_type') for r in rows if r['grade']=='F')
    lines.extend(['','## Runtime failures','',json.dumps(dict(failures),indent=2),'',
                  'Expert failures describe this controller under this protocol, not physical infeasibility. '
                  'The expert-success denominator is identical across models with complete coverage; no scenes are removed or redrawn.',
                  '', 'Full per-seed/per-model statistics are in `summary.csv`; every rollout is in `rollouts.csv`. '
                  'Atomic `jobs/*.json` preserve histories and exception details. `run.json` pins model, config, scene, protocol, code and runtime provenance.'])
    # Dedicated paired T4 comparison, including models that fail in either world location.
    bykey={(r['model_id'],r['scenario_id']):r for r in rows}
    scene_map={r['scenario_id']:r for r in json.loads((directory/'scene_manifest.json').read_text())['scenarios']}
    pairs=[]
    for r in rows:
        parent=scene_map[r['scenario_id']].get('parent_id')
        original=bykey.get((r['model_id'],parent)) if parent else None
        if original:
            pairs.append({'model_id':r['model_id'],'row':r['row'],'seed':r['seed'],'parent_id':parent,
                          'translated_id':r['scenario_id'],'parent_success':original['success'],'translated_success':r['success'],
                          'parent_terminal_mm':original['final1s_max_err_mm'],'translated_terminal_mm':r['final1s_max_err_mm'],
                          'success_changed':original['success']!=r['success']})
    write_csv(directory/'paired_translation.csv',pairs)
    common=[]
    for row in sorted(byrow):
        for suite in sorted({r['suite'] for r in rows}):
            matched=[r for r in rows if r['row']==row and r['suite']==suite and r['success'] and
                     bykey.get(('Markovian-expert-A',r['scenario_id']),{}).get('success')]
            ratios=[r['completion_time_s']/bykey['Markovian-expert-A',r['scenario_id']]['completion_time_s'] for r in matched
                    if bykey['Markovian-expert-A',r['scenario_id']]['completion_time_s']>0]
            common.append({'row':row,'suite':suite,'common_success_rollouts':len(matched),
                           'median_completion_time_ratio_to_expert':float(np.median(ratios)) if ratios else None})
    write_csv(directory/'common_success_efficiency.csv',common)
    (directory/'summary.md').write_text('\n'.join(lines)+'\n')
    from evaluation.suite_plots import plot_results
    plot_results(directory,rows,summary)
    from evaluation.suite_manifest import sha_file
    import datetime
    (directory/'report.json').write_text(json.dumps({'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'bootstrap_seed':BOOTSTRAP_SEED,'bootstrap_resamples':1000,
        'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in ['suite_report.py','suite_plots.py']},
        'run_hash':run['run_hash'],'rollouts':len(rows),'complete':complete},indent=2)+'\n')
    return {'rollouts':len(rows),'expected':expected,'complete':complete,'failures':dict(failures)}
