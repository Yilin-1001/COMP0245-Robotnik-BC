"""Unified frozen test suite. Default: 24 terminal models + expert, 240 scenes, 6000 rollouts.

python -m evaluation.test_suite --rows R4 R8 --seeds 0 --suites T0 T1
python -m evaluation.test_suite --resume results/test/<existing-run>
python -m evaluation.test_suite --plots-only results/test/<existing-run>
"""
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
import datetime
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import platform
import re
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np

from evaluation.suite_manifest import (TASK_DIR, ROWS, INTERFACE, build_manifest, freeze_manifest,
                                       load_manifest, select_models, sha_file, digest, scene_from_record)
from evaluation.suite_rollout import PROTOCOL, rollout


_WORKER_POLICIES={}


def parser():
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--rows',nargs='+',choices=list(ROWS))
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--suites',nargs='+',choices=[f'T{i}' for i in range(6)])
    p.add_argument('--models',nargs='+',help='explicit step_42000 .npz/.pt models, each with adjacent config.yaml')
    p.add_argument('--workers',type=int,default=min(12,len(os.sched_getaffinity(0))))
    p.add_argument('--out',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--plots-only',type=Path)
    p.add_argument('--manifest',type=Path,default=TASK_DIR/'experiments/splits/unified-test-v1.json')
    p.add_argument('--dry-run',action='store_true',help='freeze/preflight only; no formal rollouts')
    return p


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(data,f,indent=1,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def job_identity(run_hash,policy_id,scene_id):
    return {'run_hash':run_hash,'policy_id':policy_id,'scene_id':scene_id}


def job_path(directory,identity):
    # Full model names are retained in run/rollout tables; the hash avoids path and filename collisions.
    name=re.sub(r'[^A-Za-z0-9_.-]','_',identity['policy_id'])[:150]
    return Path(directory)/'jobs'/f"{name}__{identity['scene_id']}__{digest(identity)[:12]}.json"


def save_job(path,identity,record):
    atomic_json(path,{'identity':identity,'record_sha256':digest(record),'record':record})


def load_job(path,identity):
    if not Path(path).exists():return None
    try:data=json.loads(Path(path).read_text())
    except (ValueError,OSError) as e:raise ValueError(f'{path}: invalid result integrity') from e
    if data.get('identity')!=identity:raise ValueError(f'{path}: incompatible job identity')
    if data.get('record_sha256')!=digest(data.get('record')):raise ValueError(f'{path}: result integrity mismatch')
    validate_outcome(data['record'],identity)
    return data['record']


def validate_outcome(record,identity):
    required={'success','grade','completion_time_s','final1s_max_err_mm','scenario_id','model_id','failure'}
    if not required <= record.keys():raise ValueError('result integrity: incomplete outcome schema')
    if record['model_id']!=identity['policy_id'] or record['scenario_id']!=identity['scene_id']:
        raise ValueError('result identity disagrees with outcome')
    if type(record['success']) is not bool or record['grade'] not in tuple('ABCDEF'):raise ValueError('invalid success/grade')
    if record['grade']=='F' and (record['success'] or not record['failure']):raise ValueError('invalid F outcome')
    if record['success'] and record['completion_time_s'] is None:raise ValueError('success lacks completion time')
    if record['grade']!='F' and record['final1s_max_err_mm'] is None:raise ValueError('missing terminal error')


def simulator_preflight():
    from core.interfaces import Observation
    from evaluation.evaluate import make_robot_env,MarkovianReference
    from types import SimpleNamespace
    env=make_robot_env()
    try:
        scene=SimpleNamespace(seed=9000001,target_ee_pos=np.array([1.2,.8,.6]),base_start_pos=np.array([-.2,.1,0.]),
                              base_start_bearing=.4,physics={},scenario_id='development-preflight')
        obs=env.reset(scene);policy=MarkovianReference(lead_time=.50)
        env.step(policy.act(obs))
    finally:env.close()


def validate_run(run,manifest):
    ident=run['identity']
    if digest(ident)!=run['run_hash']:raise ValueError('run identity integrity mismatch')
    if run['models']!=ident['models'] or run['scenario_ids']!=ident['scenario_ids']:
        raise ValueError('run selection differs from pinned identity')
    if run['scene_manifest_sha256']!=manifest['sha256'] or ident['manifest_sha256']!=manifest['sha256']:
        raise ValueError('run scenario manifest identity differs')
    if len({m['id'] for m in run['models']})!=len(run['models']) or len(set(run['scenario_ids']))!=len(run['scenario_ids']):
        raise ValueError('duplicate run selection identities')
    if not set(run['scenario_ids'])<={s['scenario_id'] for s in manifest['scenarios']}:
        raise ValueError('run selected scenario missing from manifest')
    if run['expected_jobs']!=len(run['models'])*len(run['scenario_ids']):raise ValueError('run expected coverage differs')


def code_hashes():
    files=['evaluation/test_suite.py','evaluation/suite_rollout.py','evaluation/suite_manifest.py',
           'evaluation/evaluate.py','training/common/policies.py','training/common/models.py','training/common/checkpoint.py','core/interfaces.py','core/features.py',
           'core/feature_sets.py','core/env.py','data_collection/expert.py','robotnik_arm_control.py',
           'settings.yaml','configs/robotnik_arm.json','models/summit_xl_arm_description/summit_xl_complete_pin.urdf']
    files += [str(p.relative_to(TASK_DIR)) for p in (TASK_DIR/'models/summit_xl_arm_description').rglob('*') if p.is_file()]
    import simulation_and_control
    package=Path(simulation_and_control.__file__).resolve().parent
    result={f:sha_file(TASK_DIR/f) for f in sorted(set(files))}
    result.update({f'simulation_and_control/{p.relative_to(package)}':sha_file(p) for p in package.rglob('*.py')})
    return result


def _policy(record):
    if record['kind']=='expert':
        from evaluation.evaluate import MarkovianReference
        return MarkovianReference(lead_time=.50)
    from evaluation.evaluate import mlp_factory
    return mlp_factory(record['path'],backend=record.get('backend','numpy'))


def _init_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def execute_job(job):
    model,record=job
    if model['id'] not in _WORKER_POLICIES:_WORKER_POLICIES[model['id']]=_policy(model)
    result=rollout(_WORKER_POLICIES[model['id']],scene_from_record(record),record['duration_s'],record['hold_s'],INTERFACE)
    d=np.array(record['target'])-record['start'];c,s=np.cos(record['yaw']),np.sin(record['yaw'])
    return {'model_id':model['id'],'row':model['row'],'seed':model['seed'],'round':model.get('round'),
            'suite':record['suite'],'band':record['band'],'parent_id':record.get('parent_id'),
            'time_limit_s':record['duration_s'],'hold_required_s':record['hold_s'],
            'target_dist_m':float(np.linalg.norm(d[:2])),'target_z_m':record['target'][2],
            'target_bx':float(c*d[0]+s*d[1]),'target_by':float(-s*d[0]+c*d[1]),
            'start_x':record['start'][0],'start_y':record['start'][1],'start_yaw':record['yaw'],
            'target_x':record['target'][0],'target_y':record['target'][1],**result}


def benchmark(models):
    from core.interfaces import Observation
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    obs=Observation(np.zeros(3),0.,np.array([0,1.57,0,1,0,1,0.]),np.zeros(7),np.array([1.,0.,.7]))
    output={}
    for m in models:
        pol=_policy(m);pol.reset()
        for _ in range(30):pol.act(obs)
        samples=[]
        for _ in range(200):
            t=time.perf_counter_ns();pol.act(obs);samples.append((time.perf_counter_ns()-t)/1000)
        output[m['id']]={f'p{q}_us':float(np.percentile(samples,q)) for q in (50,95,99)}
    return output


def _git(args):
    p=subprocess.run(['git','-C',str(TASK_DIR),*args],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else None


def prepare(args):
    if args.workers<1:raise ValueError('--workers must be positive')
    if args.resume:
        directory=args.resume.resolve();run=json.loads((directory/'run.json').read_text())
        manifest=load_manifest(directory/'scene_manifest.json')
        validate_run(run,manifest)
        if run['identity']['code_sha256']!=code_hashes():raise ValueError('resume code identity differs')
        if run['scene_manifest_sha256']!=manifest['sha256']:raise ValueError('resume scenario identity differs')
        if args.rows or args.seeds or args.suites or args.models:raise ValueError('resume uses pinned selection; omit model/suite filters')
        for m in run['models']:
            if m['kind']=='model' and (sha_file(m['path'])!=m['sha256'] or sha_file(m['config_path'])!=m['config_sha256']):
                raise ValueError('resume model/config identity differs')
        if digest(run['identity'])!=run['run_hash']:raise ValueError('run identity integrity mismatch')
        return directory,run,manifest
    if args.models and (args.rows or args.seeds):raise ValueError('--models cannot be combined with --rows/--seeds')
    models,missing=select_models(args.rows,args.seeds,args.models)
    if not models:raise ValueError('no requested student models available')
    models.append({'id':'Markovian-expert-A','row':'expert','seed':None,'round':None,'kind':'expert',
                   'sha256':digest({'teacher':'A','lead_time':.5,'interface':INTERFACE}),'parameters':None})
    # Load all adapters before formal execution so broken dependencies/weights do not become thousands of F rows.
    for m in models:_policy(m)
    manifest=freeze_manifest(args.manifest,build_manifest())
    selected=[s for s in manifest['scenarios'] if args.suites is None or s['suite'] in args.suites]
    identity={'code_sha256':code_hashes(),'manifest_sha256':manifest['sha256'],'models':models,
              'scenario_ids':[s['scenario_id'] for s in selected],'protocol':PROTOCOL,'interface':INTERFACE}
    now=datetime.datetime.now(ZoneInfo('Europe/London'))
    names='-'.join(args.rows) if args.rows else ('custom' if args.models else 'R1-R8')
    if args.seeds:names+='_s'+'-'.join(map(str,args.seeds))
    directory=(args.out or TASK_DIR/'results/test'/f"{now:%Y-%m-%d_%H%M%S}__{names}").resolve()
    directory.mkdir(parents=True,exist_ok=False)
    packages={}
    for package in ['numpy','scipy','matplotlib','pybullet','pin']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]='unknown'
    run={'version':'unified-test-v1','status':'prepared','started_local':now.isoformat(),
         'started_utc':now.astimezone(datetime.timezone.utc).isoformat(),'run_hash':digest(identity),'identity':identity,
         'scene_manifest_sha256':manifest['sha256'],'models':models,'missing_models':missing,
         'scenario_ids':identity['scenario_ids'],'expected_jobs':len(models)*len(selected),
         'requested_rows':args.rows or list(ROWS),'requested_seeds':args.seeds or [0,1,2],
         'git_head':_git(['rev-parse','HEAD']),'git_status':_git(['status','--short']),
         'platform':platform.platform(),'python':platform.python_version(),'packages':packages,
         'cpu_count':len(os.sched_getaffinity(0)),'workers':args.workers,
         'report_code_sha256':{name:sha_file(TASK_DIR/'evaluation'/name) for name in ['suite_report.py','suite_plots.py']}}
    atomic_json(directory/'scene_manifest.json',manifest)
    atomic_json(directory/'run.json',run)
    return directory,run,manifest


def main(argv=None):
    args=parser().parse_args(argv)
    if args.plots_only:
        from evaluation.suite_report import generate_report
        print(json.dumps(generate_report(args.plots_only),indent=2));return 0
    directory,run,manifest=prepare(args)
    scenes=[s for s in manifest['scenarios'] if s['scenario_id'] in set(run['scenario_ids'])]
    pending=[];done=0
    for s in scenes:
        for m in run['models']:
            ident=job_identity(run['run_hash'],m['id'],s['scenario_id'])
            if load_job(job_path(directory,ident),ident) is None:pending.append((m,s))
            else:done+=1
    print(f"Output: {directory}\nModels: {len(run['models'])}; scenes: {len(scenes)}; completed {done}/{run['expected_jobs']}; pending {len(pending)}",flush=True)
    if run['missing_models']:print('INCOMPLETE MODEL COVERAGE: '+json.dumps(run['missing_models']),flush=True)
    if args.dry_run:return 0
    if 'latency_benchmark' not in run:
        run['latency_benchmark']=benchmark(run['models']);atomic_json(directory/'run.json',run)
    run['status']='running';run['workers']=args.workers;atomic_json(directory/'run.json',run)
    started=time.perf_counter();last_log=started;new=failed=0
    try:
        if pending:
            simulator_preflight()
            with mp.get_context('spawn').Pool(args.workers,_init_worker) as pool:
                iterator=pool.imap_unordered(execute_job,pending,chunksize=1)
                while new<len(pending):
                    try:record=iterator.next(timeout=10)
                    except mp.TimeoutError:
                        print(f"Progress {done+new}/{run['expected_jobs']}; elapsed {(time.perf_counter()-started)/60:.1f} min",flush=True)
                        continue
                    ident=job_identity(run['run_hash'],record['model_id'],record['scenario_id'])
                    save_job(job_path(directory,ident),ident,record)
                    new+=1;failed+=int(record['grade']=='F')
                    if new%10==0 or time.perf_counter()-last_log>=10:
                        elapsed=time.perf_counter()-started
                        print(f"Progress {done+new}/{run['expected_jobs']}; new F={failed}; {new/max(elapsed,1)*60:.1f} jobs/min; elapsed {elapsed/60:.1f} min",flush=True)
                        last_log=time.perf_counter()
        run['status']='evaluated';run['finished_local']=datetime.datetime.now(ZoneInfo('Europe/London')).isoformat()
    except BaseException:
        run['status']='interrupted';atomic_json(directory/'run.json',run);raise
    run['last_execution_wall_s']=time.perf_counter()-started;atomic_json(directory/'run.json',run)
    from evaluation.suite_report import generate_report
    result=generate_report(directory)
    run['status']='complete' if result['complete'] else 'incomplete';atomic_json(directory/'run.json',run)
    print(json.dumps({'directory':str(directory),**result},indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
