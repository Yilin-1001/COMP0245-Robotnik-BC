"""Immutable scenes and explicitly pinned terminal-model inventory for unified-test-v1."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import qmc

TASK_DIR = Path(__file__).resolve().parents[1]
ROWS = {'R1':'R1-baseline','R2':'R2-gate','R3':'R3-dart','R4':'R4-dart-gate',
        'R5':'R5-baseline-dagger','R6':'R6-gate-dagger','R7':'R7-dart-dagger','R8':'R8-dart-gate-dagger'}
MASTER_SEED = 1101000
INTERFACE = {'control_hz':1000,'torque_limit':30.,'wheel_limit':6.}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def _core_scene(row):
    return {'scenario_id':row['scenario_id'],'suite':'T0','band':'core','seed':int(row['seed']),
            'start':[float(row['start_x']),float(row['start_y']),0.], 'yaw':float(row['start_yaw']),
            'target':[float(row[k]) for k in ('target_x','target_y','target_z')],
            'duration_s':40.,'hold_s':3.,'parent_id':None}


def build_manifest(task_dir=TASK_DIR):
    task=Path(task_dir)
    split_path=task/'experiments/splits/clean253.json'
    split=json.loads(split_path.read_text())
    core_path=task/'data/Markovian/Clean_253/scenarios_test.csv'
    core_hash=sha_file(core_path)
    if core_hash!=split['source_sha256']['A/clean253/scenarios_test.csv']:
        raise ValueError('T0 source CSV differs from frozen clean253 split')
    scenes=[_core_scene(r) for r in read_csv(core_path)]
    if len(scenes)!=64 or len({r['scenario_id'] for r in scenes})!=64:
        raise ValueError('T0 must contain 64 unique scenarios')
    bands=[('T1','3-6m',24,(3,6),(.1,1.25),80,3),('T1','6-10m',24,(6,10),(.1,1.25),80,3),
           ('T1','10-15m',24,(10,15),(.1,1.25),120,3),('T2','near',24,(.3,.5),(.1,1.25),40,3),
           ('T3','low',16,(.5,3),(.05,.1),40,3),('T3','high',16,(.5,3),(1.25,1.30),40,3),
           ('T5','long-hold',24,(.5,3),(.1,1.25),60,20)]
    for j,(suite,band,n,dist,height,cap,hold) in enumerate(bands):
        seed=int(np.random.SeedSequence([MASTER_SEED,j]).generate_state(1)[0])
        samples=qmc.LatinHypercube(d=6,seed=seed).random(n)
        for i,u in enumerate(samples):
            sx,sy=u[:2]-.5; yaw,az=(u[2:4]*2-1)*np.pi
            d=dist[0]+u[4]*(dist[1]-dist[0]);z=height[0]+u[5]*(height[1]-height[0])
            scenes.append({'scenario_id':f'{suite}-{band}-{i:03d}','suite':suite,'band':band,
                           'seed':1200000+1000*j+i,'start':[float(sx),float(sy),0.],'yaw':float(yaw),
                           'target':[float(sx+d*np.cos(yaw+az)),float(sy+d*np.sin(yaw+az)),float(z)],
                           'duration_s':float(cap),'hold_s':float(hold),'parent_id':None})
    rng=np.random.default_rng([MASTER_SEED,40])
    parents=rng.choice(64,size=24,replace=False)
    for i,idx in enumerate(parents):
        parent=scenes[int(idx)]
        radius,angle=rng.uniform(1,3),rng.uniform(-np.pi,np.pi)
        start=np.array([radius*np.cos(angle),radius*np.sin(angle),0.])
        shift=start-np.array(parent['start'])
        scenes.append({**parent,'scenario_id':f'T4-shift-{i:03d}','suite':'T4','band':'paired-start',
                       'seed':1209000+i,'start':start.tolist(),'target':(np.array(parent['target'])+shift).tolist(),
                       'parent_id':parent['scenario_id'],'translation':shift.tolist()})
    scenes.sort(key=lambda r:(r['suite'],r['band'],r['scenario_id']))
    # Audit coordinates against all original training/validation and current augmentation inventories.
    inventories={task/'data/scenarios/scenarios_train.csv'}
    for f in (task/'data/datasets/A').glob('*/metadata.json'):
        meta=json.loads(f.read_text())
        inventories.add(task/meta['source']['root']/'scenarios_train.csv')
    keys=('start_x','start_y','start_yaw','target_x','target_y','target_z')
    original=set(); source_hashes={}; original_starts=set(); original_targets=set(); source_seeds=set()
    for f in sorted(inventories):
        source_hashes[str(f.relative_to(task))]=sha_file(f)
        source_rows=read_csv(f)
        original.update(tuple(float(r[k]) for k in keys) for r in source_rows)
        original_starts.update((float(r['start_x']),float(r['start_y'])) for r in source_rows)
        original_targets.update(tuple(float(r[k]) for k in ('target_x','target_y','target_z')) for r in source_rows)
        source_seeds.update(int(r['seed']) for r in source_rows)
    overlaps=[]
    for r in scenes:
        key=tuple(r['start'][:2])+ (r['yaw'],)+tuple(r['target'])
        if key in original or tuple(r['start'][:2]) in original_starts or tuple(r['target']) in original_targets or r['seed'] in source_seeds:
            overlaps.append(r['scenario_id'])
    if overlaps:raise ValueError(f'test scenes overlap training/validation: {overlaps}')
    assert len(scenes)==240 and len({r['seed'] for r in scenes})==240
    return {'version':'unified-test-v1','generation_seed':MASTER_SEED,'generation':'LHS per independent band; T4 paired translations',
            'source_test_csv_sha256':core_hash,'source_split_sha256':sha_file(split_path),
            'interface':INTERFACE,'scenarios':scenes,
            'source_audit':{'source_inventory_sha256':source_hashes,'unintended_heldout_overlaps':overlaps}}


def freeze_manifest(path, manifest):
    path=Path(path)
    body={k:v for k,v in manifest.items() if k!='sha256'}
    pinned={**body,'sha256':digest(body)}
    if path.exists():
        existing=load_manifest(path)
        if existing!=pinned:raise ValueError('frozen manifest differs; refusing overwrite')
        return existing
    path.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create: another process cannot silently overwrite a frozen scene set.
    with path.open('x') as f:json.dump(pinned,f,indent=2,allow_nan=False);f.write('\n')
    return pinned


def load_manifest(path):
    data=json.loads(Path(path).read_text())
    if data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('scenario manifest hash mismatch')
    return data


def scene_from_record(record):
    from data_collection.scenarios import Scenario
    return Scenario(seed=record['seed'],target_ee_pos=np.array(record['target']),
                    base_start_pos=np.array(record['start']),base_start_bearing=record['yaw'],
                    physics={},scenario_id=record['scenario_id'],split='unified-test')


def _artifact_metadata(path):
    path=Path(path)
    if path.suffix=='.pt':
        from training.common.checkpoint import load_checkpoint
        payload=load_checkpoint(path)
        return {**payload['meta'],'normaliser':payload['normaliser'],'source_step':payload['step']}, sum(v.numel() for v in payload['model_state'].values()), 'torch'
    if path.suffix!='.npz':raise ValueError('supported model artifacts are .npz and .pt')
    with np.load(path) as z:
        meta=json.loads(str(z['metadata_json']))
        if meta.get('format')!='mlp-numpy-export-v1':raise ValueError('not an MLP NumPy export')
        count=sum(z[k].size for k in z.files if k!='metadata_json' and np.issubdtype(z[k].dtype,np.number))
    return meta,count,'numpy'


def _model_record(path,row,seed,round_number):
    path=Path(path).resolve()
    meta,params,backend=_artifact_metadata(path)
    if meta.get('source_step')!=42000:
        raise ValueError(f'{path}: expected step_42000 model')
    if any(meta['interface'].get(k)!=v for k,v in INTERFACE.items()):
        raise ValueError(f'{path}: incompatible control interface')
    config=path.parent/'config.yaml'
    if not config.exists():raise FileNotFoundError(f'model config missing: {config}')
    c=yaml.safe_load(config.read_text());ident=c['identity'];rid=meta['run_id']
    inferred=re.search(r'(?:__s|-s)([0-9]+)(?:__|$)',rid)
    if (seed is not None and ident['seed']!=seed) or (inferred and int(inferred[1])!=ident['seed']):
        raise ValueError(f'{path}: config/model training seed mismatch')
    for key in ['run_id','teacher','objective']:
        if key in ident and ident[key]!=meta.get(key):raise ValueError(f'{path}: config/model {key} mismatch')
    if c['training']['updates']!=42000:raise ValueError('config terminal step differs from 42000')
    definition=meta.get('feature_definition','features-v1')
    if c.get('data',{}).get('feature_definition',definition)!=definition:
        raise ValueError('config/model feature definition mismatch')
    from core.feature_sets import feature_names
    names=feature_names(meta['teacher'],definition)
    if len(names)!=meta['input_dim'] or c.get('model',{}).get('input_dim',meta['input_dim'])!=meta['input_dim']:
        raise ValueError('config/model feature dimension mismatch')
    if meta.get('feature_names',names)!=names:raise ValueError('model feature order mismatch')
    if c.get('interface',meta['interface'])!=meta['interface']:raise ValueError('config/model interface mismatch')
    if 'normaliser' in meta:
        normaliser_sha=hashlib.sha256(json.dumps(meta['normaliser'],sort_keys=True).encode()).hexdigest()
        if meta.get('normaliser_sha256')!=normaliser_sha:raise ValueError('model normalizer integrity mismatch')
        if c.get('provenance',{}).get('normaliser_sha256',normaliser_sha)!=normaliser_sha:
            raise ValueError('config/model normalizer mismatch')
    return {'id':rid,'row':row,'seed':ident['seed'], 'round':round_number,'path':str(path),
            'sha256':sha_file(path),'config_path':str(config),'config_sha256':sha_file(config),
            'checkpoint_step':42000,'input_dim':meta['input_dim'],'backend':backend,
            'feature_definition':definition,'parameters':int(params),'kind':'model'}


def select_models(rows=None,seeds=None,paths=None,task_dir=TASK_DIR):
    task=Path(task_dir);models=[];missing=[]
    if paths:
        for path in paths:
            p=Path(path).resolve()
            meta,_,_=_artifact_metadata(p)
            rid=meta.get('run_id',p.stem)
            row=rid.split('-')[0] if rid.split('-')[0] in ROWS else rid
            models.append(_model_record(p,row,None,None))
    else:
        for row in rows or ROWS:
            name=ROWS[row]
            if row in ('R1','R2','R3','R4'):
                rnd=0
                selected={seed:task/'checkpoints/ablation/2026-10-28__training-ablation-v1'/
                          f'{name}__MLP01-256x3__s{seed}__v1__2026-10-28'/'step_42000.npz' for seed in seeds or [0,1,2]}
            else:
                summary=task/'results/ablation/2026-11-01__training-ablation-v2/dagger'/name/'summary.csv'
                entries=read_csv(summary) if summary.exists() else []
                candidates=[r for r in entries if int(r['round'])>0 and int(r['seed'])==0]
                if not candidates:
                    missing.extend({'row':row,'seed':s,'reason':'no current validation summary'} for s in seeds or [0,1,2])
                    continue
                def key(r):
                    a,b=map(int,r['sustained_hold'].split('/'))
                    return a/b,int(r['round'])
                rnd=int(max(candidates,key=key)['round'])
                selected={int(r['seed']):task/Path(r['model']).parent/'step_42000.npz' for r in entries if int(r['round'])==rnd}
            for seed in seeds or [0,1,2]:
                path=selected.get(seed)
                if path is None or not path.exists():
                    missing.append({'row':row,'seed':seed,'reason':str(path or 'selected round seed absent')})
                    continue
                models.append(_model_record(path,row,seed,rnd))
    if len({m['id'] for m in models})!=len(models):raise ValueError('duplicate model identities')
    return models,missing
