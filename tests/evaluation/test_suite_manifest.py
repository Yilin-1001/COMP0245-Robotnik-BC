import copy
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pytest
import yaml

from evaluation.suite_manifest import build_manifest, freeze_manifest, load_manifest, select_models
from conftest import TASK_DIR


@pytest.fixture(scope='module')
def manifest():
    return build_manifest()


def test_manifest_has_exact_counts_ranges_and_unmodified_core(manifest):
    records = manifest['scenarios']
    assert len(records) == 240 and len({r['scenario_id'] for r in records}) == 240
    assert Counter(r['suite'] for r in records) == {'T0':64,'T1':72,'T2':24,'T3':32,'T4':24,'T5':24}
    for r in records:
        d = np.linalg.norm(np.array(r['target'][:2])-r['start'][:2])
        if r['suite']=='T1':
            lo,hi = {'3-6m':(3,6),'6-10m':(6,10),'10-15m':(10,15)}[r['band']]
            assert lo <= d <= hi and r['duration_s'] == (120 if lo==10 else 80)
        if r['suite']=='T2': assert .3 <= d <= .5
        if r['suite']=='T3': assert (.05 <= r['target'][2] <= .1) or (1.25 <= r['target'][2] <= 1.3)
        if r['suite']=='T5': assert r['hold_s']==20 and r['duration_s']==60
    assert manifest['source_audit']['unintended_heldout_overlaps'] == []


def test_t4_is_exact_paired_translation_and_preserves_yaw(manifest):
    byid = {r['scenario_id']:r for r in manifest['scenarios']}
    for r in manifest['scenarios']:
        if r['suite']!='T4':continue
        parent=byid[r['parent_id']]
        assert 1 <= np.linalg.norm(r['start'][:2]) <= 3
        assert np.allclose(np.array(r['target'])-r['start'],np.array(parent['target'])-parent['start'],atol=1e-14)
        assert r['yaw']==parent['yaw'] and r['seed']!=parent['seed']


def test_freeze_is_reproducible_and_rejects_tampering(tmp_path,manifest):
    path=tmp_path/'frozen.json'
    freeze_manifest(path,manifest)
    original=path.read_bytes()
    freeze_manifest(path,build_manifest())
    assert path.read_bytes()==original
    changed=copy.deepcopy(manifest);changed['scenarios'][0]['duration_s']=41
    with pytest.raises(ValueError):freeze_manifest(path,changed)
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):load_manifest(path)


def _fake_run(task,row,round,seed,step=42000):
    ident=f'{row}-r{round}-s{seed}'
    d=task/'weights'/ident;d.mkdir(parents=True)
    meta={'run_id':ident,'source_step':step,'format':'mlp-numpy-export-v1','input_dim':21,
          'teacher':'A','feature_definition':'features-v1',
          'interface':{'control_hz':1000,'torque_limit':30.,'wheel_limit':6.}}
    np.savez(d/'step_42000.npz',metadata_json=np.array(json.dumps(meta)),layer0_W=np.zeros((1,21)))
    (d/'config.yaml').write_text(yaml.safe_dump({'training':{'updates':42000},'identity':{'seed':seed,'run_id':ident,'teacher':'A'},'data':{'feature_definition':'features-v1'},'model':{'input_dim':21}}))
    return str((d/f'{ident}.npz').relative_to(task))


def test_selection_uses_seed_zero_round_tie_rule_and_reports_missing(tmp_path):
    d=tmp_path/'results/ablation/2026-11-01__training-ablation-v2/dagger/R8-dart-gate-dagger';d.mkdir(parents=True)
    import csv
    rows=[]
    for rnd,holds in [(0,'32/32'),(1,'29/32'),(2,'31/32'),(3,'31/32')]:
        for seed in ([0,1,2] if rnd==3 else [0]):
            model=_fake_run(tmp_path,'R8',rnd,seed)
            rows.append({'round':rnd,'seed':seed,'sustained_hold':holds,'model':model})
    with (d/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    models,missing=select_models(['R8'],[0,1,2],task_dir=tmp_path)
    assert len(models)==3 and not missing
    assert {m['round'] for m in models}=={3}
    assert {m['checkpoint_step'] for m in models}=={42000}
    (tmp_path/Path(rows[-1]['model']).parent/'step_42000.npz').unlink()
    models,missing=select_models(['R8'],[0,1,2],task_dir=tmp_path)
    assert len(models)==2 and len(missing)==1


def test_explicit_model_wrong_step_is_rejected(tmp_path):
    model=_fake_run(tmp_path,'R4',0,0,step=4000)
    p=tmp_path/Path(model).parent/'step_42000.npz'
    with pytest.raises(ValueError,match='42000'):
        select_models(None,None,[str(p)],task_dir=tmp_path)


def test_export_and_adjacent_config_must_describe_same_training_seed(tmp_path):
    model=_fake_run(tmp_path,'R4',0,0)
    config=tmp_path/Path(model).parent/'config.yaml'
    c=yaml.safe_load(config.read_text());c['identity']['seed']=999;config.write_text(yaml.safe_dump(c))
    # Export seed becomes mandatory for real metadata; absence is allowed only with consistent caller inference.
    with pytest.raises(ValueError,match='seed'):
        select_models(None,None,[str(tmp_path/Path(model).parent/'step_42000.npz')],task_dir=tmp_path)


@pytest.mark.realdata
def test_explicit_torch_terminal_checkpoint_uses_existing_adapter():
    m,_=select_models(['R4'],[0])
    pt=Path(m[0]['path']).with_suffix('.pt')
    models,missing=select_models(None,None,[str(pt)])
    assert not missing and models[0]['backend']=='torch'
    assert models[0]['id']==m[0]['id'] and models[0]['checkpoint_step']==42000
