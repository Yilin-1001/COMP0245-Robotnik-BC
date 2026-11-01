import json
import pytest
from evaluation.test_suite import parser, job_identity, save_job, load_job


def test_cli_selects_rows_seeds_suites_without_changing_scene_seed():
    args=parser().parse_args(['--rows','R4','R8','--seeds','0','--suites','T0','T1'])
    assert args.rows==['R4','R8'] and args.seeds==[0] and args.suites==['T0','T1']


def test_atomic_result_identity_and_integrity_rejects_stale_and_corrupt_jobs(tmp_path):
    identity=job_identity('run-hash','R4-s0','test-0001')
    path=tmp_path/'job.json'
    assert load_job(path,identity) is None
    save_job(path,identity,{'success':True,'grade':'A','completion_time_s':3.,'final1s_max_err_mm':1.,'scenario_id':'test-0001','model_id':'R4-s0','failure':None,'history':[]})
    assert load_job(path,identity)['success']
    with pytest.raises(ValueError,match='identity'):
        load_job(path,job_identity('different-code','R4-s0','test-0001'))
    body=json.loads(path.read_text());body['record']['success']=False;path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='integrity'):load_job(path,identity)
    assert not list(tmp_path.glob('*.tmp'))


def test_resume_rejects_mutated_top_level_selection(tmp_path):
    from evaluation.test_suite import prepare,code_hashes,PROTOCOL
    from evaluation.suite_manifest import freeze_manifest,digest
    manifest=freeze_manifest(tmp_path/'scene_manifest.json',{'scenarios':[{'scenario_id':'a'}]})
    model={'kind':'expert','id':'Markovian-expert-A'}
    identity={'code_sha256':code_hashes(),'manifest_sha256':manifest['sha256'],'models':[model],
              'scenario_ids':['a'],'protocol':PROTOCOL,'interface':{'control_hz':1000,'torque_limit':30.,'wheel_limit':6.}}
    run={'identity':identity,'run_hash':digest(identity),'scene_manifest_sha256':manifest['sha256'],
         'models':[],'scenario_ids':['a'],'expected_jobs':1}
    (tmp_path/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='selection'):
        prepare(parser().parse_args(['--resume',str(tmp_path)]))
