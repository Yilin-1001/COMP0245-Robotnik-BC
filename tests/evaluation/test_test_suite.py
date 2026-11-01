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
    save_job(path,identity,{'success':True,'history':[]})
    assert load_job(path,identity)['success']
    with pytest.raises(ValueError,match='identity'):
        load_job(path,job_identity('different-code','R4-s0','test-0001'))
    body=json.loads(path.read_text());body['record']['success']=False;path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='integrity'):load_job(path,identity)
    assert not list(tmp_path.glob('*.tmp'))
