import numpy as np
from evaluation.suite_report import summarise
from evaluation.suite_plots import plot_results


def records(teacher_success=True):
    rows=[]
    for pol,seed in [('R1-s0',0),('R1-s1',1),('Markovian-expert-A',None)]:
        for i,sid in enumerate(['a','b']):
            success=(i==seed) if seed is not None else (i==0 and teacher_success)
            rows.append({'model_id':pol,'row':'R1' if seed is not None else 'expert','seed':seed,
                         'scenario_id':sid,'suite':'T0','band':'core','success':success,
                         'grade':'A' if success else 'D','completion_time_s':5.+i if success else None,
                         'final1s_max_err_mm':2. if success else 30.,'duration_s':40.,
                         'target_dist_m':1.+i,'target_z_m':.5,'target_bx':1.+i,'target_by':0.,
                         'failure':None,'latency_p50_us':50.})
    return rows


def test_cluster_interval_and_expert_subset_keep_same_scenes_across_seeds():
    summary=summarise(records())
    r=next(x for x in summary if x['level']=='row' and x['policy']=='R1' and x['subset']=='all')
    assert r['n']==4 and r['scene_n']==2 and r['successes']==2 and r['success_rate']==.5
    assert r['ci_low']==.5 and r['ci_high']==.5  # each scene is 1/2 across seeds
    assert r['completion_median_s']==5.5
    sub=next(x for x in summary if x['level']=='row' and x['policy']=='R1' and x['subset']=='expert_success')
    assert sub['n']==2 and sub['success_rate']==.5


def test_expert_zero_success_is_unavailable_not_zero():
    summary=summarise(records(False))
    sub=[x for x in summary if x['subset']=='expert_success']
    assert sub and all(x['n']==0 and x['success_rate'] is None for x in sub)


def test_all_failure_figures_are_created_without_finite_errors(tmp_path):
    rows=records(False)
    for r in rows:r.update(success=False,grade='F',final1s_max_err_mm=None,completion_time_s=None)
    plot_results(tmp_path,rows,summarise(rows))
    png=list((tmp_path/'plots').glob('*.png'))
    assert len(png)>=7
    assert len(list((tmp_path/'plots').glob('*.pdf')))==len(png)


def test_copied_job_files_cannot_create_fake_complete_coverage(tmp_path):
    import json
    import pytest
    from evaluation.suite_manifest import digest,freeze_manifest
    from evaluation.test_suite import save_job,job_identity
    from evaluation.suite_report import load_records
    manifest=freeze_manifest(tmp_path/'scene_manifest.json',{'scenarios':[{'scenario_id':'a'},{'scenario_id':'b'}]})
    identity={'models':[{'id':'R1-s0'}],'scenario_ids':['a','b'],'manifest_sha256':manifest['sha256']}
    rh=digest(identity)
    (tmp_path/'run.json').write_text(json.dumps({'run_hash':rh,'identity':identity,'expected_jobs':2,'models':[{'id':'R1-s0'}],
        'scenario_ids':['a','b'],'scene_manifest_sha256':manifest['sha256']}))
    row=records()[0]
    save_job(tmp_path/'jobs/first.json',job_identity(rh,'R1-s0','a'),row)
    (tmp_path/'jobs/copy.json').write_bytes((tmp_path/'jobs/first.json').read_bytes())
    with pytest.raises(ValueError,match='duplicate'):
        load_records(tmp_path)


def test_partial_results_keep_expected_denominator_and_pending_expert_visible():
    rows=records()[:1]
    models=[{'id':'R1-s0','row':'R1','seed':0},{'id':'Markovian-expert-A','row':'expert','seed':None}]
    scenes=[{'scenario_id':sid,'suite':'T0','band':'core'} for sid in ['a','b']]
    summary=summarise(rows,models,scenes)
    r=next(x for x in summary if x['level']=='model' and x['policy']=='R1-s0' and x['subset']=='all')
    assert r['n_expected']==2 and r['n']==1 and r['n_missing']==1
    assert r['coverage_status']=='partial' and r['ci_low'] is None
    sub=next(x for x in summary if x['policy']=='R1-s0' and x['subset']=='expert_success')
    assert sub['expert_pending_scenes']==2 and sub['success_rate'] is None
