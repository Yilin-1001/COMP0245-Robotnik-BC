import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl-expert-boundaries')
import copy
import csv
import functools
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
import numpy as np
import yaml

TASK=Path('/media/yilin/data/ML_Final/COMP_0245_2026_Final/final/robotnik_arm_task')
sys.path.insert(0,str(TASK))
OUT=TASK/'results/diagnostics/2026-11-03__expert-a-b-boundaries'

def run_job(job):
    teacher,sweep,value,i,extended=job
    from evaluation.expert_workspace import scene
    from evaluation.evaluate import expert_factory, make_robot_env, rollout
    base=yaml.safe_load((TASK/'experiments/configs/common.yaml').read_text())
    cfg={'evaluation':copy.deepcopy(base['evaluation']),'interface':base['interface']}
    cfg['evaluation']['episode_cap_s']=80. if sweep=='far' else 40.
    policy=expert_factory(teacher)
    if extended:
        original=policy.act
        def act(obs):
            action=original(obs)
            policy.teacher.base_params.nav_timeout=float('inf')
            return action
        policy.act=act
    scenario=scene(sweep,value,i)
    result=rollout(policy,scenario,cfg,functools.partial(make_robot_env,cfg['evaluation']['reach_tolerance_m']))
    return {'teacher':teacher,'extended_B_timeout':extended,'sweep':sweep,'value':value,'sample':i,
            'cap_s':cfg['evaluation']['episode_cap_s'],'target':scenario.target_ee_pos.tolist(),
            'start':scenario.base_start_pos.tolist(),'yaw':scenario.base_start_bearing,**result}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(t,s,v,i,False) for s,values in [('height',[1.31,1.32,1.33]),('far',[6.5,7.,7.5])] for v in values for i in range(8) for t in ('A','B')]
    jobs += [('B','far',v,i,True) for v in (8.,10.) for i in range(8)]
    start=time.perf_counter()
    sources=['data_collection/expert.py','data_collection/template_teacher.py','robotnik_arm_control.py','evaluation/evaluate.py','evaluation/expert_workspace.py','core/env.py']
    manifest={'date':'2026-11-03','A_lead_time_s':.5,'B_original_nav_timeout_s':25.,'supplement_n_per_cell':8,
              'extended_B_is_counterfactual':True,'jobs':jobs,
              'sources_sha256':{p:hashlib.sha256((TASK/p).read_bytes()).hexdigest() for p in sources}}
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    rows=[]
    with mp.get_context('spawn').Pool(8) as pool:
        for result in pool.imap_unordered(run_job,jobs,chunksize=1):
            rows.append(result)
            with (OUT/'supplement_rollouts.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if len(rows)%8==0: print(f'PROGRESS {len(rows)}/{len(jobs)}, wall {time.perf_counter()-start:.1f}s',flush=True)
    rows.sort(key=lambda r:(r['extended_B_timeout'],r['sweep'],r['value'],r['sample'],r['teacher']))
    (OUT/'supplement_rollouts.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for extended,sweep,value,teacher in sorted({(r['extended_B_timeout'],r['sweep'],r['value'],r['teacher']) for r in rows}):
        rr=[r for r in rows if (r['extended_B_timeout'],r['sweep'],r['value'],r['teacher'])==(extended,sweep,value,teacher)]
        reach=[r['t_first_reach'] for r in rr if r['t_first_reach'] is not None]
        entry={'teacher':teacher,'extended_B_timeout':extended,'sweep':sweep,'value':value,'n':len(rr),
               'reach':sum(r['ever_reached'] for r in rr),'hold':sum(r['sustained_hold_pass'] for r in rr),
               'median_final_err_mm':float(np.median([r['final_err_mm'] for r in rr])),
               'median_reach_t_s':float(np.median(reach)) if reach else None,
               'failures':sum(r['failure'] is not None for r in rr)}
        summary.append(entry);print('RESULT',json.dumps(entry),flush=True)
    (OUT/'supplement_summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',round(time.perf_counter()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
