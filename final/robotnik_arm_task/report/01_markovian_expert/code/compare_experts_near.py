import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/mpl-experts-near')
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
OUT=TASK/'results/diagnostics/2026-11-03__expert-a-b-near'
DISTANCES=[0.,.01,.03,.05,.1,.3,.5]

def scenario(distance,i):
    from data_collection.scenarios import Scenario
    rng=np.random.default_rng([20261103,4401,i])
    start=np.r_[rng.uniform(-.5,.5,2),0.]
    yaw,az=rng.uniform(-np.pi,np.pi,2)
    z=[.2,.5,.8,1.2][i%4]
    target=np.r_[start[:2]+distance*np.array([np.cos(yaw+az),np.sin(yaw+az)]),z]
    return Scenario(seed=2440100+i,base_start_pos=start,base_start_bearing=float(yaw),target_ee_pos=target,
                    scenario_id=f'near-comparison-{distance:g}-{i}',split='development')

def job(args):
    teacher,distance,i=args
    from core.env import RobotEnv
    from evaluation.evaluate import expert_factory,rollout
    policy=expert_factory(teacher)
    class TrackedEnv(RobotEnv):
        def reset(self,sc):
            self.previous=super().reset(sc)
            self.start_xy=self.previous.base_pos[:2].copy()
            self.stats={'base_path_m':0.,'reverse_command_s':0.,'forward_command_s':0.,'first_reach_gate_s':None,
                        'distance_at_first_gate_m':None,'final_base_target_distance_m':None}
            self.trace=[]
            return self.previous
        def step(self,action):
            prev=self.previous
            if teacher=='A':
                gate=policy.expert.base_command(prev)[1]
            else:
                gate=policy.teacher.base_state.done
            d=float(np.linalg.norm(prev.target_ee_pos[:2]-prev.base_pos[:2]))
            if gate and self.stats['first_reach_gate_s'] is None:
                self.stats['first_reach_gate_s']=self.t
                self.stats['distance_at_first_gate_m']=d
            v=.11*float(np.mean(action.wheel_cmds))
            self.stats['reverse_command_s']+=.001*int(v < -1e-3)
            self.stats['forward_command_s']+=.001*int(v > 1e-3)
            if round(self.t*1000)%50==0:
                self.trace.append({'time_s':self.t,'base_x':float(prev.base_pos[0]),'base_y':float(prev.base_pos[1]),
                                   'target_distance_m':d,'gate':int(gate),'commanded_linear_velocity_m_s':v,
                                   'ee_err_mm':float(np.linalg.norm(prev.target_ee_pos-self.ee_pos())*1000)})
            obs,info=super().step(action)
            self.stats['base_path_m']+=float(np.linalg.norm(obs.base_pos[:2]-prev.base_pos[:2]))
            self.stats['final_base_target_distance_m']=float(np.linalg.norm(obs.target_ee_pos[:2]-obs.base_pos[:2]))
            self.previous=obs
            return obs,info
    env=TrackedEnv(gui=False)
    base=yaml.safe_load((TASK/'experiments/configs/common.yaml').read_text())
    cfg={'evaluation':base['evaluation'],'interface':base['interface']}
    sc=scenario(distance,i)
    result=rollout(policy,sc,cfg,lambda:env)
    return {'teacher':teacher,'initial_distance_m':distance,'sample':i,'target_z_m':float(sc.target_ee_pos[2]),
            'start':sc.base_start_pos.tolist(),'yaw':sc.base_start_bearing,'target':sc.target_ee_pos.tolist(),
            **result,**env.stats,'trace':env.trace}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[(t,d,i) for d in DISTANCES for i in range(8) for t in ('A','B')]
    sources=['data_collection/expert.py','data_collection/template_teacher.py','robotnik_arm_control.py','evaluation/evaluate.py','core/env.py']
    (OUT/'run_manifest.json').write_text(json.dumps({'date':'2026-11-03','seed':[20261103,4401],'A_lead_time_s':.5,
        'cap_s':40,'n_contexts_per_distance':8,'distance_grid_m':DISTANCES,'heights_m':[.2,.5,.8,1.2],
        'paired_contexts_across_teachers_and_distances':True,
        'sources_sha256':{p:hashlib.sha256((TASK/p).read_bytes()).hexdigest() for p in sources}},indent=2))
    start=time.perf_counter();results=[]
    with mp.get_context('spawn').Pool(8) as pool:
        for r in pool.imap_unordered(job,jobs,chunksize=1):
            results.append(r)
            with (OUT/'rollouts.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if len(results)%16==0:print('PROGRESS',len(results),len(jobs),'wall_s',round(time.perf_counter()-start,1),flush=True)
    results.sort(key=lambda r:(r['initial_distance_m'],r['sample'],r['teacher']))
    (OUT/'rollouts.json').write_text(json.dumps(results,indent=2))
    summary=[]
    for d in DISTANCES:
        for t in ('A','B'):
            rr=[r for r in results if r['initial_distance_m']==d and r['teacher']==t]
            r={'teacher':t,'distance_m':d,'n':len(rr),'holds':sum(r['sustained_hold_pass'] for r in rr),
               'failures':sum(bool(r['failure']) for r in rr)}
            for key in ('first_arrival_t','t_first_reach','final_err_mm','final1s_max_err_mm','base_path_m','reverse_command_s','forward_command_s','first_reach_gate_s'):
                values=[x[key] for x in rr if x[key] is not None]
                r['median_'+key]=float(np.median(values)) if values else None
            r['episodes_with_reverse_commands']=sum(r['reverse_command_s']>.05 for r in rr)
            summary.append(r);print('RESULT',json.dumps(r),flush=True)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
