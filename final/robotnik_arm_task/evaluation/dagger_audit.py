"""Requery recorded current states and replay failed mixed prefixes before expert recovery."""
import os
if __name__ == "__main__":
    for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_key] = "1"
import argparse
import json
import numpy as np

from core.interfaces import Action, Observation, Policy
from data_collection.dagger import DEFAULT_OUTPUT, load_contract
from data_collection.dagger_preflight import ROOT, PARENT, sha
from data_collection.dagger_rollout import action_vector
from data_collection.dagger_scenarios import freeze_json
from data_collection.scenarios import to_scenario
from evaluation.evaluate import expert_factory, mlp_factory, rollout, make_robot_env


def audit_current_labels(arrays, target, expert, student, stride=500):
    T = len(arrays["expert_action"])
    indices = sorted(set(range(0,T,stride)) | ({T-1} if T else set()))
    expert.reset()
    student.reset()
    error = raw_error = 0.
    for k in indices:
        obs = Observation(arrays["obs_base_pos"][k],arrays["obs_base_yaw"][k],arrays["obs_q_mes"][k],arrays["obs_qd_mes"][k],target)
        label = action_vector(expert.act(obs))
        student.act(obs)
        error = max(error,float(np.max(np.abs(label-arrays["expert_action"][k]))))
        raw_error = max(raw_error,float(np.max(np.abs(student.last_raw-arrays["student_raw"][k]))))
    if error > 1e-9 or raw_error > 1e-8:
        raise ValueError(f"same-state label/student query mismatch: expert={error}, student={raw_error}")
    return {"queried_rows":len(indices),"expert_max_abs_difference":error,"student_raw_max_abs_difference":raw_error}


class ReplayThenExpert(Policy):
    def __init__(self, actions, prefix_steps):
        self.actions,self.prefix_steps = actions,int(prefix_steps)
        self.expert = expert_factory("A")

    def reset(self):
        self.k = 0
        self.expert.reset()

    def act(self, obs):
        expert = self.expert.act(obs)
        if self.k < self.prefix_steps:
            a = self.actions[self.k]
            action = Action(a[:7],a[7:])
        else:
            action = expert
        self.k += 1
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=__import__("pathlib").Path,default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = load_contract(args.output)
    record = json.loads((args.output/"mixed_collection.json").read_text())
    dest = ROOT/record["directory"]
    if sha(dest/"round_manifest.json") != record["manifest_sha256"]:
        raise ValueError("mixed manifest changed")
    manifest = json.loads((dest/"round_manifest.json").read_text())
    expert,student = expert_factory("A"),mlp_factory(str(ROOT/PARENT/"inference.npz"))
    from training.common.dataset import FeatureBuilder, OBS_FIELDS
    builder = FeatureBuilder("A",None)
    checks,failed,phase_mass = [],[],np.zeros(3)
    phase_rows = np.zeros(3,dtype=np.int64)
    torque_sat = wheel_sat = total = expert_steps = 0
    for entry in manifest["episodes"]:
        path = dest/entry["file"]
        if sha(path) != entry["sha256"]:
            raise ValueError("episode changed")
        with np.load(path) as z:
            arrays = {k:z[k] for k in z.files if k != "metadata_json"}
            meta = json.loads(str(z["metadata_json"]))
        row = meta["scenario"]
        target = np.array([row[k] for k in ("target_x","target_y","target_z")])
        check = audit_current_labels(arrays,target,expert,student)
        # Check deployed feature extraction independently at predeclared positions.
        idx = np.array(sorted({0,len(arrays["expert_action"])//2,len(arrays["expert_action"])-1}))
        x = builder({k:arrays[k][idx] for k in OBS_FIELDS},{"target_ee_pos_W":target})
        feature_error = 0.
        for j,k in enumerate(idx):
            obs = Observation(arrays["obs_base_pos"][k],arrays["obs_base_yaw"][k],arrays["obs_q_mes"][k],arrays["obs_qd_mes"][k],target)
            student.act(obs)
            feature_error = max(feature_error,float(np.max(np.abs(student.last_features-x[j]))))
        if feature_error > 1e-10:
            raise ValueError("dataset/deployment feature mismatch")
        check.update(scenario_id=entry["scenario_id"],feature_max_abs_difference=feature_error)
        checks.append(check)
        phase = arrays["phase"][arrays["valid_supervision"]]
        counts = np.bincount(phase,minlength=3)
        phase_rows += counts
        phase_mass += counts/max(1,len(phase))
        raw = arrays["student_raw"]
        torque_sat += int(np.any(np.abs(raw[:,:7]) >= 30,axis=1).sum())
        wheel_sat += int(np.any(np.abs(raw[:,7:]) >= 6,axis=1).sum())
        total += len(raw)
        expert_steps += int(arrays["expert_executes"].sum())
        if not entry["outcome"]["sustained_hold_pass"]:
            failed.append((row,arrays))
    from data_collection.dagger import parent_config
    recovery = []
    for row,arrays in failed[:2]:
        active = np.flatnonzero(arrays["phase"] >= 1)
        T = len(arrays["executed_action"])
        cuts = sorted({T//2,min(T-1,int(active[0])+1000) if len(active) else T//4})
        for cut in cuts:
            outcome = rollout(ReplayThenExpert(arrays["executed_action"],cut),to_scenario(row),parent_config(),make_robot_env)
            recovery.append({"scenario_id":row["scenario_id"],"replayed_steps":cut,"outcome":outcome})
            print(json.dumps({"recovery_prefix":row["scenario_id"],"steps":cut,"held":outcome["sustained_hold_pass"]}),flush=True)
    report = {"source_sha256":sha(__file__),"mixed_manifest_sha256":record["manifest_sha256"],"episodes":checks,
              "queried_rows":sum(c["queried_rows"] for c in checks),"recovery":recovery,
              "new_phase_rows":dict(zip(("navigation","approach","hold"),map(int,phase_rows))),
              "conditional_new_episode_phase_sampling_mass":dict(zip(("navigation","approach","hold"),(phase_mass/64).tolist())),
              "actual_expert_execution_fraction":expert_steps/total,"student_raw_torque_saturation_fraction":torque_sat/total,
              "student_raw_wheel_saturation_fraction":wheel_sat/total,"passed":True}
    freeze_json(args.output/"collection_audit.json",report)
    print(json.dumps({"audit_passed":True,"queried_rows":report["queried_rows"],"recovery_count":len(recovery)}),flush=True)


if __name__ == "__main__":
    main()
