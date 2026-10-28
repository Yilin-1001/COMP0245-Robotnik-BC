import importlib.util
import pytest


def test_learning_curves_use_matched_fixed_probes_and_reject_duplicate_updates():
    assert importlib.util.find_spec("evaluation.dagger_report"), "DAgger learning-curve reporting is missing"
    from evaluation.dagger_report import curve_series
    def row(u,a,b):
        return {"update":u,"fitting":{"overall":{"action_huber":a,"arm_rmse_nm":2*a,"wheel_rmse_rad_s":3*a}},
                "validation":{"overall":{"action_huber":b,"arm_rmse_nm":2*b,"wheel_rmse_rad_s":3*b}}}
    rows = [row(0,1.,2.),row(5000,.5,3.)]
    series = curve_series(rows)
    assert series["updates"] == [0,5000]
    assert series["fitting"]["action_huber"] == [1.,.5]
    assert series["validation"]["arm_rmse_nm"] == [4.,6.]
    with pytest.raises(ValueError):
        curve_series(rows+[rows[-1]])


def test_sampling_audit_replays_saved_batch_rng_and_counts_actual_gradient_rows(tmp_path, unit_normaliser):
    import numpy as np
    import torch
    from training.common.engine import fit, init_model
    from training.baseline.train import objective_from
    from training.dagger.config import validate
    from test_training import tiny, cache
    from evaluation import dagger_report
    assert hasattr(dagger_report,"sampling_audit"), "actual gradient sampling audit is missing"
    cfg = tiny(tmp_path,unit_normaliser)
    c = cache()
    r = fit(cfg,c,None,init_model(cfg),unit_normaliser,objective_from(cfg),validator=validate)
    state = torch.load(r.run_dir/"latest.pt",map_location="cpu",weights_only=True)
    result = dagger_report.sampling_audit(cfg,c.offsets,np.full(24,2),0,state["rng"]["gen_batch"],device="cpu")
    assert result["source_phase_counts"] == [[0,0,0],[0,0,32]]
    assert result["rng_matches_checkpoint"]
    with pytest.raises(ValueError,match="RNG"):
        dagger_report.sampling_audit(cfg,c.offsets,np.full(24,2),0,torch.Generator().manual_seed(99).get_state(),device="cpu")
