import pytest
import yaml

from experiments.run import BlockedError, build_jobs, rank_offline, rank_rollout, reuse_allowed, write_frozen_protocol

ARCHS = ["m1", "m2", "m3", "m4", "m5", "m6"]


@pytest.fixture
def frozen_protocol():
    return {"protocol_id": "test", "decisions": {
        "budget": {"updates": 30000, "pilot_lr": 3.0e-4, "plateau_update": 20000},
        "loss": {"action": "huber"},
        "lr": {t: {a: 1.0e-3 for a in ARCHS} for t in ("A", "B")},
        "finalists": {"A": ["m2", "m4"], "B": ["m3", "m2"]},
        "selected": {"A": "m2", "B": "m3"},
        "calibration": {"A": "checkpoints/calibration/A/m2/calibration-aaa.json",
                        "B": "checkpoints/calibration/B/m3/calibration-bbb.json"}}}


def test_pilot_is_two_constant_lr_m2_a_huber_runs():
    jobs = build_jobs("pilot", {"protocol_id": "none", "decisions": {}})
    assert len(jobs) == 2
    assert {j["overrides"]["training.lr"] for j in jobs} == {1e-3, 3e-4}
    for j in jobs:
        assert j["identity"]["teacher"] == "A" and j["identity"]["architecture"] == "m2"
        assert j["overrides"]["training.schedule"] == "constant" and j["overrides"]["training.updates"] == 50000
        assert j["overrides"]["loss.action"] == "huber" and j["overrides"]["training.log_every"] == 1000


def test_loss_stage_needs_a_budget_decision(frozen_protocol):
    with pytest.raises(BlockedError, match="budget"):
        build_jobs("loss", {"protocol_id": "x", "decisions": {}})
    jobs = build_jobs("loss", frozen_protocol)
    assert len(jobs) == 2 and {j["overrides"]["loss.action"] for j in jobs} == {"huber", "mse"}
    assert all(j["overrides"]["training.updates"] == 30000 and j["overrides"]["training.lr"] == 3e-4 and
               j["overrides"]["training.schedule"] == "cosine" and j["identity"]["seed"] == 0 for j in jobs)


def test_architecture_grid_is_offline_only(frozen_protocol):
    jobs = build_jobs("screen", frozen_protocol)
    assert len(jobs) == 6 * 2 * 3
    assert all(j["identity"]["regime"] == "clean" for j in jobs)
    assert all(j["identity"]["objective"] == "bc" for j in jobs)
    assert {(j["identity"]["teacher"], j["identity"]["architecture"], j["overrides"]["training.lr"]) for j in jobs} == \
        {(t, a, lr) for t in "AB" for a in ARCHS for lr in (1e-3, 3e-4, 1e-4)}
    assert all(j["identity"]["seed"] == 0 and j["overrides"]["loss.action"] == "huber" for j in jobs)


def test_finalist_seeds_add_eight_fits(frozen_protocol):
    jobs = build_jobs("seeds", frozen_protocol)
    assert len(jobs) == 8 and {j["identity"]["seed"] for j in jobs} == {1, 2}
    assert {(j["identity"]["teacher"], j["identity"]["architecture"]) for j in jobs} == \
        {("A", "m2"), ("A", "m4"), ("B", "m3"), ("B", "m2")}


def test_jacobian_pairs_three_seeds_per_selected_architecture(frozen_protocol):
    jobs = build_jobs("jacobian", frozen_protocol)
    assert len(jobs) == 6 and all(j["launcher"] == "jacobian" and j["identity"]["objective"] == "bcj" for j in jobs)
    assert {(j["identity"]["teacher"], j["identity"]["architecture"], j["identity"]["seed"]) for j in jobs} == \
        {(t, a, s) for t, a in (("A", "m2"), ("B", "m3")) for s in (0, 1, 2)}
    a0 = next(j for j in jobs if j["identity"]["teacher"] == "A" and j["identity"]["seed"] == 0)
    assert a0["overrides"]["loss.jacobian.calibration"].endswith("calibration-aaa.json")
    assert a0["overrides"]["training.lr"] == 1e-3  # the plain-BC pair's LR, budget and schedule


def test_no_automatic_test_access_or_collection_jobs(frozen_protocol):
    with pytest.raises(BlockedError, match="final protocol"):
        build_jobs("test", frozen_protocol)
    for stage in ("dart", "dagger"):
        with pytest.raises(ValueError):
            build_jobs(stage, frozen_protocol)
    for stage in ("pilot", "loss", "screen", "seeds", "jacobian"):
        assert all(j["identity"]["regime"] == "clean" for j in build_jobs(stage, frozen_protocol))


def test_reuse_requires_identical_full_config_hash():
    a = {"identity": {"config_hash": "1" * 64}}
    assert reuse_allowed(a, {"identity": {"config_hash": "1" * 64}})
    assert not reuse_allowed(a, {"identity": {"config_hash": "2" * 64}})


def test_frozen_protocols_are_immutable(tmp_path):
    ev = tmp_path / "metrics.csv"
    ev.write_text("update,lr\n1000,0.001\n")
    p1 = write_frozen_protocol(tmp_path, {"budget": {"updates": 30000}}, [ev])
    saved = yaml.safe_load(p1.read_text())
    assert saved["decisions"]["budget"]["updates"] == 30000 and len(saved["evidence"]) == 1
    with pytest.raises(FileExistsError):
        write_frozen_protocol(tmp_path, {"budget": {"updates": 30000}}, [ev])
    p2 = write_frozen_protocol(tmp_path, {"budget": {"updates": 45000}}, [ev], parent=saved["protocol_id"])
    assert p2 != p1 and yaml.safe_load(p2.read_text())["parent"] == saved["protocol_id"]


def test_offline_ranking_is_lexicographic():
    gate_fail = {"hold_bias": {"gate_pass": False, "max_abs_nm": 0.6}, "arm_rmse_nm": 0.1, "wheel_rmse_rad_s": 0.01}
    good = {"hold_bias": {"gate_pass": True, "max_abs_nm": 0.4}, "arm_rmse_nm": 0.9, "wheel_rmse_rad_s": 0.5}
    better_bias = {"hold_bias": {"gate_pass": True, "max_abs_nm": 0.2}, "arm_rmse_nm": 2.0, "wheel_rmse_rad_s": 0.5}
    ranked = sorted([gate_fail, good, better_bias], key=rank_offline)
    assert ranked == [better_bias, good, gate_fail]


def test_rollout_ranking_prefers_success_then_error_then_earlier_update():
    base = {"sustained_hold_success": 0.75, "reach_success": 1.0, "median_final_err_mm": 3.0,
            "mean_saturation_frac": 0.0, "mean_torque_variation_nm": 0.1, "checkpoint_step": 4000}
    more_hold = dict(base, sustained_hold_success=0.875, median_final_err_mm=8.0)
    later_same = dict(base, checkpoint_step=6000)
    ranked = sorted([later_same, base, more_hold], key=rank_rollout)
    assert ranked == [more_hold, base, later_same]


def test_tampered_or_unknown_protocols_are_rejected(tmp_path):
    from experiments.run import load_protocol
    ev = tmp_path / "e.csv"
    ev.write_text("x\n")
    path = write_frozen_protocol(tmp_path, {"budget": {"updates": 30000, "pilot_lr": 3e-4}}, [ev])
    assert load_protocol(path)["decisions"]["budget"]["updates"] == 30000
    text = path.read_text().replace("30000", "10000")
    path.write_text(text)
    with pytest.raises(ValueError, match="tampered"):
        load_protocol(path)
    bad = write_frozen_protocol(tmp_path, {"budgett": {"updates": 1}}, [ev])
    with pytest.raises(ValueError, match="unknown decision"):
        load_protocol(bad)
