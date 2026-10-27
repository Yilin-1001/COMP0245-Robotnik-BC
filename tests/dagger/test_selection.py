import importlib.util


def module():
    assert importlib.util.find_spec("training.dagger.run"), "DAgger round scheduling is missing"
    from training.dagger import run
    return run


def summary(hold, reach, error):
    return {"sustained_hold_success": hold, "reach_success": reach, "median_final_err_mm": error}


def test_screening_uses_hold_then_reach_then_error_and_earliest_exact_tie():
    rows = [{"step": 5000, "summary": summary(.125,.25,20)},
            {"step": 10000, "summary": summary(.25,.25,1000)},
            {"step": 15000, "summary": summary(.25,.5,500)},
            {"step": 20000, "summary": summary(.25,.5,500)}]
    assert module().choose_candidate(rows)["step"] == 15000
    assert not module().promotes(summary(.25,.5,500), summary(.25,.5,500))
    assert module().promotes(summary(.25,.5,499), summary(.25,.5,500))
    assert not module().promotes(summary(.125,1.,1), summary(.25,.5,500))
