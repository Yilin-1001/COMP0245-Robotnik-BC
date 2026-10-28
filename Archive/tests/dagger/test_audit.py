import importlib.util
import numpy as np
import pytest

from test_collection import run, Controller


def test_requery_audit_detects_a_shifted_expert_label():
    assert importlib.util.find_spec("evaluation.dagger_audit"), "DAgger label audit is missing"
    from evaluation.dagger_audit import audit_current_labels
    arrays,_ = run(0.)
    result = audit_current_labels(arrays,np.array([1.,0.,0.]),Controller(10),Controller(1),stride=1)
    assert result["queried_rows"] == 23
    assert result["expert_max_abs_difference"] == 0
    arrays["expert_action"][1,0] += 1
    with pytest.raises(ValueError,match="label"):
        audit_current_labels(arrays,np.array([1.,0.,0.]),Controller(10),Controller(1),stride=1)
