"""GatedMLP (R5 split networks): branch selection, gradient isolation, export parity and unchanged plain specs."""
import numpy as np
import pytest
import torch

from training.common.checkpoint import _mlp_arrays, model_spec
from training.common.models import MLP, GatedMLP, build_model, model_from_spec
from training.common.policies import _numpy_net


def gated(seed=0, d=22, hidden=(16, 16)):
    torch.manual_seed(seed)
    return GatedMLP(d, list(hidden), gate_index=d - 1)


def rows(n=64, d=22, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    x[: n // 2, -1] = -0.9   # standardised gate 0 (navigation)
    x[n // 2:, -1] = 1.1     # standardised gate 1 (reaching)
    return x


def test_sign_of_gate_column_selects_branch():
    m, x = gated(), rows()
    y = m(x)
    assert torch.allclose(y[:32], m.nav(x[:32]))
    assert torch.allclose(y[32:], m.reach(x[32:]))


def test_each_branch_only_learns_from_its_rows():
    m, x = gated(), rows()
    m(x[:32]).sum().backward()  # navigation rows only
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.nav.parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.reach.parameters())


def test_velocity_jacobian_works_through_the_switch():
    from evaluation.offline import velocity_jacobian
    m, x = gated(), rows()
    J = velocity_jacobian(m, x)
    assert J.shape == (64, 7, 7)
    assert torch.allclose(J[:32], velocity_jacobian(m.nav, x[:32]), atol=1e-6)
    assert torch.allclose(J[32:], velocity_jacobian(m.reach, x[32:]), atol=1e-6)


def test_numpy_export_matches_torch_and_spec_round_trips():
    m, x = gated(), rows()
    spec = model_spec(m)
    assert spec["type"] == "gated" and spec["gate_index"] == 21
    arrays = {**_mlp_arrays(m.nav, "nav_"), **_mlp_arrays(m.reach, "reach_")}
    net = _numpy_net(arrays, {"model": spec})
    with torch.no_grad():
        ref = m(x).double().numpy()
    assert np.allclose(net(x.double().numpy()), ref, atol=1e-5)
    rebuilt = model_from_spec(spec)
    rebuilt.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert torch.allclose(rebuilt(x), m(x))


def test_plain_mlp_spec_is_unchanged():
    assert model_spec(MLP(21, [8])) == {"input_dim": 21, "hidden": [8], "layernorm_eps": 1e-5}


def test_build_model_dispatch():
    base = {"model": {"input_dim": 22, "architectures": {"MLP01": [8]}, "layernorm_eps": 1e-5},
            "identity": {"architecture": "MLP01"}, "data": {"feature_definition": "features-v1-gate"}}
    assert type(build_model(base)) is MLP
    base["model"]["type"] = "gated"
    m = build_model(base)
    assert isinstance(m, GatedMLP) and m.gate_index == 21
    base["data"]["feature_definition"] = "features-v1"
    with pytest.raises(ValueError):
        build_model(base)
