import pytest
import torch

from training.common.models import MLP, expand_wheels

COUNTS = {  # docs/mlp_baseline_training_plan.md §5 (7 torque + 2 wheel heads)
    (64, 64): (6409, 6729), (128, 128): (21001, 21641), (256, 256): (74761, 76041),
    (128, 128, 128): (37769, 38409), (256, 256, 256): (141065, 142345), (256, 128, 64): (48265, 49545)}


@pytest.mark.parametrize("hidden", list(COUNTS))
def test_parameter_counts_match_plan(hidden):
    for dim, expected in zip((21, 26), COUNTS[hidden]):
        m = MLP(dim, list(hidden))
        assert sum(p.numel() for p in m.parameters()) == expected


def test_structure_and_output_shape():
    m = MLP(21, [128, 128], layernorm_eps=1e-5)
    kinds = [type(l).__name__ for l in m.trunk]
    assert kinds == ["Linear", "LayerNorm", "SiLU"] * 2
    assert all(l.eps == 1e-5 and l.elementwise_affine for l in m.trunk if isinstance(l, torch.nn.LayerNorm))
    assert not any(isinstance(l, torch.nn.Dropout) for l in m.modules())
    assert m(torch.zeros(5, 21)).shape == (5, 9)


def test_wheels_expand_to_right_left_left_right():
    y = torch.arange(18.0).reshape(2, 9)
    out = expand_wheels(y)
    assert out.shape == (2, 11)
    assert torch.equal(out[:, 7], y[:, 7]) and torch.equal(out[:, 10], y[:, 7])
    assert torch.equal(out[:, 8], y[:, 8]) and torch.equal(out[:, 9], y[:, 8])
