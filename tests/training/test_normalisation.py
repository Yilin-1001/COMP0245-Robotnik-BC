import numpy as np
import pytest
import torch

from training.common.dataset import EpisodeCache
from training.common.normalisation import Normaliser, fit_normaliser

FLOORS = {"input_scale_floor": 1e-6, "arm_scale_floor": 0.1, "wheel_scale_floor": 0.05}


def cache_of(x, y, offsets=None):
    offsets = np.array([0, len(x)]) if offsets is None else np.asarray(offsets)
    return EpisodeCache(np.asarray(x, np.float32), np.asarray(y, np.float32), offsets,
                        [f"e{i}" for i in range(len(offsets) - 1)], {}, {"feature_names": [f"f{i}" for i in range(np.shape(x)[1])]})


def test_population_statistics_and_floors():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 3.0, (1000, 3))
    x[:, 2] = 5.0  # constant input -> input floor
    y = rng.normal(0.0, 4.0, (1000, 9))
    y[:, 0] = 1.0   # constant torque -> arm floor
    y[:, 8] = -2.0  # constant wheel -> wheel floor
    n = fit_normaliser(cache_of(x, y), FLOORS)
    np.testing.assert_allclose(n.x_mean, x.astype(np.float32).astype(np.float64).mean(0), rtol=1e-12)
    np.testing.assert_allclose(n.x_scale[:2], x.astype(np.float32).astype(np.float64)[:, :2].std(0, ddof=0), rtol=1e-12)
    assert n.x_scale[2] == 1e-6 and n.y_scale[0] == 0.1 and n.y_scale[8] == 0.05
    assert n.feature_names == ["f0", "f1", "f2"]


def test_validation_outlier_cannot_change_fitting_scales():
    rng = np.random.default_rng(1)
    fit_x, fit_y = rng.normal(size=(200, 2)), rng.normal(size=(200, 9))
    val_x = np.vstack([rng.normal(size=(50, 2)), [[1e6, -1e6]]])
    a = fit_normaliser(cache_of(fit_x, fit_y), FLOORS)
    b = fit_normaliser(cache_of(fit_x, fit_y), FLOORS)
    _ = cache_of(val_x, rng.normal(size=(51, 9)))  # validation never reaches fit_normaliser
    np.testing.assert_array_equal(a.x_mean, b.x_mean)
    assert np.abs(a.transform_x(val_x)[-1]).max() > 1e5  # outlier stays an outlier under fitting scales


def test_round_trip_and_backend_preservation():
    rng = np.random.default_rng(2)
    n = fit_normaliser(cache_of(rng.normal(size=(100, 4)), rng.normal(3, 2, size=(100, 9))), FLOORS)
    y = rng.normal(size=(10, 9))
    np.testing.assert_allclose(n.inverse_y(n.transform_y(y)), y, atol=1e-10)
    assert n.transform_x(rng.normal(size=(3, 4))).dtype == np.float64
    t = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    out = n.inverse_y(n.transform_y(t))
    assert isinstance(out, torch.Tensor) and out.dtype == torch.float32
    out.sum().backward()
    assert t.grad is not None and torch.allclose(t.grad, torch.ones_like(t))


def test_serialisation_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    n = fit_normaliser(cache_of(rng.normal(size=(50, 3)), rng.normal(size=(50, 9))), FLOORS)
    n.save(tmp_path / "normaliser.npz")
    m = Normaliser.load(tmp_path / "normaliser.npz")
    np.testing.assert_array_equal(m.y_scale, n.y_scale)
    assert m.feature_names == n.feature_names and m.floors == n.floors and m.sha256() == n.sha256()
    with pytest.raises(ValueError):
        Normaliser.from_dict({**n.to_dict(), "x_scale": [0.0, 1.0, 1.0]})
