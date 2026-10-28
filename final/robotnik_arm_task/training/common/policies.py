"""Deployable MLP policy: deployable features -> saved normaliser -> network -> inverse scaling -> limits.

The NumPy backend loads an explicit .npz export and never imports torch; the torch backend loads a .pt checkpoint.
For teacher B the fixed navigation goal is computed once per episode from the target with the saved geometric
rule, and elapsed time advances by one control period per action; reset() clears both. No expert runs here.
"""
import json
from pathlib import Path

import numpy as np

from core.interfaces import Action, Observation, Policy


class _NumpyNet:
    def __init__(self, z, meta: dict):
        n = len(meta["model"]["hidden"])
        self.eps = float(meta["model"]["layernorm_eps"])
        self.layers = [(z[f"layer{i}_W"].T.copy(), z[f"layer{i}_b"], z[f"ln{i}_gamma"], z[f"ln{i}_beta"]) for i in range(n)]
        self.head_W = np.concatenate((z["torque_W"], z["wheel_W"])).T.copy()
        self.head_b = np.concatenate((z["torque_b"], z["wheel_b"]))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x
        for W, b, g, beta in self.layers:
            h = h @ W + b
            mu = h.mean(axis=-1, keepdims=True)
            var = ((h - mu) ** 2).mean(axis=-1, keepdims=True)  # population variance, as torch LayerNorm
            h = (h - mu) / np.sqrt(var + self.eps) * g + beta
            h = h / (1.0 + np.exp(-h))
        return h @ self.head_W + self.head_b


class _TorchNet:
    def __init__(self, payload: dict):
        import torch
        from training.common.models import MLP
        spec = payload["meta"]["model"]
        self.torch = torch
        self.model = MLP(spec["input_dim"], spec["hidden"], spec["layernorm_eps"])
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.as_tensor(x, dtype=self.torch.float32)).double().numpy()


class MLPPolicy(Policy):
    name = "mlp"

    def __init__(self, net, normaliser: dict, meta: dict, backend: str, artifact: str):
        self.net, self.meta, self.backend, self.artifact = net, meta, backend, artifact
        self.x_mean, self.x_scale = np.asarray(normaliser["x_mean"]), np.asarray(normaliser["x_scale"])
        self.y_mean, self.y_scale = np.asarray(normaliser["y_mean"]), np.asarray(normaliser["y_scale"])
        self.teacher, self.task_context = meta["teacher"], meta.get("task_context")
        # Exports predating feature definitions are features-v1.
        self.feature_definition = meta.get("feature_definition") or "features-v1"
        interface = meta["interface"]
        self.torque_limit, self.wheel_limit = float(interface["torque_limit"]), float(interface["wheel_limit"])
        self.dt = 1.0 / float(interface["control_hz"])
        if len(self.x_mean) != meta["input_dim"] or (self.teacher == "B") != (self.task_context is not None):
            raise ValueError("policy metadata is inconsistent: input dimension or task context")
        self._fk = None
        self.reset()

    @classmethod
    def load(cls, ckpt_path: str, backend: str = "numpy", expected: dict | None = None,
             strict_features: bool = True) -> "MLPPolicy":
        path = Path(ckpt_path)
        if backend == "numpy":
            if path.suffix != ".npz":
                raise ValueError(f"the NumPy backend loads .npz exports, not {path.name}")
            with np.load(path) as z:
                meta = json.loads(str(z["metadata_json"]))
                if meta.get("format") != "mlp-numpy-export-v1":
                    raise ValueError(f"{path}: not an MLP NumPy export")
                net = _NumpyNet(z, meta)
        elif backend == "torch":
            if path.suffix != ".pt":
                raise ValueError(f"the torch backend loads .pt checkpoints, not {path.name}")
            from training.common.checkpoint import load_checkpoint
            payload = load_checkpoint(path)
            meta = dict(payload["meta"], normaliser=payload["normaliser"])
            net = _TorchNet(payload)
        else:
            raise ValueError(f"backend must be numpy or torch, got {backend!r}")
        if strict_features:
            from core.feature_sets import feature_names
            definition = meta.get("feature_definition") or "features-v1"
            if list(meta.get("feature_names") or []) != feature_names(meta["teacher"], definition):
                raise ValueError(f"{path}: saved feature order differs from the deployable features of teacher "
                                 f"{meta['teacher']} ({definition})")
        for field, value in (expected or {}).items():
            if meta.get(field) != value:
                raise ValueError(f"policy {field} mismatch: stored {meta.get(field)!r}, expected {value!r}")
        return cls(net, meta["normaliser"], meta, backend, str(path))

    def reset(self) -> None:
        self._k = 0
        self._goal = None
        self.last_features = self.last_raw = None

    def features(self, obs: Observation) -> np.ndarray:
        from core.feature_sets import extend
        from core.features import DeployableFK, features_a, features_b, stateful_goal
        if self._fk is None:
            self._fk = DeployableFK()
        args = (np.asarray(obs.base_pos, dtype=float)[None], np.array([obs.base_bearing], dtype=float),
                np.asarray(obs.q_mes, dtype=float)[None], np.asarray(obs.qd_mes, dtype=float)[None])
        target = np.asarray(obs.target_ee_pos, dtype=float)
        if self.teacher == "A":
            return extend("A", self.feature_definition, features_a(*args, target, self._fk))[0]
        if self._goal is None:
            self._goal = stateful_goal(target, self.task_context)
        elapsed = np.array([self._k * self.dt])
        return features_b(*args, elapsed, target, self._goal[0], self._goal[1],
                          float(self.task_context["time_scale_s"]), self._fk)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Physical 9-dimensional prediction (7 torques, right, left wheel) before execution limits."""
        return self.net((np.asarray(x, dtype=float) - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean

    def act(self, obs: Observation) -> Action:
        x = self.features(obs)
        raw = self.predict(x)
        self._k += 1
        self.last_features, self.last_raw = x, raw
        torque = np.clip(raw[:7], -self.torque_limit, self.torque_limit)
        right, left = np.clip(raw[7:9], -self.wheel_limit, self.wheel_limit)
        return Action(torque_arm=torque, wheel_cmds=np.array([right, left, left, right]))
