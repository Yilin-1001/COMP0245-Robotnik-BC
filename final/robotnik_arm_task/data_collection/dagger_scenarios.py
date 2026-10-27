"""Balanced nominal scenes and immutable JSON artifacts, independent of legacy collection settings."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def freeze_json(path, value):
    path = Path(path)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"frozen content differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(encoded)
        os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_scenes(seed, per_cell, split):
    from scipy.stats import qmc
    from data_collection.scenarios import DIMENSIONS
    if per_cell <= 0 or per_cell % 2 or split not in ("dagger_train", "dagger_test"):
        raise ValueError("scenes need an even positive count per cell and a DAgger split")
    ranges = [(0.5, 3.), (-np.pi, np.pi), (.1, 1.25), (-.5, .5), (-.5, .5), (-np.pi, np.pi)]
    bins = {(d, h, a): [] for d in range(2) for h in range(4) for a in range(4)}
    for i, u in enumerate(qmc.Sobol(6, scramble=True, seed=seed).random_base2(12)):
        cell = (int(u[0]*2), int(u[2]*4), int(u[1]*4))
        if len(bins[cell]) >= per_cell:
            continue
        row = {d: float(lo + u[j]*(hi-lo)) for j, (d, (lo, hi)) in enumerate(zip(DIMENSIONS, ranges))}
        row.update({f"u_{d}": float(u[j]) for j, d in enumerate(DIMENSIONS)})
        heading = row["start_yaw"] + row["rel_azimuth"]
        row.update(target_x=float(row["start_x"] + row["target_dist"]*np.cos(heading)),
                   target_y=float(row["start_y"] + row["target_dist"]*np.sin(heading)),
                   scenario_id=f"{split}-{i:04d}", split=split, index=i,
                   seed=int(seed*10000+i), cell=list(cell), round=len(bins[cell])//2+1)
        bins[cell].append(row)
    if any(len(v) != per_cell for v in bins.values()):
        raise ValueError("Sobol pool did not fill all cells")
    rows = [r for cell in sorted(bins) for r in bins[cell]]
    return sorted(rows, key=lambda r: (r["round"], r["cell"], r["index"]))


def assert_disjoint(new, existing):
    """Compare physical geometry, including canonical yaw, not just names or Sobol indices."""
    def geometry(r):
        yaw = (float(r["start_yaw"]) + np.pi) % (2*np.pi) - np.pi
        return np.array([float(r[k]) for k in ("target_x", "target_y", "target_z", "start_x", "start_y")] + [yaw])
    seen = [geometry(r) for r in existing]
    ids = {r["scenario_id"] for r in existing}
    for row in new:
        g = geometry(row)
        if row["scenario_id"] in ids or any(np.allclose(g, old, atol=1e-9, rtol=0) for old in seen):
            raise ValueError(f"duplicate scene identity or geometry: {row['scenario_id']}")
        ids.add(row["scenario_id"])
        seen.append(g)
