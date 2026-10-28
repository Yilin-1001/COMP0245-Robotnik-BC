"""Load published feature datasets (training.datasets.build) as EpisodeCaches, select scenarios and mix datasets."""
import csv
import json
from pathlib import Path

import numpy as np

from training.common.dataset import EpisodeCache, sha256_file

TASK_DIR = Path(__file__).resolve().parents[2]


def dataset_dir(name: str, root: Path | str | None = None, teacher: str = "A") -> Path:
    root = Path(root or "data/datasets")
    return (root if root.is_absolute() else TASK_DIR / root) / teacher / name


def load_dataset(name: str, root: Path | str | None = None) -> EpisodeCache:
    """A complete, hash-checked dataset; metadata gains scenario_ids, source_sha256 and complete_sha256."""
    d = dataset_dir(name, root)
    done = d / "COMPLETE.json"
    if not done.exists():
        raise FileNotFoundError(f"{d}: no complete dataset (build it with training.datasets.build)")
    recorded = json.loads(done.read_text())["files"]
    if {n: sha256_file(d / n) for n in recorded} != recorded:
        raise ValueError(f"{d}: dataset files changed since publishing (hash mismatch)")
    meta = json.loads((d / "metadata.json").read_text())
    with open(d / "episodes.csv", newline="") as f:
        episodes = list(csv.DictReader(f))
    meta.update(scenario_ids=[e["scenario_id"] for e in episodes], source_sha256=[e["sha256"] for e in episodes],
                complete_sha256=sha256_file(done))
    with np.load(d / "diagnostics.npz") as z:
        diagnostics = {k: z[k] for k in z.files}
    return EpisodeCache(np.load(d / "x.npy"), np.load(d / "y.npy"), np.load(d / "offsets.npy"),
                        [e["episode_id"] for e in episodes], diagnostics, meta)


def select_scenarios(cache: EpisodeCache, scenario_ids) -> EpisodeCache:
    """Episodes whose scenario is in scenario_ids, in dataset order (every draw of a scenario is kept)."""
    keep = set(scenario_ids)
    chosen = [i for i, s in enumerate(cache.metadata["scenario_ids"]) if s in keep]
    rows = [np.arange(cache.offsets[i], cache.offsets[i + 1]) for i in chosen]
    idx = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int64)
    pick = lambda key: [cache.metadata[key][i] for i in chosen]
    meta = dict(cache.metadata, scenario_ids=pick("scenario_ids"), source_sha256=pick("source_sha256"))
    return EpisodeCache(cache.x[idx], cache.y[idx], offsets, [cache.episode_ids[i] for i in chosen],
                        {k: v[idx] for k, v in cache.diagnostics.items()}, meta)


def concat(caches: list[EpisodeCache], shares: list[float] | None = None) -> EpisodeCache:
    """One cache from several; shares[k] is dataset k's probability per sampled episode (None: every episode alike)."""
    names = [c.metadata["feature_names"] for c in caches]
    if any(n != names[0] for n in names):
        raise ValueError("datasets in one mix must share the feature definition and column order")
    offsets = [caches[0].offsets]
    for c in caches[1:]:
        offsets.append(c.offsets[1:] + offsets[-1][-1])
    weights = None
    if shares is not None:
        if len(shares) != len(caches) or any(s <= 0 for s in shares) or not np.isclose(sum(shares), 1.0):
            raise ValueError("mix shares must be positive, one per dataset, and sum to 1")
        weights = np.concatenate([np.full(len(c.episode_ids), s / len(c.episode_ids)) for c, s in zip(caches, shares)])
    meta = {"feature_names": names[0], "label_names": caches[0].metadata["label_names"],
            "datasets": [c.metadata["name"] for c in caches],
            "scenario_ids": [s for c in caches for s in c.metadata["scenario_ids"]],
            "source_sha256": [h for c in caches for h in c.metadata["source_sha256"]]}
    return EpisodeCache(np.concatenate([c.x for c in caches]), np.concatenate([c.y for c in caches]),
                        np.concatenate(offsets), [e for c in caches for e in c.episode_ids],
                        {k: np.concatenate([c.diagnostics[k] for c in caches]) for k in caches[0].diagnostics},
                        meta, weights)
