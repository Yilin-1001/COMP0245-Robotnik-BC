"""Source validation, the frozen scenario split, raw feature caches and episode-uniform sampling.

Contracts (docs/mlp_baseline_training_plan.md §2 and §4, docs/clean_dataset_subsets.md):
- Every recorded episode has T actions and T+1 observations; feature row t uses observation t, label t is a_expert[t].
- Labels keep the 7 torques plus the right (FR) and left (FL) wheel speeds; FR == BR and FL == BL must hold exactly.
- Only the derived subset inventories are read (counts, scenarios_*.csv, training_manifest.csv,
  episode_provenance.csv). Training code never opens test/*.npz.
- Caches are float32, unnormalised, keyed by source hashes and feature definition, published atomically.

Usage (writes experiments/splits/clean253.json):
    python -m training.common.dataset --write-split
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[2]
TEACHER_VERSIONS = {"A": "markovian-lt0.50-v1", "B": "template-origin-goal-v1"}
ACTION_ORDER = "a = [torque_arm (7), wheel_cmds FR, FL, BL, BR (4)]"
CONTROL_HZ = 1000
LABEL_NAMES = [f"torque_{i}" for i in range(7)] + ["wheel_right", "wheel_left"]
PHASES = {"navigation": 0, "approach": 1, "hold": 2}
SPLIT_VERSION = "clean253-split-v1"
SPLIT_SEED = 20261026
SPLIT_RULE = ("validation: one scenario per cell of 2 u_target_dist bins x 4 u_target_z bins x 4 u_rel_azimuth "
              "sectors (floor(u*n)), cells in lexicographic (distance, height, azimuth) order, cell members sorted by "
              f"ID, index drawn with numpy default_rng({SPLIT_SEED}).integers(len(cell)); screening: lexicographically "
              "smallest validation ID in each of 8 equal u_rel_azimuth sectors; fitting: remaining training IDs")
EXPECTED_VALIDATION_IDS = [
    "train-0143", "train-0219", "train-0092", "train-0188", "train-0091", "train-0243", "train-0024", "train-0104",
    "train-0103", "train-0023", "train-0000", "train-0140", "train-0251", "train-0195", "train-0068", "train-0236",
    "train-0185", "train-0201", "train-0250", "train-0082", "train-0145", "train-0085", "train-0154", "train-0162",
    "train-0081", "train-0005", "train-0130", "train-0186", "train-0093", "train-0245", "train-0226", "train-0182"]
OBS_FIELDS = ("obs_time", "obs_base_pos", "obs_base_yaw", "obs_q_mes", "obs_qd_mes")


@dataclass
class EpisodeRef:
    path: Path
    scenario_id: str
    teacher: str
    sha256: str
    metadata: dict


@dataclass
class EpisodeCache:
    """x[N,D] raw features, y[N,9] raw labels, offsets[E+1]; diagnostics are never model inputs.

    episode_weights[E], when set, replaces uniform episode sampling (dataset mixes); None keeps the original stream.
    """
    x: np.ndarray
    y: np.ndarray
    offsets: np.ndarray
    episode_ids: list
    diagnostics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    episode_weights: np.ndarray | None = None

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_metadata(path: Path) -> dict:
    with np.load(path) as z:
        return json.loads(str(z["metadata_json"]))


# ---------------------------------------------------------------- source inventory

def load_manifest(root: Path, teacher: str) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("teacher_version") != TEACHER_VERSIONS[teacher]:
        raise ValueError(f"{root}: teacher version {manifest.get('teacher_version')!r} is not teacher {teacher}'s "
                         f"{TEACHER_VERSIONS[teacher]!r}")
    kind = str(manifest.get("dataset_kind", "")).lower()
    if "clean" not in kind or any(w in kind for w in ("dart", "dagger", "noise")):
        raise ValueError(f"{root}: dataset_kind {kind!r} is not a clean recording")
    return manifest


def manifest_interface(manifest: dict) -> dict:
    cc = manifest["collection_configuration"]
    return {"action_order": ACTION_ORDER, "torque_limit": float(cc["exec_torque_limit"]),
            "wheel_limit": float(cc["exec_wheel_limit"]), "control_hz": CONTROL_HZ}


def _safe_path(root: Path, split: str, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts or p.parts[:1] != (split,):
        raise ValueError(f"unsafe manifest path {rel!r}")
    full = (root / p).resolve()
    if full.parent != (root / split).resolve():
        raise ValueError(f"unsafe manifest path {rel!r}")
    return full


def _check_episode_metadata(meta: dict, teacher: str, split: str, scenario_id: str, interface: dict) -> None:
    if meta.get("teacher_version") != TEACHER_VERSIONS[teacher]:
        raise ValueError(f"{scenario_id}: teacher version {meta.get('teacher_version')!r}")
    if meta.get("scenario_id") != scenario_id or meta.get("split") != split:
        raise ValueError(f"{scenario_id}: episode metadata names {meta.get('scenario_id')}/{meta.get('split')}")
    if meta.get("source") != "expert_clean" or (meta.get("noise_configuration") or {}).get("enabled"):
        raise ValueError(f"{scenario_id}: not a clean expert recording (source {meta.get('source')!r})")
    ex = meta.get("execution_interface") or {}
    if ex.get("order") != ACTION_ORDER:
        raise ValueError(f"{scenario_id}: action order {ex.get('order')!r} differs from {ACTION_ORDER!r}")
    limits = ex.get("limits") or {}
    if (limits.get("torque_Nm"), limits.get("wheel_rad_s")) != (interface["torque_limit"], interface["wheel_limit"]):
        raise ValueError(f"{scenario_id}: execution limits {limits} differ from the manifest")
    if meta.get("control_hz") != CONTROL_HZ:
        raise ValueError(f"{scenario_id}: control_hz {meta.get('control_hz')}")


def load_episode_refs(root: Path, teacher: str, split: str) -> list[EpisodeRef]:
    """Validated references to one split's episodes, in training-manifest order (hashes checked)."""
    root = Path(root)
    manifest = load_manifest(root, teacher)
    interface = manifest_interface(manifest)
    scenarios = [r["scenario_id"] for r in _read_csv(root / f"scenarios_{split}.csv")]
    rows = [r for r in _read_csv(root / "training_manifest.csv") if r["split"] == split]
    prov = {r["episode_id"]: r for r in _read_csv(root / "episode_provenance.csv") if r["split"] == split}
    ids = [r["episode_id"] for r in rows]
    sids = [r["scenario_id"] for r in rows]
    if len(set(ids)) != len(ids) or len(set(sids)) != len(sids):
        raise ValueError(f"{root}: duplicate episode or scenario IDs in training_manifest.csv ({split})")
    expected = manifest["counts"][f"{split}_episodes"]
    if not (expected == len(rows) == len(scenarios) == len(prov)):
        raise ValueError(f"{root}: {split} count mismatch: manifest {expected}, training manifest {len(rows)}, "
                         f"scenarios {len(scenarios)}, provenance {len(prov)}")
    if sids != scenarios:
        raise ValueError(f"{root}: training_manifest.csv and scenarios_{split}.csv list different scenarios")
    refs = []
    for row in rows:
        if split == "train" and (row["eligible_initial_bc"] != "True" or row["integrity_status"] != "valid"):
            raise ValueError(f"{row['episode_id']}: not eligible for initial BC")
        p = prov.get(row["episode_id"])
        if p is None:
            raise ValueError(f"{row['episode_id']}: provenance row missing")
        path = _safe_path(root, split, p["file"])
        if _safe_path(root, split, row["file"]) != path:
            raise ValueError(f"{row['episode_id']}: training manifest and provenance disagree on the file")
        digest = sha256_file(path)
        if digest != p["sha256"]:
            raise ValueError(f"{path.name}: file hash {digest[:12]} != provenance {p['sha256'][:12]}")
        meta = _read_metadata(path)
        _check_episode_metadata(meta, teacher, split, row["scenario_id"], interface)
        refs.append(EpisodeRef(path=path, scenario_id=row["scenario_id"], teacher=teacher, sha256=digest, metadata=meta))
    return refs


# ---------------------------------------------------------------- split

def _cell(row: dict) -> tuple:
    b = lambda u, n: min(int(float(u) * n), n - 1)
    return b(row["u_target_dist"], 2), b(row["u_target_z"], 4), b(row["u_rel_azimuth"], 4)


def validation_ids_by_rule(rows: list[dict]) -> list[str]:
    cells = {}
    for r in rows:
        cells.setdefault(_cell(r), []).append(r["scenario_id"])
    if len(cells) != 32:
        raise ValueError(f"split rule needs 32 occupied cells, found {len(cells)}")
    rng = np.random.default_rng(SPLIT_SEED)
    out = []
    for key in sorted(cells):
        members = sorted(cells[key])
        out.append(members[int(rng.integers(len(members)))])
    return out


def screening_ids_by_rule(rows: list[dict], validation_ids: list[str]) -> list[str]:
    by_id = {r["scenario_id"]: r for r in rows}
    out = []
    for sector in range(8):
        members = sorted(i for i in validation_ids if min(int(float(by_id[i]["u_rel_azimuth"]) * 8), 7) == sector)
        if not members:
            raise ValueError(f"screening sector {sector} has no validation scenario")
        out.append(members[0])
    return out


def build_split(primary_roots: dict[str, Path], reduced_roots: dict[str, Path]) -> dict:
    """The frozen 221/32 (Clean_253) and 196/32 (Clean_223) split, from scenario inventories only."""
    hashes, inventories = {}, {}
    for cohort, roots in (("clean253", primary_roots), ("clean223", reduced_roots)):
        for teacher, root in roots.items():
            root = Path(root)
            load_manifest(root, teacher)
            for name in ("manifest.json", "scenarios_train.csv", "scenarios_test.csv", "training_manifest.csv",
                         "episode_provenance.csv"):
                hashes[f"{teacher}/{cohort}/{name}"] = sha256_file(root / name)
            inventories[(cohort, teacher)] = _read_csv(root / "scenarios_train.csv")
        if hashes[f"A/{cohort}/scenarios_train.csv"] != hashes[f"B/{cohort}/scenarios_train.csv"] or \
                hashes[f"A/{cohort}/scenarios_test.csv"] != hashes[f"B/{cohort}/scenarios_test.csv"]:
            raise ValueError(f"{cohort}: teachers A and B do not share identical scenario manifests")
    primary = inventories[("clean253", "A")]
    primary_ids = [r["scenario_id"] for r in primary]
    reduced_ids = [r["scenario_id"] for r in inventories[("clean223", "A")]]
    if not set(reduced_ids) <= set(primary_ids):
        raise ValueError("Clean_223 training scenarios are not a subset of Clean_253")
    validation = validation_ids_by_rule(primary)
    if validation != EXPECTED_VALIDATION_IDS:
        raise ValueError("split rule no longer reproduces the documented 32 validation IDs")
    val = set(validation)
    fitting = {"clean253": [i for i in primary_ids if i not in val], "clean223": [i for i in reduced_ids if i not in val]}
    by_id = {r["scenario_id"]: r for r in primary}
    return {"version": SPLIT_VERSION, "rule": SPLIT_RULE, "seed": SPLIT_SEED, "validation_ids": validation,
            "screening_ids": screening_ids_by_rule(primary, validation), "fitting_ids": fitting,
            "validation_source": "clean253", "validation_scenarios": [by_id[i] for i in validation],
            "test_scenarios_csv": "scenarios_test.csv (Clean_253, sealed until the frozen final protocol)",
            "source_sha256": hashes}


def write_split(path: Path, split: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(split, indent=1) + "\n")
    os.replace(tmp, path)
    return sha256_file(path)


def load_split(path: Path) -> dict:
    split = json.loads(Path(path).read_text())
    if split.get("version") != SPLIT_VERSION or split["validation_ids"] != EXPECTED_VALIDATION_IDS:
        raise ValueError(f"{path}: not the frozen {SPLIT_VERSION} split")
    return split


def select_refs(refs: list[EpisodeRef], ids: list[str]) -> list[EpisodeRef]:
    by_id = {r.scenario_id: r for r in refs}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"scenarios missing from the source inventory: {missing[:5]}")
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------- feature cache

class FeatureBuilder:
    """Deployable A/B features from the first T recorded observations (core.features is shared with inference).

    definition names a core.feature_sets entry; features-v1 keeps its original builder_id, so existing cache IDs hold.
    """

    def __init__(self, teacher: str, task_context: dict | None = None, definition: str | None = None):
        from core.feature_sets import DEFAULT_DEFINITION, feature_names
        self.teacher, self.task_context = teacher, task_context
        self.definition = definition or DEFAULT_DEFINITION
        self._fk = None
        from core.features import FEATURE_DEFINITION_VERSION, FK_CONFIG
        self.feature_names = feature_names(teacher, self.definition)
        code = hashlib.sha256((TASK_DIR / "core" / "features.py").read_bytes()).hexdigest()
        key = {"teacher": teacher, "features": FEATURE_DEFINITION_VERSION, "fk": FK_CONFIG,
               "task_context": task_context, "features_py_sha256": code}
        if self.definition != DEFAULT_DEFINITION:
            key.update(definition=self.definition, feature_sets_py_sha256=hashlib.sha256(
                (TASK_DIR / "core" / "feature_sets.py").read_bytes()).hexdigest())
        self.builder_id = json.dumps(key, sort_keys=True)

    def __call__(self, obs: dict, meta: dict) -> np.ndarray:
        from core.feature_sets import extend
        from core.features import DeployableFK, features_a, features_b
        if self._fk is None:
            self._fk = DeployableFK()
        target = np.asarray(meta["target_ee_pos_W"], dtype=float)
        args = (obs["obs_base_pos"], obs["obs_base_yaw"], obs["obs_q_mes"], obs["obs_qd_mes"])
        if self.teacher == "A":
            return extend("A", self.definition, features_a(*args, target, self._fk))
        ctx = meta["teacher_initial_state"]
        if float(ctx["time_scale_s"]) != float(self.task_context["time_scale_s"]):
            raise ValueError(f"{meta['scenario_id']}: time scale {ctx['time_scale_s']} != {self.task_context}")
        return features_b(*args, obs["obs_time"], target, np.asarray(ctx["base_goal_pos_W"], dtype=float),
                          float(ctx["base_goal_bearing"]), float(ctx["time_scale_s"]), self._fk)


def _episode_rows(ref: EpisodeRef, feature_builder):
    with np.load(ref.path) as z:
        a = np.asarray(z["a_expert"], dtype=float)
        T = len(a)
        obs = {k: np.asarray(z[k]) for k in OBS_FIELDS}
        active_key = "nav_done_after" if ref.teacher == "B" else "teacher_reach_active"
        active = np.asarray(z[active_key], dtype=bool) if active_key in z.files else np.asarray(
            z["teacher_reach_active"], dtype=bool)
        ee_err = np.asarray(z["truth_ee_err"], dtype=float)
    if any(len(v) != T + 1 for v in obs.values()) or len(ee_err) != T + 1 or len(active) != T or \
            ref.metadata["outcome"]["T_actions"] != T:
        raise ValueError(f"{ref.scenario_id}: expected T+1 observations for T={T} actions")
    if not (np.isfinite(a).all() and all(np.isfinite(v).all() for v in obs.values())):
        raise ValueError(f"{ref.scenario_id}: non-finite observation or label")
    if not (np.array_equal(a[:, 7], a[:, 10]) and np.array_equal(a[:, 8], a[:, 9])):
        raise ValueError(f"{ref.scenario_id}: same-side wheel labels differ (FR != BR or FL != BL); wrong action order?")
    first_T = {k: v[:T] for k, v in obs.items()}
    x = np.asarray(feature_builder(first_T, ref.metadata), dtype=float)
    if x.shape[0] != T or not np.isfinite(x).all():
        raise ValueError(f"{ref.scenario_id}: feature builder returned {x.shape} or non-finite values for T={T}")
    y = np.c_[a[:, :7], a[:, 7], a[:, 8]]
    reach = ref.metadata["outcome"].get("first_reach_step")
    phase = np.where(active, PHASES["approach"], PHASES["navigation"]).astype(np.int8)
    if reach is not None:
        phase[int(reach):] = PHASES["hold"]
    diag = {"t": obs["obs_time"][:T].astype(np.float32), "phase": phase, "ee_err": ee_err[:T].astype(np.float32)}
    return x.astype(np.float32), y.astype(np.float32), diag


def _array_hashes(directory: Path, names) -> dict:
    return {n: sha256_file(directory / n) for n in names}


CACHE_FILES = ("x.npy", "y.npy", "offsets.npy", "diagnostics.npz", "metadata.json")


def cache_id(refs: list[EpisodeRef], builder_id: str) -> str:
    key = json.dumps({"sources": [[r.scenario_id, r.sha256] for r in refs], "builder": builder_id,
                      "labels": LABEL_NAMES}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def build_cache(refs: list[EpisodeRef], feature_builder, destination: Path, workers: int = 0) -> EpisodeCache:
    """Build (or reuse a complete, matching) float32 cache at destination; published by atomic rename."""
    destination = Path(destination)
    if (destination / "COMPLETE.json").exists():
        return load_cache(destination, refs)
    if destination.exists():
        raise ValueError(f"{destination}: incomplete cache directory; remove it to rebuild")
    if workers and workers > 1 and len(refs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.starmap(_episode_rows, [(r, feature_builder) for r in refs])
    else:
        parts = [_episode_rows(r, feature_builder) for r in refs]
    lengths = [len(p[0]) for p in parts]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    diagnostics = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    names = getattr(feature_builder, "feature_names", None)
    metadata = {"episode_ids": [r.scenario_id for r in refs], "source_sha256": [r.sha256 for r in refs],
                "teacher": refs[0].teacher, "feature_builder": getattr(feature_builder, "builder_id", repr(feature_builder)),
                "feature_names": list(names) if names is not None else [f"x{i}" for i in range(x.shape[1])],
                "label_names": LABEL_NAMES, "n_rows": int(len(x)), "input_dim": int(x.shape[1]),
                "alignment": "feature row t = observation t of the first T observations; label t = a_expert[t]"}
    tmp = destination.parent / f".tmp-{destination.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "x.npy", x)
    np.save(tmp / "y.npy", y)
    np.save(tmp / "offsets.npy", offsets)
    np.savez(tmp / "diagnostics.npz", **diagnostics)
    (tmp / "metadata.json").write_text(json.dumps(metadata))
    (tmp / "COMPLETE.json").write_text(json.dumps({"files": _array_hashes(tmp, CACHE_FILES)}))
    os.replace(tmp, destination)
    return EpisodeCache(x, y, offsets, metadata["episode_ids"], diagnostics, metadata)


def load_cache(destination: Path, refs: list[EpisodeRef]) -> EpisodeCache:
    destination = Path(destination)
    done = destination / "COMPLETE.json"
    if not done.exists():
        raise ValueError(f"{destination}: incomplete cache (no completion manifest)")
    recorded = json.loads(done.read_text())["files"]
    if _array_hashes(destination, CACHE_FILES) != recorded:
        raise ValueError(f"{destination}: cache files were tampered with (hash mismatch)")
    metadata = json.loads((destination / "metadata.json").read_text())
    if metadata["source_sha256"] != [r.sha256 for r in refs] or metadata["episode_ids"] != [r.scenario_id for r in refs]:
        raise ValueError(f"{destination}: stale cache for a different source inventory")
    with np.load(destination / "diagnostics.npz") as z:
        diagnostics = {k: z[k] for k in z.files}
    return EpisodeCache(np.load(destination / "x.npy"), np.load(destination / "y.npy"),
                        np.load(destination / "offsets.npy"), metadata["episode_ids"], diagnostics, metadata)


def subset_cache(cache: EpisodeCache, ids: list[str]) -> EpisodeCache:
    """Episodes of an existing cache, in the given order (used for e.g. the one-episode sanity run)."""
    pos = {e: i for i, e in enumerate(cache.episode_ids)}
    rows = [np.arange(cache.offsets[pos[i]], cache.offsets[pos[i] + 1]) for i in ids]
    idx = np.concatenate(rows)
    offsets = np.concatenate(([0], np.cumsum([len(r) for r in rows]))).astype(np.int64)
    meta = dict(cache.metadata, episode_ids=list(ids),
                source_sha256=[cache.metadata["source_sha256"][pos[i]] for i in ids], n_rows=int(len(idx)))
    return EpisodeCache(cache.x[idx], cache.y[idx], offsets, list(ids),
                        {k: v[idx] for k, v in cache.diagnostics.items()}, meta)


# ---------------------------------------------------------------- sampling

def sample_indices(cache: EpisodeCache, size: int, generator, device):
    """Row indices: an episode uniformly (or by cache.episode_weights), then a timestep uniformly within it."""
    import torch
    offsets = torch.as_tensor(cache.offsets, device=device)
    lengths = offsets[1:] - offsets[:-1]
    if cache.episode_weights is None:
        e = torch.randint(len(lengths), (size,), generator=generator, device=device)
    else:
        w = torch.as_tensor(cache.episode_weights, dtype=torch.float64, device=device)
        e = torch.multinomial(w, size, replacement=True, generator=generator)
    u = torch.rand(size, generator=generator, device=device)
    step = torch.minimum((u * lengths[e]).long(), lengths[e] - 1)
    return offsets[:-1][e] + step


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-split", action="store_true", help="materialise experiments/splits/clean253.json")
    args = parser.parse_args(argv)
    if args.write_split:
        data = TASK_DIR / "data"
        split = build_split({"A": data / "Markovian/Clean_253", "B": data / "Stateful/Clean_253"},
                            {"A": data / "Markovian/Clean_223", "B": data / "Stateful/Clean_223"})
        digest = write_split(TASK_DIR / "experiments/splits/clean253.json", split)
        print(f"wrote experiments/splits/clean253.json sha256={digest}: {len(split['validation_ids'])} validation, "
              f"{len(split['fitting_ids']['clean253'])}/{len(split['fitting_ids']['clean223'])} fitting, "
              f"screening {split['screening_ids']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
