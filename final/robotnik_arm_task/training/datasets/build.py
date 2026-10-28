"""Publish training-ready feature datasets: raw teacher-A episodes x one feature definition -> data/datasets/A/<name>/.

A dataset is the training cache made durable and documented: x[N,D] features, y[N,9] labels (7 torques, right and
left wheel), offsets[E+1], diagnostics (never inputs), episodes.csv, metadata.json, qa.json, README.md and
COMPLETE.json (file hashes). Raw episodes are never copied or changed; every row traces back to a source file by SHA-256.

Sources:
  clean253                      data/Markovian/Clean_253, all 253 training episodes (fitting 221 + validation 32);
                                the frozen split picks fitting and validation rows at training time. Test is excluded.
  <path to a Clean_* subset>    the same, for another clean subset (requires --name)
  <path to a DART collection>   every sustained-hold episode of data/Markovian/DART/<collection>/ (requires --name)
Dataset name: <source name>__<feature definition>, e.g. clean253__features-v1-gate, dart-d2s075x2__features-v1.

QA (qa.json; a failed check aborts before publishing): finite features and labels, T+1 observations per T actions,
equal same-side wheel labels, source hashes, reach_gate equal to the teacher's recorded reach flag on every row, the
first 21 columns equal to the sibling features-v1 dataset when it exists, and no validation scenario in DART data.

Usage (from final/robotnik_arm_task/):
    python -m training.datasets.build --source clean253 --features features-v1-gate
    python -m training.datasets.build --source data/Markovian/DART/<collection> --name dart-d2s075x2 --features features-v1
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from training.common.dataset import (ACTION_ORDER, CONTROL_HZ, LABEL_NAMES, TEACHER_VERSIONS, EpisodeRef, FeatureBuilder,
                                     _episode_rows, _read_metadata, load_episode_refs, load_manifest, load_split,
                                     manifest_interface, sha256_file)

TASK_DIR = Path(__file__).resolve().parents[2]
ALIASES = {"clean253": "data/Markovian/Clean_253"}
DATA_FILES = ("x.npy", "y.npy", "offsets.npy", "diagnostics.npz", "episodes.csv", "metadata.json", "qa.json", "README.md")
SPLIT = "experiments/splits/clean253.json"


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(TASK_DIR)) if p.is_relative_to(TASK_DIR) else str(p)


def clean_sources(root: Path) -> tuple[list[EpisodeRef], dict, dict]:
    manifest = load_manifest(root, "A")
    refs = load_episode_refs(root, "A", "train")
    return refs, manifest_interface(manifest), {"kind": "clean", "dataset_kind": manifest["dataset_kind"]}


def dart_sources(root: Path) -> tuple[list[EpisodeRef], dict, dict]:
    """Hash-checked sustained-hold episodes of one DART collection, in episodes.csv order."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("dataset_kind") != "dart" or manifest.get("teacher_version") != TEACHER_VERSIONS["A"]:
        raise ValueError(f"{root}: not a teacher-A DART collection")
    cc = manifest["collection_configuration"]
    interface = {"action_order": ACTION_ORDER, "torque_limit": float(cc["exec_torque_limit"]),
                 "wheel_limit": float(cc["exec_wheel_limit"]), "control_hz": CONTROL_HZ}
    with open(root / "episodes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    refs, excluded = [], []
    for row in rows:
        path = (root / row["file"]).resolve()
        if path.parent != (root / "train").resolve():
            raise ValueError(f"unsafe episode path {row['file']!r}")
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(f"{path.name}: file hash {digest[:12]} != episodes.csv {row['sha256'][:12]}")
        meta = _read_metadata(path)
        if meta.get("source") != "expert_dart" or not meta["noise_configuration"].get("enabled") or \
                meta.get("teacher_version") != TEACHER_VERSIONS["A"] or meta["execution_interface"]["order"] != ACTION_ORDER \
                or meta.get("control_hz") != CONTROL_HZ:
            raise ValueError(f"{path.name}: metadata is not a teacher-A DART episode")
        if not meta["outcome"]["sustained_hold_pass"]:
            excluded.append({"episode_id": meta["episode_id"], "category": meta["outcome"]["category"]})
            continue
        refs.append(EpisodeRef(path=path, scenario_id=meta["scenario_id"], teacher="A", sha256=digest, metadata=meta))
    return refs, interface, {"kind": "dart", "dataset_kind": "dart", "protocol": manifest["protocol"],
                             "code_digest": manifest["code_digest"], "excluded_episodes": excluded}


def _rows(ref: EpisodeRef, builder: FeatureBuilder):
    x, y, diag = _episode_rows(ref, builder)
    with np.load(ref.path) as z:
        active = np.asarray(z["teacher_reach_active"], dtype=bool)
    return x, y, diag, active


def build(source: str, definition: str, name: str | None = None, out_root: Path | None = None, workers: int = 16) -> Path:
    from core.feature_sets import GATE_RULE, feature_names
    src = Path(ALIASES.get(source, source))
    src = src if src.is_absolute() else TASK_DIR / src
    prefix = name or (source if source in ALIASES else None)
    if prefix is None:
        raise ValueError("--name is required when --source is a path")
    if "__" in prefix:
        raise ValueError("a source name may not contain '__' (it separates the source from the feature definition)")
    out_root = Path(out_root or TASK_DIR / "data/datasets") / "A"
    dest = out_root / f"{prefix}__{definition}"
    kind = json.loads((src / "manifest.json").read_text()).get("dataset_kind", "")
    refs, interface, source_info = (dart_sources if kind == "dart" else clean_sources)(src)
    builder = FeatureBuilder("A", None, definition)
    names = feature_names("A", definition)
    key = hashlib.sha256(json.dumps({"builder": builder.builder_id, "sources": [[r.metadata["episode_id"], r.sha256] for r in refs],
                                     "build_py_sha256": sha256_file(Path(__file__))}, sort_keys=True).encode()).hexdigest()
    if (dest / "COMPLETE.json").exists():
        if json.loads((dest / "metadata.json").read_text())["build_key"] == key:
            print(f"{_rel(dest)} is up to date")
            return dest
        raise FileExistsError(f"{dest} was built from different sources or code; remove it to rebuild")
    if dest.exists():
        raise FileExistsError(f"{dest}: incomplete dataset directory; remove it to rebuild")

    if workers > 1 and len(refs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.starmap(_rows, [(r, builder) for r in refs])
    else:
        parts = [_rows(r, builder) for r in refs]
    lengths = [len(p[0]) for p in parts]
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    active = np.concatenate([p[3] for p in parts])
    diagnostics = {k: np.concatenate([p[2][k] for p in parts]) for k in parts[0][2]}
    episodes = [{"episode_id": r.metadata["episode_id"], "scenario_id": r.scenario_id, "draw": r.metadata.get("dart_draw", ""),
                 "row_start": int(lo), "row_end": int(hi), "rows": int(hi - lo), "source_file": _rel(r.path), "sha256": r.sha256}
                for r, lo, hi in zip(refs, offsets[:-1], offsets[1:])]

    qa = {"episodes": len(refs), "rows": int(len(x)), "finite": bool(np.isfinite(x).all() and np.isfinite(y).all()),
          "feature_dim": int(x.shape[1]), "expected_dim": len(names), "source_hashes": "verified",
          "observation_alignment": "T+1 observations per T actions, checked per episode"}
    if "reach_gate" in names:
        mismatch = int(np.sum(x[:, names.index("reach_gate")].astype(bool) != active))
        qa["reach_gate_vs_teacher_flag"] = {"mismatched_rows": mismatch, "active_fraction": float(active.mean())}
    sibling = out_root / f"{prefix}__features-v1"
    if definition != "features-v1" and (sibling / "COMPLETE.json").exists():
        same = np.array_equal(np.load(sibling / "offsets.npy"), offsets) and \
            np.array_equal(np.load(sibling / "x.npy", mmap_mode="r"), x[:, :len(feature_names("A", "features-v1"))])
        qa["v1_prefix_equal_to_sibling"] = {"sibling": sibling.name, "equal": bool(same)}
    else:
        qa["v1_prefix_equal_to_sibling"] = "not checked (no features-v1 sibling yet)"
    if source_info["kind"] == "dart":
        split = load_split(TASK_DIR / SPLIT)
        leaked = sorted({r.scenario_id for r in refs} & set(split["validation_ids"]))
        qa["validation_scenarios_present"] = leaked
    failed = [k for k, ok in (("finite", qa["finite"]), ("feature_dim", qa["feature_dim"] == qa["expected_dim"]),
                              ("reach_gate", qa.get("reach_gate_vs_teacher_flag", {}).get("mismatched_rows", 0) == 0),
                              ("v1_prefix", not isinstance(qa["v1_prefix_equal_to_sibling"], dict)
                               or qa["v1_prefix_equal_to_sibling"]["equal"]),
                              ("validation_leak", not qa.get("validation_scenarios_present"))) if not ok]
    qa["passed"] = not failed
    if failed:
        raise ValueError(f"dataset QA failed: {failed}: {json.dumps(qa)}")

    metadata = {"name": dest.name, "teacher": "A", "feature_definition": definition, "feature_names": names,
                "input_dim": len(names), "label_names": LABEL_NAMES,
                "label_units": "torque N*m (7), wheel speed rad/s (right = FR = BR, left = FL = BL)",
                "alignment": "feature row t = pre-action observation t; label t = clean teacher action a_expert[t]",
                "interface": interface, "source": {"root": _rel(src), **source_info},
                "builder_id": builder.builder_id, "build_key": key,
                **({"gate_rule": GATE_RULE} if "reach_gate" in names else {}),
                "counts": {"episodes": len(refs), "rows": int(len(x)), "scenarios": len({r.scenario_id for r in refs})},
                "split_note": "clean sources keep fitting and validation episodes; training selects rows with "
                              f"{SPLIT}. DART sources hold fitting scenarios only.",
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    tmp = out_root / f".tmp-{dest.name}-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "x.npy", x)
    np.save(tmp / "y.npy", y)
    np.save(tmp / "offsets.npy", offsets)
    np.savez(tmp / "diagnostics.npz", **diagnostics)
    with open(tmp / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(episodes[0]))
        w.writeheader()
        w.writerows(episodes)
    (tmp / "metadata.json").write_text(json.dumps(metadata, indent=1))
    (tmp / "qa.json").write_text(json.dumps(qa, indent=1))
    (tmp / "README.md").write_text(readme(dest.name, metadata, source, definition, name))
    (tmp / "COMPLETE.json").write_text(json.dumps({"files": {n: sha256_file(tmp / n) for n in DATA_FILES}}, indent=1))
    os.replace(tmp, dest)
    return dest


def readme(dataset: str, meta: dict, source: str, definition: str, name: str | None) -> str:
    c = meta["counts"]
    cmd = f"python -m training.datasets.build --source {source}" + (f" --name {name}" if name else "") + f" --features {definition}"
    return (f"# {dataset}\n\nTeacher A, feature definition `{definition}` ({meta['input_dim']} inputs), built from "
            f"`{meta['source']['root']}` ({meta['source']['kind']}). {c['episodes']} episodes, {c['rows']:,} rows, "
            f"{c['scenarios']} scenarios.\n\n"
            "| File | Content |\n|---|---|\n"
            "| `x.npy` | float32 [rows, inputs]: deployable features, column order in metadata.json `feature_names` |\n"
            "| `y.npy` | float32 [rows, 9]: clean teacher action (7 torques, right wheel, left wheel) |\n"
            "| `offsets.npy` | int64 [episodes + 1]: row range of each episode |\n"
            "| `diagnostics.npz` | t, phase (0 navigation, 1 approach, 2 hold), true EE error: analysis only, never inputs |\n"
            "| `episodes.csv` | episode and scenario IDs, row range, source file and its SHA-256 |\n"
            "| `metadata.json` | definitions, interface, source, gate rule, build key |\n"
            "| `qa.json` | checks run before publishing |\n\n"
            f"Rebuild (from `final/robotnik_arm_task/`): `{cmd}`\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="clean253 or a path to a Clean_* subset or DART collection")
    parser.add_argument("--features", required=True, help="feature definition, e.g. features-v1 or features-v1-gate")
    parser.add_argument("--name", help="source name in the dataset name (required for path sources)")
    parser.add_argument("--out-root", help="default data/datasets")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    dest = build(args.source, args.features, args.name, Path(args.out_root) if args.out_root else None, args.workers)
    print(json.dumps({"dataset": _rel(dest), **json.loads((dest / "qa.json").read_text())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
