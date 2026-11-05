"""Verify the robustness run in ../data/run and write ../data/run/verification.json.

Checks (no simulation):
  1. every policy x scene appears exactly once (16 policies x 108 scenes = 1,728 rollouts)
  2. each evaluated .npz still has the SHA-256 recorded in the run manifest and in checkpoints/2026-11-05final/SHA256SUMS
  3. the 108 scenes regenerate identically from evaluation.robust_probe.scenes()
  4. summary.json's hold counts reproduce from rollouts.csv
  5. the source files hashed in the manifest are unchanged in the working tree

Usage (from final/robotnik_arm_task/):
    python report/10_robustness/code/verify_run.py
"""
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parents[1] / "data/run"
sys.path.insert(0, str(TASK))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from evaluation.robust_probe import CONDITIONS, scenes
    rows = list(csv.DictReader(open(RUN / "rollouts.csv")))
    manifest = json.loads((RUN / "manifest.json").read_text())
    n = manifest["n_per_condition"]
    checks = {}

    keys = collections.Counter((r["policy"], r["condition"], r["scenario_id"]) for r in rows)
    policies = {r["policy"] for r in rows}
    checks["1_complete"] = {"policies": len(policies), "rollouts": len(rows),
                            "expected": manifest["expected_rollouts"], "duplicates": sum(v > 1 for v in keys.values()),
                            "ok": len(rows) == manifest["expected_rollouts"] and all(v == 1 for v in keys.values())}

    sums = {}
    for line in (TASK / "checkpoints/2026-11-05final/SHA256SUMS").read_text().splitlines():
        if line.strip():
            h, p = line.split(maxsplit=1)
            sums[p.lstrip("*")] = h
    bad = [m["run_id"] for m in manifest["models"]
           if not (sha(TASK / m["path"]) == m["inference_sha256"] == sums.get(m["path"].split("2026-11-05final/")[1]))]
    checks["2_model_hashes"] = {"models": len(manifest["models"]), "mismatches": bad, "ok": not bad}

    saved = json.loads((RUN / "scenes.json").read_text())
    regenerated = [s for c in CONDITIONS for s in scenes(c, n)]
    diff = [sv["scenario_id"] for s, sv in zip(regenerated, saved)
            if s.seed != sv["seed"] or not np.allclose(s.target_ee_pos, sv["target_ee_pos"])
            or not np.allclose(s.base_start_pos, sv["base_start_pos"])]
    checks["3_scenes"] = {"scenes": len(saved), "regenerated": len(regenerated), "mismatches": diff,
                          "ok": len(saved) == len(regenerated) and not diff}

    summary = json.loads((RUN / "summary.json").read_text())["conditions"]
    mism = []
    for cond, entry in summary.items():
        for pol, v in entry.items():
            rs = [r for r in rows if r["condition"] == cond and r["policy"] == pol]
            if f'{sum(r["sustained_hold_pass"] == "True" for r in rs)}/{len(rs)}' != v["hold"]:
                mism.append(f"{cond}/{pol}")
    checks["4_summary"] = {"mismatches": mism, "ok": not mism}

    changed = [f for f, h in manifest["source_sha256"].items() if (TASK / f).exists() and sha(TASK / f) != h]
    missing = [f for f in manifest["source_sha256"] if not (TASK / f).exists()]
    checks["5_source"] = {"files": len(manifest["source_sha256"]), "changed": changed, "missing": missing,
                          "ok": not changed and not missing}

    out = {"all_ok": all(c["ok"] for c in checks.values()), "checks": checks}
    (RUN / "verification.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v["ok"] for k, v in checks.items()}), "all_ok:", out["all_ok"])
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
