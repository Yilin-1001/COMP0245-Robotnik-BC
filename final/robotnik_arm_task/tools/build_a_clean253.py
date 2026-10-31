"""Build data/Markovian/Clean_253 (teacher A only) from the frozen split.

Used when teacher B's local collection differs from the recorded one (here B fails train-0220), so the paired
data_collection.build_clean_subsets refuses to run. A's files are checked against the hashes frozen in
experiments/splits/clean253.json; the folder is only published if they match (byte-identical to the original).
Usage (from final/robotnik_arm_task/): python tools/build_a_clean253.py
"""
import csv, datetime, hashlib, json, shutil, sys
from copy import deepcopy
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
SRC, DST = TASK / "data/Markovian/Clean", TASK / "data/Markovian/Clean_253"
SPLIT = json.loads((TASK / "experiments/splits/clean253.json").read_text())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_csv(p, fields, rows):
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


assert not DST.exists(), f"{DST} already exists"
selected = set(SPLIT["fitting_ids"]["clean253"]) | set(SPLIT["validation_ids"])
assert len(selected) == 253
fields, rows = read_csv(SRC / "training_manifest.csv")
ep_fields, episodes = read_csv(SRC / "episodes.csv")
sc_fields, train = read_csv(SRC / "scenarios_train.csv")
rows = [r for r in rows if r["split"] == "test" or r["scenario_id"] in selected]
episodes = [r for r in episodes if r["split"] == "test" or r["scenario_id"] in selected]
train = [r for r in train if r["scenario_id"] in selected]
assert len(rows) == len(episodes) == 317 and len(train) == 253
assert all(r["eligible_initial_bc"] == "True" for r in rows if r["split"] == "train")

stage = DST.with_name(".Clean_253.building")
shutil.rmtree(stage, ignore_errors=True)
for d in ("train", "test", "qa"):
    (stage / d).mkdir(parents=True)
write_csv(stage / "training_manifest.csv", fields, rows)
write_csv(stage / "episodes.csv", ep_fields, episodes)
write_csv(stage / "scenarios_train.csv", sc_fields, train)
shutil.copy2(SRC / "scenarios_test.csv", stage / "scenarios_test.csv")
prov = []
for i, r in enumerate(rows, 1):
    src, dst = SRC / r["file"], stage / r["file"]
    shutil.copy2(src, dst)
    prov.append(dict(episode_id=r["episode_id"], scenario_id=r["scenario_id"], split=r["split"], file=r["file"],
                     source_file=str(Path("..") / "Clean" / r["file"]), sha256=sha(dst), bytes=src.stat().st_size))
    if i % 50 == 0:
        print(f"copied {i}/{len(rows)}", flush=True)
write_csv(stage / "episode_provenance.csv", list(prov[0]), prov)

ok = True
for name in ("scenarios_train.csv", "scenarios_test.csv", "training_manifest.csv", "episode_provenance.csv"):
    got, want = sha(stage / name), SPLIT["source_sha256"][f"A/clean253/{name}"]
    print(f"{name:26s} {'MATCH' if got == want else 'MISMATCH'}  {got[:12]} vs frozen {want[:12]}")
    ok &= got == want
if not ok:
    sys.exit(f"NOT published: A data differs from the frozen record; staging left at {stage}")

manifest = deepcopy(json.loads((SRC / "manifest.json").read_text()))
manifest["dataset_id"] = "Markovian/Clean_253"
manifest["dataset_kind"] = "paired clean trajectory subset; byte-identical episode copies"
manifest["counts"] = {"train_episodes": 253, "test_episodes": 64, "total_episodes": 317}
manifest["derivation"] = {
    "version": "a-only-from-frozen-split-v1",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "selection": "the 253 training scenarios of experiments/splits/clean253.json (fitting 221 + validation 32)",
    "reason": "local teacher-B collection fails train-0220, so the paired builder refuses; teacher A is unaffected",
    "verified": "scenarios_*, training_manifest.csv and episode_provenance.csv (all episode SHA-256) match the frozen split",
    "builder": "tools/build_a_clean253.py"}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2))
for f in ("qa_report.json",):
    if (SRC / "qa" / f).exists():
        shutil.copy2(SRC / "qa" / f, stage / "qa" / "source_qa_report.json")
stage.rename(DST)
print(f"DONE: {DST} published; teacher-A data is byte-identical to the frozen original.")
