"""Build the release assets that hold everything Git does not: weights and the unified-test inventory.

Run from final/robotnik_arm_task/ (author side, once, before uploading to GitHub Releases):
    python tools/package_release.py            # writes <repo>/dist/release/

The zips are byte-reproducible (sorted entries, fixed timestamps), so their SHA-256 values, which
tools/fetch_release.py pins, stay valid if the same files are packaged again. Every archive path is
relative to final/robotnik_arm_task/, so assessors extract them there.

Assets
  robotnik-bc-checkpoints-2026-11-05final.zip  checkpoints/2026-11-05final/ (27 final + 22 round-history models)
  robotnik-bc-test-inventory-2026-11-05.zip    scenario CSVs and dataset metadata read by the unified-test audit
  robotnik-bc-small-net-2026-11-05.zip         optional: MLP02/04/06 width study (report/09_network_size)
  SHA256SUMS.txt                               hashes of the three zips
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[1]
RELEASE = "checkpoints/2026-11-05final"
ASSETS = {
    "checkpoints": "robotnik-bc-checkpoints-2026-11-05final.zip",
    "inventory": "robotnik-bc-test-inventory-2026-11-05.zip",
    "small_net": "robotnik-bc-small-net-2026-11-05.zip",
}
FIXED_TIME = (2026, 11, 5, 0, 0, 0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_zip(out: Path, files: dict[str, Path]) -> None:
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for arcname in sorted(files):
            info = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, files[arcname].read_bytes())


def checkpoint_files() -> dict[str, Path]:
    root = TASK / RELEASE
    files = {p.relative_to(TASK).as_posix(): p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    listed = {line.split(maxsplit=1)[1] for line in (root / "SHA256SUMS").read_text().splitlines() if line.strip()}
    present = {Path(a).relative_to(RELEASE).as_posix() for a in files} - {"SHA256SUMS"}
    if listed != present:
        raise SystemExit(f"{RELEASE}/SHA256SUMS does not match the folder: "
                         f"missing {sorted(listed - present)[:5]}, unlisted {sorted(present - listed)[:5]}")
    return files


def inventory_files() -> dict[str, Path]:
    """Exactly the files evaluation.suite_manifest.build_manifest reads (no training arrays)."""
    names = ["data/scenarios/scenarios_train.csv", "data/scenarios/scenarios_test.csv",
             "data/scenarios/scenario_manifest.json", "data/Markovian/Clean_253/scenarios_test.csv"]
    for meta in sorted((TASK / "data/datasets/A").glob("*/metadata.json")):
        names.append(meta.relative_to(TASK).as_posix())
        root = json.loads(meta.read_text())["source"]["root"]
        names.append(f"{Path(root).as_posix()}/scenarios_train.csv")
    files = {n: TASK / n for n in sorted(set(names))}
    missing = [n for n, p in files.items() if not p.is_file()]
    if missing:
        raise SystemExit(f"inventory files missing: {missing}")
    return files


def small_net_files() -> dict[str, Path]:
    src = TASK / "checkpoints/small_net_models.zip"
    if not src.is_file():
        return {}
    staging = {}
    with zipfile.ZipFile(src) as z:
        for name in z.namelist():
            if not name.endswith("/"):
                staging[f"checkpoints/{name}"] = z.read(name)
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO / "dist/release")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    write_zip(args.out / ASSETS["checkpoints"], checkpoint_files())
    write_zip(args.out / ASSETS["inventory"], inventory_files())
    small = small_net_files()
    if small:
        with zipfile.ZipFile(args.out / ASSETS["small_net"], "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for arcname in sorted(small):
                info = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
                info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o644 << 16
                z.writestr(info, small[arcname])

    lines = []
    for key in ("checkpoints", "inventory", "small_net"):
        path = args.out / ASSETS[key]
        if path.exists():
            lines.append(f"{sha256(path)}  {path.name}")
            print(f"{path.stat().st_size / 2**20:8.1f} MiB  {path}")
    (args.out / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
