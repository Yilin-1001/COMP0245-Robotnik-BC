"""Download, verify and unpack the submitted model weights (standard library only).

Run from final/robotnik_arm_task/:
    python tools/fetch_release.py                    # weights + unified-test inventory (about 56 MiB)
    python tools/fetch_release.py --with-small-net   # also the MLP02/04/06 width study (report/09_network_size)

Default source: the repository's GitHub Release. Any mirror holding the same files works with
    python tools/fetch_release.py --base-url <folder URL>   (e.g. https://huggingface.co/<user>/<repo>/resolve/main)
Already downloaded the zips by hand? Put them in one folder and pass --from-dir <folder>.

Each zip is checked against the SHA-256 pinned below before extraction, and every extracted model file is then
checked against checkpoints/2026-11-05final/SHA256SUMS. Files are extracted relative to final/robotnik_arm_task/.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
RELEASE_DIR = TASK / "checkpoints/2026-11-05final"
BASE_URL = "https://github.com/Yilin-1001/COMP0245-Robotnik-BC/releases/download/v1.0-submission"
ASSETS = {  # name -> SHA-256 (written by tools/package_release.py)
    "robotnik-bc-checkpoints-2026-11-05final.zip": "3830effee125115733aa4fa96a03d56eab622ef130a6ab0dac61bf08a2d19018",
    "robotnik-bc-test-inventory-2026-11-05.zip": "132dd474a287bf37e9a697669aa43719c36504c4abb8b4060d31ecfbe42eb458",
}
SMALL_NET = {"robotnik-bc-small-net-2026-11-05.zip": "808e84c84717543e289e2c7dab4f24d11f83f422cf2e5516f3591ac7023f0e04"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def obtain(name: str, expected: str, args) -> Path:
    if args.from_dir:
        path = Path(args.from_dir) / name
    else:
        path = args.cache / name
        if not (path.exists() and sha256(path) == expected):
            args.cache.mkdir(parents=True, exist_ok=True)
            url = f"{args.base_url.rstrip('/')}/{name}"
            print(f"downloading {url}", flush=True)
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(path)
    actual = sha256(path)
    if actual != expected:
        sys.exit(f"{path}: SHA-256 {actual} differs from the published {expected}; download it again")
    print(f"OK  {name}")
    return path


def extract(path: Path) -> None:
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            target = (TASK / name).resolve()
            if TASK.resolve() not in target.parents:  # refuse absolute or ../ entries
                sys.exit(f"{path.name}: unsafe archive path {name}")
        z.extractall(TASK)


def check_release() -> None:
    bad = []
    lines = (RELEASE_DIR / "SHA256SUMS").read_text().splitlines()
    for line in filter(str.strip, lines):
        digest, rel = line.split(maxsplit=1)
        f = RELEASE_DIR / rel
        if not f.is_file() or sha256(f) != digest:
            bad.append(rel)
    if bad:
        sys.exit(f"{len(bad)} release files missing or changed, e.g. {bad[:3]}")
    print(f"OK  {len(lines)} files in {RELEASE_DIR.relative_to(TASK)} match SHA256SUMS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=BASE_URL, help="folder URL that holds the release zips")
    parser.add_argument("--from-dir", help="use zips already downloaded into this folder")
    parser.add_argument("--cache", type=Path, default=TASK.parents[1] / "dist/release",
                        help="where downloads are kept (default: <repo>/dist/release, git-ignored)")
    parser.add_argument("--with-small-net", action="store_true", help="also fetch the optional width-study models")
    args = parser.parse_args()
    assets = {**ASSETS, **(SMALL_NET if args.with_small_net else {})}
    for name, digest in assets.items():
        extract(obtain(name, digest, args))
    check_release()
    print("Done. Next: python tools/verify_release.py")


if __name__ == "__main__":
    main()
