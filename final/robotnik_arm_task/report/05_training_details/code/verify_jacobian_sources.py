"""Check the report's code/data copies against their recorded hashes and live sources.

Run from final/robotnik_arm_task/:
    python report/05_training_details/code/verify_jacobian_sources.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[3]


if __name__ == "__main__":
    files = json.loads((ROOT / "data/jacobian_validation_sources.json").read_text())["files"]
    for entry in files:
        copied = ROOT / entry["copy"]
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == entry["sha256"], copied
        source = REPO / entry["source"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry["sha256"], source
    print(f"Verified {len(files)} code/data copies against SHA-256 and repository sources")
