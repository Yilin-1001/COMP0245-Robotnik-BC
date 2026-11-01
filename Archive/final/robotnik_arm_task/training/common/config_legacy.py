"""Archived on 2026-11-01 from training/common/config.py: content IDs and checks of the frozen clean-pipeline protocols (their files are in Archive/final/robotnik_arm_task/experiments/protocols/).

Kept for reference only; it is not importable as is (its imports refer to the original module)."""

# ---------------------------------------------------------------- frozen protocols

PROTOCOL_DECISIONS = {"budget", "loss", "lr", "finalists", "selected", "calibration", "refit", "final", "notes"}


def protocol_id_for(body: dict) -> str:
    """Content-derived ID of a frozen protocol (decisions, evidence hashes and parent)."""
    return "p-" + hashlib.sha256(canonical_json(body).encode()).hexdigest()[:12]


def validate_protocol(protocol: dict) -> None:
    body = {k: protocol.get(k) for k in ("decisions", "evidence", "parent")}
    if protocol.get("protocol_id") != protocol_id_for(body):
        raise ValueError("protocol content does not match its ID: tampered or edited in place")
    unknown = set(protocol.get("decisions") or {}) - PROTOCOL_DECISIONS
    if unknown:
        raise ValueError(f"unknown decision groups {sorted(unknown)}")
