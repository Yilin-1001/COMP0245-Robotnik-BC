"""Report names and upload-safe identifiers; historical run IDs remain traceable aliases.

Gate always means the extra reach_gate input, not the OU collection's noise gate.
DART is reserved for full learner-fitted Gaussian DART (R9), never R3/R4/R7/R8.
"""
import re

# row: (report name, canonical file/directory ID, historical row ID)
MODEL_NAMES = {
    "R1": ("Baseline", "R1-Baseline", "R1-baseline"),
    "R2": ("Baseline + Gate", "R2-Baseline-Gate", "R2-gate"),
    "R3": ("OU", "R3-OU", "R3-dart"),
    "R4": ("OU + Gate", "R4-OU-Gate", "R4-dart-gate"),
    "R5": ("DAgger", "R5-DAgger", "R5-baseline-dagger"),
    "R6": ("DAgger + Gate", "R6-DAgger-Gate", "R6-gate-dagger"),
    "R7": ("f-DAgger", "R7-f-DAgger", "R7-dart-dagger"),
    "R8": ("f-DAgger + Gate", "R8-f-DAgger-Gate", "R8-dart-gate-dagger"),
    "R9": ("DART + Gate", "R9-DART-Gate", "R9-dart-full"),
}


def row_id(identifier: str) -> str | None:
    match = re.match(r"^(R[1-9])(?:-|__|$)", str(identifier))
    return match[1] if match else None


def canonical_id(identifier: str) -> str:
    """Replace the method prefix only, preserving round, seed, architecture and date."""
    row = row_id(identifier)
    if row is None:
        return identifier
    _, canonical, legacy = MODEL_NAMES[row]
    for prefix in (canonical, legacy):
        if identifier == prefix or identifier.startswith((prefix + "-", prefix + "__")):
            return canonical + identifier[len(prefix):]
    return canonical if identifier == row else identifier


def legacy_id(identifier: str) -> str:
    """Resolve a canonical CLI alias to a historical config/manifest key."""
    row = row_id(identifier)
    if row is None:
        return identifier
    _, canonical, legacy = MODEL_NAMES[row]
    if identifier == row:
        return legacy
    if identifier == canonical or identifier.startswith((canonical + "-", canonical + "__")):
        return legacy + identifier[len(canonical):]
    return identifier


def report_label(identifier: str) -> str:
    """A readable report label, retaining variant/round qualifiers when present."""
    row = row_id(identifier)
    if row is None:
        return identifier
    name, canonical, _ = MODEL_NAMES[row]
    converted = canonical_id(identifier).split("__")[0]
    if identifier != row and not converted.startswith(canonical):
        return identifier
    suffix = converted[len(canonical):] if converted.startswith(canonical) else ""
    return f"{row} {name}" + (f" ({suffix.lstrip('-')})" if suffix else "")
