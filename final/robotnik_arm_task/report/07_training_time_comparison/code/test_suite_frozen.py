"""Run evaluation.test_suite with the frozen unified-test-v1 manifest after re-auditing it against current inventories.

The frozen manifest's source_audit lists every dataset inventory present when it was frozen; new training datasets
add inventories, so a rebuilt manifest differs only there. This wrapper rebuilds it, requires identical scenarios and
zero overlaps with the current inventories, then hands test_suite the frozen body (same sha256 as earlier runs).
All other test_suite arguments pass through unchanged. Used for the two full-DART test runs of 2026-11-04.

Usage (from final/robotnik_arm_task/):
    python report/07_training_time_comparison/code/test_suite_frozen.py --models <step_42000.npz ...> \
        --out results/test/<datetime>__<name> --workers 20
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # final/robotnik_arm_task
import evaluation.test_suite as ts
from evaluation.suite_manifest import TASK_DIR, build_manifest, load_manifest



def main():
    frozen = load_manifest(TASK_DIR / 'experiments/splits/unified-test-v1.json')
    fresh = build_manifest()
    assert fresh['scenarios'] == frozen['scenarios'], 'test scenes changed'
    assert not fresh['source_audit']['unintended_heldout_overlaps'], 'test scenes overlap current inventories'
    assert [k for k in set(fresh) | set(frozen) if k != 'sha256' and fresh.get(k) != frozen.get(k)] in ([], ['source_audit'])
    added = sorted(set(fresh['source_audit']['source_inventory_sha256']) - set(frozen['source_audit']['source_inventory_sha256']))
    print(f'manifest re-audit OK: scenes identical, 0 overlaps, {len(added)} inventories added since freezing: {added}', flush=True)
    ts.build_manifest = lambda: {k: v for k, v in frozen.items() if k != 'sha256'}
    raise SystemExit(ts.main())


if __name__ == '__main__':
    main()
