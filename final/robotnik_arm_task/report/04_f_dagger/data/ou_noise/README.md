# OU parameter-selection evidence

Start with [the parameter-selection report](../../ou_parameter_selection.md). This directory combines the archived D2 pilot, the expert time-scale stress test, the fixed protocol files and the source of the later student strength controls. Files were archived on 2026-11-04; no simulation or training was launched.

Diagnostic NPZ traces are versioned with this evidence. Worker `.log` files remain local; the source inventory records their hashes as part of the original archive.

| Location | Contents |
| --- | --- |
| `protocols/` | Base d2-s075-x2 protocol; half/double noise strength and double collection-count variants |
| `pilot/2026-10-26__report__ou-noise-validation.md` | Original D2 report, including pilot/confirmation settings and limitations |
| `pilot/{smoke,tuning,confirmation,d2_sampled}/` | Cohort results.csv, results.jsonl, manifest.json and diagnostic NPZ traces; 376 episodes in total. Worker logs are local only. |
| `pilot/validate_ou.py`, `pilot/summarize.py` | Original experiment and summary scripts, preserved byte for byte |
| `pilot/2026-10-26__summary__ou-noise-validation.json` | Original machine-readable pilot summary |
| `pilot/*_results.csv`, `pilot/*_manifest.json` | Existing flat copies retained for `../../code/make_tables.py` compatibility |
| `expert_stress/expert_noise_robustness.csv` | 2,112 episode results, including white, OU and constant disturbances; 1,248 OU episodes |
| `expert_stress/disturbance_thresholds.csv` | Existing threshold summary, also retained under `report/01_markovian_expert/tables/` |
| `expert_stress/2026-10-26__report__expert-operating-limits.md` | Original stress-test report; its protocol differs from D2 collection |
| `strength_controls/` | Original report supporting the teammate's validation-only controls, plus the dataset variant configs |
| `source_inventory.csv` | Source/target mappings, bytes and SHA-256 for all archived source files |
| `verification_summary.json` | Copy-integrity and archived trace audit results |

Supporting tables are [OU pilot](../../tables/ou_pilot.csv), [OU stress by condition](../../tables/ou_stress_by_condition.csv), [parameter evidence](../../tables/ou_parameter_evidence.csv), [student strength controls](../../tables/noise_strength_controls.csv) and [formal collection](../../tables/ou_collection.csv).

Current implementation snapshots are under [`../../code/ou_noise_snapshot/`](../../code/ou_noise_snapshot/). These are inspection snapshots of the working code at archive time, not a claim that historical pilot code hashes match the current controller. Run module-based experiments from the task directory with its original package layout.

The archived pilot `summarize.py` checks historical controller hashes and expects the full cohort layout. A failed controller-hash assertion after later source edits is a version mismatch, not evidence that the archived data were corrupted. Use `source_inventory.csv` for the copy-integrity record. Historical reports are preserved verbatim and may contain links relative to their former locations; use this index and the parameter-selection report to navigate the archive.
