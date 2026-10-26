# Paired Clean Dataset Subsets

Built: 26 October 2026.

These datasets are filtered, paired copies of the recordings described in the [clean collection report](clean_collection_report.md). No new trajectories were collected, no expert parameters were changed, and the original `Clean/` folders remain unchanged.

The suffix denotes the number of **training episodes per expert**. Every dataset also retains all **64 original test episodes**, without test-based filtering.

## 1. Dataset locations and inventories

Paths are relative to `final/robotnik_arm_task/data/`.

| Dataset | Training episodes | Test episodes | Training control steps | Training chatter flags |
| --- | ---: | ---: | ---: | ---: |
| `Markovian/Clean_253/` | 253 | 64 | 3,045,616 | 28 |
| `Stateful/Clean_253/` | 253 | 64 | 3,356,970 | 27 |
| `Markovian/Clean_223/` | 223 | 64 | 2,639,448 | 0 |
| `Stateful/Clean_223/` | 223 | 64 | 2,913,498 | 0 |

Within each suffix, both experts use exactly the same training scenarios and the same test scenarios. `Clean_223` is a strict training-scenario subset of `Clean_253`.

The recordings retain their original teacher versions: `markovian-lt0.50-v1` and `template-origin-goal-v1`. Episode metadata, action labels, internal-state labels, sampling frequency, timestamps and trajectory lengths are unchanged.

## 2. Selection rules

### Clean_253: paired successful demonstrations

Include a training scenario only when both experts' source manifests mark its episode eligible for initial BC: valid recording, complete post-reach window and sustained-hold pass.

Both experts therefore exclude `train-0064`, `train-0131` and `train-0183`. Their Stateful recordings failed navigation; the corresponding successful Markovian recordings are excluded from this comparison to maintain pairing. All original recordings, including these failures, remain available in `Clean/`.

This subset retains the original `hold_chatter` flags. It is not a chatter-free training set.

### Clean_223: paired successful demonstrations without either chatter flag

Start from the 253 paired successful scenarios. Exclude a scenario from **both** experts if either source recording has `hold_chatter`.

This removes another 30 paired scenarios and leaves 223. Each teacher's training set has zero `hold_chatter` flags under the existing detection rule; this is not a guarantee that every possible label-quality problem has been eliminated.

### Test set

The test cohort is identical across all four datasets and remains unfiltered. Its existing chatter flags are retained: seven Markovian and six Stateful test episodes. The suffix changes only the training selection. Do not use test data for normalization, tuning or checkpoint selection.

## 3. Files in each dataset

- `train/`, `test/`: independent, byte-identical copies of selected original `.npz` files, not symlinks or hard links.
- `manifest.json`: derived dataset identity, exact counts, selection policy, source hashes, original teacher configuration and field definitions.
- `scenarios_train.csv`: selected training scenarios, preserving original IDs and order.
- `scenarios_test.csv`: unchanged original test scenario manifest.
- `episodes.csv`, `training_manifest.csv`: rows corresponding to the episodes actually present in this dataset.
- `selection_audit.csv`: include/exclude decision and paired reasons for all 256 original training scenarios.
- `episode_provenance.csv`: original file location, byte size and SHA-256 for each copied episode.
- `qa/subset_qa_report.json`: new inventory, selection, pairing and file-integrity checks.
- `qa/coverage.json`: recomputed parameter-coverage counts for the selected training cohort, unchanged test cohort and original training cohort.
- `qa/source_qa_report.json`, `qa/inherited_episode_checks.csv`: explicitly inherited original QA evidence. These are not new simulator replay results.

The inherited `collection_configuration` describes the original acquisition. The derived manifest's `counts` and `scenario_split_definition` describe the filtered inventory. Consult those fields when loading a subset.

## 4. Verification and coverage

All **1,208 copied episode files** passed SHA-256 verification against their original files. All 640 original episode hashes also matched their collection attempt logs. Before/after checks confirmed that all **673 files** in the two original `Clean/` folders remained unchanged.

Verified selection counts, on-disk inventories, train/test separation, matching scenario manifests across teachers, and preservation of every test episode. Neither building nor verification ran a simulator or fitted normalization statistics. Original trajectory-level QA is inherited through byte-identical episode copies.

Both selected training cohorts occupy all 32 coarse parameter cells defined by **2 distance bins × 2 height bins × 8 relative-azimuth sectors**. Filtering changes the marginal distribution, and these subsets no longer have the complete original 256-point Sobol balance. Occupied coarse cells do not prove full six-dimensional, trajectory-state or exact-boundary coverage.

**Important finer-scale gap:** `Clean_223` has no training targets below **0.22753 m**. All 27 successful `Clean_253` scenarios below that height were removed by the paired chatter filter. The unchanged test cohort contains seven targets below this new training minimum. Coarse height bins hid this missing interval. Keep `Clean_253` as the full-range reference, and interpret `Clean_223` as a label-quality/coverage tradeoff rather than an equally broad clean replacement.

Summary and the rebuild helper are stored in `data/clean_subsets/summary.json` and `data/clean_subsets/build_clean_subsets.py`. The helper refuses to overwrite an existing subset folder.

## 5. Using the subsets for comparisons

Use the two `Clean_253` datasets for a paired comparison retaining successful expert behavior, and the two `Clean_223` datasets for a paired label-quality comparison. Keep the training budget and sampling rules controlled: equal episode counts do not imply equal numbers of control steps or optimizer updates. Preserve complete recurrent histories rather than truncating trajectories to force equal lengths.
