# 09. Network size (f-DAgger + Gate at 32×3 to 256×3)

This folder supports the report's **MLP Size** result. It compares f-DAgger + Gate (R8) at four hidden-layer widths: 32×3, 64×3, 128×3 and the main 256×3. OU + Gate (R4) at the same widths is the no-DAgger reference.

Main point: **with f-DAgger, network size has little effect on accuracy inside the training range.** A 32×3 network with 3,337 parameters, 42× fewer than 256×3, reaches 89.6% validation and 89.0% in-range test success. Without DAgger, the smaller networks fall further behind: OU + Gate goes from 86.5% at 256×3 to 75.0% at 32×3 on validation. f-DAgger therefore gains the most where capacity is smallest.

## Model names

Small-network models use the same Report names as the rest of the report ([`docs/model-naming.md`](../../../../docs/model-naming.md)). The method name is unchanged, and the width is given as the size label (32×3, 64×3, 128×3, 256×3). The complete mapping, with every seed, is in [`tables/naming.csv`](tables/naming.csv).

| Report name | Size | Canonical run id (seed *s*) | Original run id |
|---|---|---|---|
| f-DAgger + Gate | 32×3 | `R8-f-DAgger-Gate-r3__MLP02-32x3__s<s>__v3__2026-11-05` | `R8-dart-gate-dagger-MLP02nodev-r3__MLP02-32x3__s<s>__v2__2026-11-05` |
| f-DAgger + Gate | 64×3 | `R8-f-DAgger-Gate-r2__MLP04-64x3__s<s>__v3__2026-11-05` | `R8-dart-gate-dagger-MLP04nodev-r2__MLP04-64x3__s<s>__v2__2026-11-05` |
| f-DAgger + Gate | 128×3 | `R8-f-DAgger-Gate-r3__MLP06-128x3__s<s>__v3__2026-11-05` | `R8-dart-gate-dagger-MLP06nodev-r3__MLP06-128x3__s<s>__v2__2026-11-05` |
| f-DAgger + Gate | 256×3 | `R8-f-DAgger-Gate-r3__MLP01-256x3__s<s>__v3__2026-11-05` | `R8-dart-gate-dagger-nodev-r3__MLP01-256x3__s<s>__v3__2026-11-05` |
| OU + Gate | 32×3 / 64×3 / 128×3 | `R4-OU-Gate__MLP0<k>-<w>x3__s<s>__v1__2026-11-02` | `R4-dart-gate__MLP0<k>-<w>x3__s<s>__v1__2026-11-02` |
| OU + Gate | 256×3 | `R4-OU-Gate__MLP01-256x3__s<s>__v1__2026-10-28` | `R4-dart-gate__MLP01-256x3__s<s>__v1__2026-10-28` |

About the version tag: the teammate labelled his small-network runs `v2`. In this report, `v2` means DAgger *with* the 64 development scenes, and `v3` means DAgger on the 221 fitting scenarios only ([`../08_dev_scene_removal/`](../08_dev_scene_removal/README.md)). His runs follow the v3 protocol (`--dev-scenes 0`, 221 episodes per round), so their Report names use `v3`. The original ids are kept as provenance: checkpoint configs, dataset names and variant-yaml hashes all refer to them.

## Results

All rates pool seeds 0–2 and use 95% Wilson intervals. Full tables: [`tables/summary.md`](tables/summary.md).

**Validation** (32 scenes, 96 rollouts per row; [`tables/validation.csv`](tables/validation.csv))

| Size | Params | OU + Gate | f-DAgger + Gate | Gain | Selected round |
|---|---:|---:|---:|---:|---:|
| 32×3 | 3,337 | 75.0 (72/96) | 89.6 (86/96) | +14.6 | 3 |
| 64×3 | 10,761 | 79.2 (76/96) | 90.6 (87/96) | +11.4 | 2 |
| 128×3 | 37,897 | 84.4 (81/96) | 90.6 (87/96) | +6.2 | 3 |
| **256×3** | 141,321 | 86.5 (83/96) | **92.7 (89/96)** | +6.2 | 3 |

**Unified test** (240 scenes, 720 rollouts per row; [`tables/test.csv`](tables/test.csv)). The groups follow the teammate's summary, because only these pooled counts exist for the small networks.

| f-DAgger + Gate | In range T0+T4+T5 | Edges T2+T3 | Far T1 | All 240 |
|---|---:|---:|---:|---:|
| 32×3 | 89.0 | 51.2 | 70.4 | 74.6 |
| 64×3 | 90.5 | 53.0 | 65.3 | 74.2 |
| 128×3 | 89.9 | 57.7 | 70.4 | 76.5 |
| **256×3** | **93.2** | 54.8 | 45.8 | 70.0 |
| *OU + Gate 256×3 (reference)* | 90.2 | 51.2 | 90.7 | 81.2 |
| Expert | 100.0 | 89.3 | 100.0 | 97.5 |

What the report can claim:

- **In range, size barely matters under f-DAgger.** In-range test success is 89.0–93.2% at every width, and the Wilson intervals overlap. Validation is 89.6–92.7%.
- **f-DAgger closes the capacity gap.** On validation, the gap between 32×3 and 256×3 is 11.5 pp for OU + Gate but 3.1 pp for f-DAgger + Gate. The gain from f-DAgger shrinks as width grows (+14.6, +11.4, +6.2, +6.2 pp).
- **Small networks cost much less to deploy.** The 32×3 network has 42× fewer parameters and 46× fewer FLOPs, and its .npz file is far smaller ([`tables/efficiency.csv`](tables/efficiency.csv)).

What the report must not claim:

- **"Smaller networks generalise better."** On T1, 32×3 to 128×3 score 65–70% against 45.8% for 256×3. However, every width has its own seed-0 DAgger collection chain, and folder 08 shows that chain randomness alone moves T1 by about 20 pp. The report already attributes 256×3's T1 loss to DAgger labelling only visited states. Report this T1 difference as a chain-to-chain difference, not a capacity effect.
- **"f-DAgger beats OU + Gate on the test set at every width."** OU + Gate was not tested on the 240 scenes at 32×3–128×3. The 2026-11-03 architecture test stopped at about 34 of 240 scenes per model and is not used. The test-set comparison of methods exists for 256×3 only, where OU + Gate is higher on all 240 (81.2 vs 70.0) because of T1.
- **Anything per suite (T0 alone, T5 alone) or paired by scene for 32×3–128×3.** That needs the teammate's per-scene `rollouts.csv`, which is not in the repository.

## Limitations and provenance

- **Two machines.** The 32×3–128×3 f-DAgger runs were collected, trained and tested on the teammate's machine (RTX 3070 Ti, branch `feature/small-net-dagger`, merged into `develop` as `e1b732b`). Everything else ran on this machine. Simulation reproduces across the two: the expert's 240-scene results are identical on both machines, scene by scene in every group (`summary.md`). Each teammate model was also re-validated here ([`tables/rerun.csv`](tables/rerun.csv), when present).
- **Different data builds.** His `clean253` and `dart-d2s075x2` datasets have the same 221 and 441 episodes as ours, but different SHA-256 values and row counts (2,657,690 vs 2,658,124 rows for `clean253`). See `data/teammate/configs/`.
- **Different OU + Gate students.** The OU + Gate rows for 32×3–128×3 come from this machine's 2026-11-02 architecture ablation, using the per-architecture learning rates of `results/ablation/2026-11-02__architecture-ablation-v1/manifest.yaml`: 1e-2 for 32×3, 1e-3 for 64×3 and 128×3. The teammate used the same rates. His chains instead started from his own round-0 OU + Gate students (2026-11-04, seed 0 only: 19/32, 24/32 and 26/32). So the per-width gain mixes two training runs of the round-0 student. The seed-0 trajectory of each chain is in `validation.csv` (`chain_seed0_by_round`).
- **Round selection.** The round is chosen as in the main report: the best seed-0 validation over rounds 1–3, with ties going to the later round. This picked round 2 for 64×3 (29/32 vs 28/32 at round 3) and round 3 for the others.
- **Timing.** Latency, training time and DAgger pipeline minutes from the teammate's machine (`*_teammate` columns) must not be mixed with this machine's numbers. Latency measured on this machine is in the `*_this_machine` columns.
- **Statistics.** Wilson intervals treat the pooled rollouts as independent. The three seeds share scenes, so the intervals understate the uncertainty.

## Reproduce

Rebuild the tables (no simulation):

```bash
python report/09_network_size/code/make_tables.py
```

The small-network chains were run on the teammate's machine with `experiments/small_net_nodev.sh` (`experiments.dagger_arch_rounds --tag nodev --dev-scenes 0 --rounds 3 --final-seeds 1 2`). His unified test used `experiments/splits/unified-test-v1-local-nodev.json`. Its 240 scenarios are field-for-field identical to `experiments/splits/unified-test-v1.json`, and only the machine-specific source-audit block differs.

## Contents

```text
09_network_size/
  code/make_tables.py     data/ -> tables/
  tables/  naming.csv       Report name, size and canonical run id for every model, with the original run id
           validation.csv   32-scene validation per width: OU + Gate and f-DAgger + Gate, per seed and pooled
           test.csv         240-scene unified test per width, 4 groups, plus the 256x3 OU + Gate reference and the expert
           efficiency.csv   parameters, FLOPs, latency and .npz size (this machine), teammate training/pipeline minutes
           rerun.csv        teammate models re-validated on this machine vs his recorded counts
           network-size.tex LaTeX table for the paper
           summary.md       all tables as Markdown
  data/    teammate/        his summary.{md,csv}, per-width DAgger summary.csv and cost.csv, and the 9 checkpoint configs
           validation/per_scene/  32-scene records: OU + Gate at 4 widths and f-DAgger + Gate 256x3 (seeds 0-2)
           validation/rerun_this_machine.json  teammate models re-validated on this machine
           latency/latency_local.csv  evaluation.latency_bench on this machine
           test/            256x3 OU + Gate, f-DAgger + Gate and expert rollouts; scene manifest; his local test split
```
