# 10. Robustness to unseen perturbations

This folder supports the report's **Robustness Performance** result. The report's final models run under nine conditions that never appear in training: nominal physics, changed floor friction and link mass, 20× sensor noise, and short pushes on the base or the hand.

## Run

`evaluation.robust_probe` was run on the release `checkpoints/2026-11-05final/` (Codex session, 2026-11-05, 20:45–21:15 UTC, `git_head` c3c114d). The run includes:

- Teacher A, run once.
- OU + Gate (R4, v1), DAgger, DAgger + Gate, f-DAgger and f-DAgger + Gate (R5–R8, v3 round 3), each with seeds 0–2.

There are 12 in-range scenes per condition: start within ±0.5 m, target 0.5–3 m away and 0.1–1.25 m high. They use development seeds 990000+, which are disjoint from every frozen split. All 16 policies share the same 108 scenes, giving 1,728 rollouts. Full DART (R9) was not run.

| Condition | Change (one at a time) |
|---|---|
| Nominal | none |
| Friction ×0.5 / ×1.5 | floor friction |
| Mass ×0.9 / ×1.1 | every link's mass and inertia; teacher A's gravity model stays nominal |
| Sensor noise ×20 | Gaussian noise, std 2e-3, on every observed quantity |
| Base push, driving | 120 N for 0.2 s (24 N·s) on the base at t = 3 s, random horizontal direction; 60 s cap |
| Base push, at target | the same push, 1 s after the hand first enters 10 mm |
| Hand push, at target | 20 N for 0.2 s (4 N·s) on the end effector, 1 s after it first enters 10 mm |

**Success** follows the probe's predicate: the hand reaches the target below 10 mm after at least 8 s, and every step of the final 1 s of the post-reach window stays below 10 mm. The window is 3 s, or 10 s for the at-target pushes so that the push lands inside it. This is a final-window hold, **not** the unified test's 3 s continuous hold, so rates are not comparable with Tables III–V.

**Verification** (`code/verify_run.py`, written to `data/run/verification.json`): all five checks pass.

1. There are 1,728 rollouts with no duplicates.
2. All 15 student `.npz` files match the SHA-256 in the run manifest and in the release `SHA256SUMS`.
3. The 108 scenes regenerate identically from `robust_probe.scenes()`.
4. `summary.json` reproduces from `rollouts.csv`.
5. The 54 source files hashed in the manifest are unchanged.

## Results

All values are sustained hold (%), with three seeds pooled (n = 36; the expert has n = 12). The full tables with 95% Wilson intervals are in [`tables/summary.md`](tables/summary.md).

| Condition | Expert | OU + Gate | DAgger | DAgger + Gate | f-DAgger | **f-DAgger + Gate** |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 12/12 | 86.1 | 77.8 | 80.6 | 83.3 | **91.7** |
| Friction ×0.5 | 12/12 | 86.1 | 83.3 | 80.6 | 77.8 | **88.9** |
| Friction ×1.5 | 12/12 | 94.4 | 91.7 | 91.7 | **97.2** | 91.7 |
| Sensor noise ×20 | 12/12 | 94.4 | 97.2 | **100.0** | 94.4 | 97.2 |
| Hand push, at target | 12/12 | 83.3 | 75.0 | 83.3 | 75.0 | **88.9** |
| Mass ×0.9 | 6/12 | **50.0** | 36.1 | 16.7 | 19.4 | **50.0** |
| Mass ×1.1 | 9/12 | **55.6** | 44.4 | 16.7 | 27.8 | 41.7 |
| Base push, driving | 6/12 | 19.4 | 27.8 | 36.1 | 33.3 | **38.9** |
| Base push, at target | 7/12 | 22.2 | **38.9** | 22.2 | 16.7 | 27.8 |
| **All 8 perturbations** | 76/96 | 63.2 | 61.8 | 55.9 | 55.2 | **65.6** |

Restricted to the scenes the expert itself holds ([`tables/expert_held.csv`](tables/expert_held.csv)), the eight perturbations give 80.3% for f-DAgger + Gate, 75.4% for OU + Gate, 73.2% for DAgger, 69.7% for f-DAgger and 68.4% for DAgger + Gate.

What the report can claim (descriptively; each cell has n = 36 and Wilson half-widths of about 12–16 pp):

- **Perturbations the expert tolerates** (friction, sensor noise, hand push): every student stays at 75–100%. f-DAgger + Gate is highest in 3 of the 5 conditions (nominal, friction ×0.5, hand push) and within two rollouts of the highest in the other 2.
- **Perturbations that break the expert** (link mass, base pushes): the expert holds only 6–9 of 12 scenes, and every student falls further. A cloned policy inherits its expert's limits.
- **f-DAgger + Gate vs DAgger + Gate**: f-DAgger + Gate is higher in 7 of 9 conditions, equal in 1, and lower in 1 (sensor noise, 35 vs 36 of 36). The largest gaps are under mass changes: 50.0 vs 16.7 at ×0.9 and 41.7 vs 16.7 at ×1.1. Over the 8 perturbations combined it scores 65.6% vs 55.9%.
- **Pooled over the 8 perturbations, f-DAgger + Gate has the highest rate** (65.6%), with OU + Gate next (63.2%). Their Wilson intervals overlap ([60.0, 70.9] vs [57.5, 68.6]).

What it must not claim:

- **Significance.** The intervals overlap almost everywhere, and the three seeds share the 12 scenes, so the Wilson intervals are optimistic.
- **Base-push robustness.** No method reaches 40% under either base push. The median final error while driving is 1.0–2.3 m for every policy, including the expert (1.5 m).
- **Comparison with full DART.** It was not run.

## Limitations

- **Small samples.** There are 12 scenes per condition, one push direction per scene, and one magnitude per perturbation. These are spot checks, not thresholds.
- **Cross-machine reproducibility near the expert's limits.** An earlier run of the same probe, on the teammate's machine, on 2026-10-30, with code unchanged apart from a default output path, gave the expert 4/12 and 10/12 under mass ×0.9 and ×1.1, against 6/12 and 9/12 here. The other seven conditions match. Conditions at the edge of the expert's capability are therefore not bit-reproducible across machines. Nominal-physics results are: the unified test's expert outcomes are identical on both machines (see `../09_network_size/`).
- **Superseded data.** That earlier run evaluated different models: R4 dated 2026-10-29, DAgger + Gate v2 round 3, and f-DAgger + Gate v2 **round 2**. It is kept in `data/superseded/` only as a record. Its conclusion that "f-DAgger is weak under base pushes while driving" (17% vs 42%) does not hold for the report's models (38.9% vs 36.1%).

## Reproduce

From `final/robotnik_arm_task/`:

```bash
# the run (about 30 min with 22 workers); the exact command, with all 15 model paths, is in data/run/manifest.json
python -m evaluation.robust_probe --models checkpoints/2026-11-05final/models/R{4..8}-*/seed-*/*.npz \
    --n 12 --workers 22 --out results/ablation/2026-11-05__training-ablation-v3/probes/robust/R4-R8-release-s012
python report/10_robustness/code/verify_run.py     # integrity checks -> data/run/verification.json
python report/10_robustness/code/make_tables.py    # data/ -> tables/ (no simulation)
```

The original output is in `results/ablation/2026-11-05__training-ablation-v3/probes/robust/R4-R8-release-s012/`. Its `run.log` (2.3 MB, mostly pybullet start-up lines) is not copied.

## Contents

```text
10_robustness/
  code/verify_run.py       five integrity checks on data/run
       make_tables.py      data/run -> tables/
  tables/  rates.csv          hold % per condition and method, Wilson 95%, plus pooled rows
           expert_held.csv    the same on the scenes the expert holds
           seeds.csv          holds per seed
           final_error.csv    median final error (mm)
           robustness.tex     LaTeX table for the paper
           summary.md         all tables as Markdown
  data/    run/           rollouts.csv, summary.json, manifest.json (command, models, hashes), scenes.json,
                          model_verification.json, environment.json, verification.json
           superseded/    the 2026-10-30 probe on earlier models (two summary.json files and its LaTeX table)
```
