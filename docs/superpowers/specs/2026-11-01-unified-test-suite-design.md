# Unified R1–R8 test suite: final design

Date: 2026-11-01. Status: requirements confirmed in conversation; written design independently reviewed with no important unanswered implementation questions; approved by the user; implementation and full execution authorized.

## Intent and scope

Build `evaluation/test_suite.py` to evaluate the current R1–R8 terminal models and Markovian expert A under one reproducible closed-loop protocol. Evaluate task success, terminal precision, and completion efficiency separately. Deliver the program, meaningful automated checks, and the complete formal evaluation with figures. The user explicitly authorized the full evaluation, not just a program demonstration.

The user confirmed `step_42000` (the earlier `step_4200` message was a typo), exclusion of numerical/simulation failures from successes, and all decisions below. No T6 robustness suite or combined extreme suite is included. Do not train models or select checkpoints using the test results.

## Approaches

1. **Selected approach: a separate test-suite entry point reusing policy/environment interfaces.** Use the existing `MLPPolicy`, `MarkovianReference`, `RobotEnv`, and frozen-test loading/checks. Add a dedicated scorer/rollout for the new continuous-hold and fixed-duration protocol. Keep scene construction, execution, and plotting separated sufficiently to test scoring without the simulator and replot without rerunning physics.
2. Modifying the old `evaluate.rollout` defaults would change existing validation/test results and is unnecessary. Do not do this.
3. Merging/replacing all development probes would expand the scope and conflate already-viewed development scenes with new test scenes. Reuse their geometry/physical concepts where relevant, not their seeds or result-selected scenes. Preserve existing tools.

The selected approach was explained in the conversation; there is one sensible implementation path that preserves the historical protocol while implementing the newly agreed one.

## Frozen scenarios

Freeze all scene coordinates and protocol values before evaluating any formal model or expert. Store a versioned JSON manifest with a canonical content SHA-256 under `experiments/splits/`. The manifest is immutable once present; regeneration that differs must fail. Never change `clean253.json` or the original 64-row CSV. Verify T0's CSV against the original frozen split hash.

| Suite | Scenario geometry | Count | Fixed duration | Continuous hold required |
|---|---|---:|---:|---:|
| T0-core | Original 64 test scenarios, exact coordinates/IDs | 64 | 40 s | 3 s |
| T1-far / 3–6 m | Horizontal start-to-target distance 3–6 m | 24 | 80 s | 3 s |
| T1-far / 6–10 m | Distance 6–10 m | 24 | 80 s | 3 s |
| T1-far / 10–15 m | Distance 10–15 m | 24 | 120 s | 3 s |
| T2-near | Distance 0.3–0.5 m | 24 | 40 s | 3 s |
| T3-height / low | Target world z 0.05–0.1 m | 16 | 40 s | 3 s |
| T3-height / high | Target world z 1.25–1.30 m | 16 | 40 s | 3 s |
| T4-start | Paired horizontal translations of 24 predetermined T0 scenes; new base start radius from origin 1–3 m | 24 | 40 s | 3 s |
| T5-long-hold | Fresh scenes inside the nominal training ranges | 24 | 60 s | 20 s |
| **Total** | | **240** | | |

Unless changed in the table, nominal ranges are base start x/y in [-0.5,0.5] m, initial yaw and relative target azimuth in [-pi,pi], horizontal start-to-target distance 0.5–3 m, and world target z 0.1–1.25 m. Initial arm/finger configuration, velocities, control rate (1 kHz), physics, and execution limits are the existing nominal task settings. Targets are static end-effector goals, not prescribed final base poses.

T4: select 24 distinct T0 parents with a fixed scene-generation stream, without consulting outcomes. Translate start and target by the same horizontal vector so the new start is at radius 1–3 m. Preserve yaw, relative target geometry, target z, and joint initialization. Record parent IDs and translations and compare paired outcomes/terminal errors with T0. These are intentional paired geometries, not independent new relative tasks.

Implementation defaults (builder decisions, not additional user requirements): use fixed, separately derived generation streams per suite/band, a new master seed (1101000), and Latin-hypercube sampling to cover each new band with the agreed non-power-of-two scene counts. Assign distinct environment seeds from a verified unused namespace, keeping original T0 seeds. Store explicit coordinates; do not rely solely on seed regeneration. Check exact scene/target/start overlaps with current fitting/validation/test and DAgger source inventories. T4 parents are intentional within-test relationships. Do not reject or redraw scenes according to any controller's result.

## Models and reference

Default model sources:

- R1–R4: `2026-10-28__training-ablation-v1`, seeds 0/1/2.
- R5–R8: current local `2026-11-01__training-ablation-v2`; use the round selected from seed-0 validation success, tie to later round, identically for all three seeds. Inspection currently selects round 3 for all four rows. Exclude round 0 from DAgger selection.
- For every run, use `step_42000.npz` (verify export source step and matching metadata/config). Do not mix teammate historical results, intermediate DAgger rounds, or arbitrary checkpoint files into defaults.
- Markovian expert A at the recorded lead time 0.50 s; one rollout per scenario, no training-seed replication.

Pin the resolved selection and file SHA-256 values in `run.json` before execution. Resolve all jobs before the first rollout. Validate interface and feature definitions per model; do not assume every model has the first model's input dimension. Default is 24 students + one expert = **6000 rollouts**. Missing default models are listed and skipped; mark the report incomplete and show exact row/seed coverage. Explicitly supplied unreadable/incompatible models fail preflight with a clear error rather than silently disappearing. Accept arbitrary model paths supported by the existing inference adapter and a compatible adjacent config.

## New scoring protocol

Use ground-truth simulator end-effector error for scoring, never expose privileged metrics to policies. Policy inference uses the same deployable features and saved training normalizer.

- **Only error <10 mm is task success.** The earlier secondary success at 20 mm was explicitly withdrawn. Do not output 20 mm success or completion-time metrics.
- Start checking at t=0; remove the old t>8 s restriction.
- Success occurs when a continuous interval of 3 s (ordinary suites) or 20 s (T5) has all observed errors strictly below 10 mm. An error >=10 mm resets the streak. Define elapsed streak duration using integer control-step timestamps to avoid accumulating clock drift. Count the endpoint samples consistently.
- Record first arrival below 10 mm, the start of the first qualifying streak, and its completion time. **Completion efficiency = first qualifying streak completion time**, including the hold duration.
- Continue to the fixed suite duration after success; ignore the environment's early `done` signal. Later finite drift does not erase an earlier success, and appears separately in terminal precision.
- Any nonfinite state/action/error, model exception, or simulator exception causes early termination, grade F, and task failure, **overriding any earlier success**. Preserve exception category/message and the time of failure. A finite large error alone is not an exception.
- A normal full-duration rollout without a qualifying streak is unsuccessful; do not call its cap a successful completion time.

Terminal precision is independent of success and uses the maximum error over the final inclusive 1 s window of the fixed-duration rollout:

| Grade | Condition |
|---|---|
| A | e <5 mm |
| B | 5 <=e <10 mm |
| C | 10 <=e <20 mm |
| D | 20 <=e <50 mm |
| E | e >=50 mm |
| F | Numerical/model/simulator error or incomplete execution due to error |

A+B is NOT a substitute for the task-success flag. Retain last-frame error and position, last-second mean/max error, minimum error, longest continuous <10 mm interval, saturation fraction, torque variation, and inference latency diagnostics. Record target/start poses and suite/band in every output or through its pinned scene record. Keep enough downsampled error history for time plots without storing all 1-kHz observations for 6000 runs.

## Reporting and statistics

Primary denominator: **all frozen scenes** in the requested suite, with exceptions counted as failed. Expert and students share the same scoring protocol. Additional view: identical metrics on the **expert-success subset**. Expert failure means reference-controller failure, not physical infeasibility; preserve all scenes and never redraw them. If no expert scene succeeds, subset metrics are unavailable (N/A), not zero. Missing model runs are not successful/failed scenes; show incomplete coverage separately.

For efficiency, report completion-time median only for successes with its success/total counts. Plot cumulative completion fraction versus time using all evaluated scenes as the denominator, so a method succeeding on few easy scenes does not look unconditionally fast. For paired efficiency comparisons, state the common-success denominator.

Show per-training-seed results and row aggregates. Repeated seeds use the same scenes: use scenario-cluster resampling for row-level descriptive intervals, preserving seed results together for a scene. Avoid treating 3*n outcomes as independent scenes in an ordinary pooled Wilson interval. Record bootstrap settings and fixed plotting seed. Do not pool heterogeneous suites into an unexplained headline success rate. Report T1 distance bands and T3 height bands as well as suite totals.

Figures (PNG + PDF, regenerated solely from stored results):

1. Main row success comparison by suite/band with seed points, uncertainty and expert reference.
2. A–F terminal-grade stacked bars.
3. Row-by-suite/band success heatmap.
4. Terminal-error ECDF and/or boxplots with logarithmic error axis and 10/20 mm guides; handle zero/error cases explicitly and show excluded F count.
5. Success versus distance/height with fixed bins and sample counts; do not blend different hold/time/physics protocols invisibly.
6. Target-location plots in the initial robot frame colored by terminal grade, including meaningful height views; special T4 paired-shift comparison.
7. Efficiency curves/table with parameter counts, success-conditional completion time, and inference latency. Measure comparison latency in a separate, single-process warmed inference benchmark to avoid presenting concurrent simulation scheduling delays as intrinsic model speed. Preserve rollout latency as diagnostics.

Outputs: `summary.md`, `summary.csv`, `rollouts.csv`, `run.json`, scene manifest copy, raw per-job result JSON, downsampled histories, and plots. Include exact model/source/checkpoint selection, scene/protocol hashes, relevant-code hashes, Git revision/dirty state, software/hardware information, start/finish timestamps, requested and available model coverage. Reports state that T0 has previously informed development and this is a new scoring protocol; no direct old/new success-rate equivalence is claimed.

## Entry point, output layout and resumption

```bash
python -m evaluation.test_suite
python -m evaluation.test_suite --rows R4 R8 --seeds 0 --suites T0 T1
python -m evaluation.test_suite --models <a.npz> <b.npz>
python -m evaluation.test_suite --plots-only <result-directory>
```

Expose worker count and an explicit resume directory. Output example: `results/test/2026-11-01_HHMMSS__R1-R8/`, or a row/seed subset name. Use Europe/London for user-facing timestamp names and record UTC/offset as provenance. Per-model result filenames retain full unambiguous model IDs. Avoid overwriting existing complete runs.

Write each completed job atomically so interruption preserves progress. Resumption checks model hashes, full frozen scene manifest, scoring settings and relevant code hashes; filename existence alone is insufficient. Expert results are computed once and reused inside the same compatible run. A normal evaluation failure is a recorded F and does not abort unrelated jobs. Infrastructure/preflight errors must not silently turn thousands of jobs into F. `--plots-only` does not load simulator or model weights, and does not rerun jobs.

Implementation details such as exact helper-module boundaries and safe default worker count are left to the builder. Changes must preserve pre-existing unrelated modifications, especially current edits in README/docs and the development probes.

## Verification and full execution

1. Automated scorer checks: strict threshold boundaries, uninterrupted duration, reset/recovery, t=0, post-success drift, exception override, cap before hold completion, and full last-second window.
2. Manifest checks: exact counts/ranges, reproducibility, immutable hash, original T0 integrity, T4 paired transforms, distinct generation/environment streams and source-split separation.
3. Model-selection checks: correct experiment/date/round/ties/seeds/terminal step, filters, missing-model coverage, adapter metadata.
4. Resume and plotting checks on synthetic outcomes: mismatch rejection, no reruns, all-failure/empty-success/expert-zero-success cases, complete tables/figures.
5. Run existing relevant evaluation checks to confirm old tools still behave as before.
6. Small real physics/inference smoke check using a separate development fixture, not a formal test scene; check environment and selected-model feature dimensions, benchmark throughput and choose worker count before freezing the formal run.
7. Freeze the 240 scenarios and 24-model selection; execute all 6000 jobs, provide progress updates and measured throughput, resume if needed, and generate/inspect final figures.
8. Verify requested job cardinality, uniqueness and coverage, report F categories and partial coverage honestly, and commit only verified task changes using explicit paths. Generated runs are artifacts, not source-code commits.

No implementation or formal rollouts are part of this design-only stage.
