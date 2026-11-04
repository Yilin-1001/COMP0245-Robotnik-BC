# D2 OU Validation Report

Date: 26 October 2026.

## Scope

This is an independent simulator pilot, not training-dataset collection. The repository controllers and configurations were not modified. No Claude implementation prompt has been generated.

- Markovian: repository default `lead_time=0.15 s`. The older `0.50 s` override was checked only in the four-target smoke comparison.
- Stateful: original `base_controller` and `arm_controller`, original world-origin navigation-goal rule, 25 s navigation timeout, base update before arm update, full true orientation as in the original template.
- Nominal masses and friction; fixed arm/finger initialization; no sensor noise, external pushes or parameter randomization.
- 1 kHz control. OU is the only added action perturbation: independent channels, zero initial noise, exact discrete OU update, correlation time 0.2 s. Sigma denotes stationary standard deviation.
- Arm actions clipped to +/-30 Nm; wheel actions clipped to the controller-derived bound of about 5.273 rad/s.
- Wheel noise is permanently gated off at the first reach-active flag (Markovian) or navigation-done flag (Stateful). This latch belongs to the noise scheduler, not the teacher.
- First task reach requires reach-active, time >8 s, and true end-effector error <10 mm. All action noise switches off 1.5 s after that reach. Recording ends 3 s after reach or at 40 s, whichever comes first.
- Sustained-hold pass requires reaching and every 1 kHz sample in the final 1 s to be below 10 mm. This measures recovery after noise removal, not continuous disturbance rejection.
- As in the proposed collector, early stall detection is recorded but not used to end these pilot episodes. Reaching counts therefore use the declared 40 s collection protocol, not necessarily an unmodified original `check_stop` run.

## Scenarios and separation

The tuning cohort contains 16 scrambled Sobol scenarios (seed 427601). The separate confirmation cohort contains 32 scenarios (seed 429913), with target distance 0.5–3 m, relative azimuth +/-pi, height 0.1–1.25 m, start x/y +/-0.5 m and initial yaw +/-pi. Targets are transformed from robot-relative coordinates into world coordinates. Neither cohort uses the repository training or final test scenario lists.

Each confirmation scenario has one clean rollout and two OU rollouts with different noise seeds, per teacher. A further rollout on the same 32 scenarios tests the actual D2 episode-scale draw, alpha uniformly sampled from 0.25 to 1. It is a paired check, not another independent set of targets.

## Tuning results

| Teacher | Arm sigma / wheel sigma | Reached | Sustained hold |
| --- | --- | --- | --- |
| Markovian | 0 Nm / 0 rad/s | 16/16 | 16/16 |
| Markovian | 0.75 Nm / 0.15 rad/s | 16/16 | 15/16 |
| Markovian | 1.5 Nm / 0.3 rad/s | 16/16 | 14/16 |
| Stateful | 0 Nm / 0 rad/s | 15/16 | 15/16 |
| Stateful | 0.75 Nm / 0.15 rad/s | 15/16 | 15/16 |
| Stateful | 1.5 Nm / 0.3 rad/s | 14/16 | 14/16 |

The stronger 1.5 Nm / 0.3 rad/s candidate was not selected. The lower 0.75 Nm / 0.15 rad/s candidate was frozen before the confirmation runs. Clean failures are retained rather than removed from the denominator.

## Confirmation results

| Teacher | Mode | Trials | Reached | Sustained hold | Median final error |
| --- | --- | --- | --- | --- | --- |
| Markovian | Clean | 32 | 32/32 | 32/32 | 1.21 mm |
| Markovian | OU at fixed maximum | 64 | 64/64 | 62/64 | 1.39 mm |
| Markovian | D2 alpha-sampled | 32 | 32/32 | 31/32 | 1.09 mm |
| Stateful | Clean | 32 | 31/32 | 31/32 | 0.18 mm |
| Stateful | OU at fixed maximum | 64 | 60/64 | 59/64 | 0.74 mm |
| Stateful | D2 alpha-sampled | 32 | 31/32 | 31/32 | 0.45 mm |

## Failures and paired interpretation

| Cohort | Teacher | Scenario | Noise seed | Reached | Last-1-second max error | Clean on same scenario | Navigation timeout |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirmation | Markovian | sobol_429913_007 | 429913071 | True | 14.21 mm | held | False |
| confirmation | Markovian | sobol_429913_016 | 429913160 | True | 13.89 mm | held | False |
| Clean | Stateful | sobol_429913_002 | 429913020 | False | 1405.99 mm | failed | True |
| confirmation | Stateful | sobol_429913_002 | 429913021 | False | 2339.07 mm | failed | True |
| confirmation | Stateful | sobol_429913_009 | 429913090 | True | 13.71 mm | held | False |
| confirmation | Stateful | sobol_429913_017 | 429913170 | False | 2686.72 mm | held | True |
| confirmation | Stateful | sobol_429913_029 | 429913291 | False | 1783.06 mm | held | True |
| confirmation | Stateful | sobol_429913_030 | 429913301 | False | 1757.10 mm | held | True |
| d2_sampled | Markovian | sobol_429913_016 | 429913160 | True | 11.78 mm | held | False |
| d2_sampled | Stateful | sobol_429913_017 | 429913170 | False | 1724.20 mm | held | True |

## Original early-stop diagnostics

For transparency, the original `StopCriteria` was evaluated alongside collection. The table below counts whether its first stop message was a reach, and whether the collection episode also passed the final hold check. Early stall messages can occur before an eventual reach within the collector time budget.

| Teacher | Mode | First original stop was reach | Original reach plus final hold |
| --- | --- | --- | --- |
| Markovian | Clean | 27/32 | 27/32 |
| Markovian | OU at fixed maximum | 56/64 | 55/64 |
| Markovian | D2 alpha-sampled | 28/32 | 27/32 |
| Stateful | Clean | 31/32 | 31/32 |
| Stateful | OU at fixed maximum | 60/64 | 59/64 |
| Stateful | D2 alpha-sampled | 31/32 | 31/32 |

## Reproducibility and limits

All 376 episode traces passed finite-value, clean-zero-noise, wheel-gating and all-noise-off checks. Source SHA-256 hashes in every manifest still match the repository. These observations support only the tested protocol and range, not a guarantee on every target or random seed. Repeated noise trials on the same target are paired observations, not independent target samples.

The candidate D2 maximum is 0.75 Nm for arm torque and 0.15 rad/s for wheel speed, with tau=0.2 s and per-episode alpha in [0.25,1]. Further collection should retain all failures and keep Clean and DART/OU data in separate folders. The OU variant should be described as DART-style unless the full DART noise-selection procedure is implemented.

Diagnostic traces are decimated to 100 Hz to keep this pilot small; simulation, success checks and noise updates ran at 1 kHz. These traces are not substitutes for full-rate training recordings.

- [Pilot script](validate_ou.py)
- [Summary and trace checks](summarize.py)
- [Tuning manifest](tuning/manifest.json) and [CSV](tuning/results.csv)
- [Confirmation manifest](confirmation/manifest.json) and [CSV](confirmation/results.csv)
- [Actual D2 sampling manifest](d2_sampled/manifest.json) and [CSV](d2_sampled/results.csv)
- Per-episode traces and worker logs are retained inside each run folder.
