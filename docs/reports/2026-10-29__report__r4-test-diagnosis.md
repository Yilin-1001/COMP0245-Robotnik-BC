# R4-dart-gate on the 64 Test Scenarios: Bottleneck Diagnosis Before DAgger

Date: 29 October 2026. Status: diagnostic, run at the user's request before DAgger is designed.

**Protocol note.** This is a look at the sealed test scenes outside the frozen final protocol. Nothing was written to `results/results.csv`, so the protocol gate in `evaluation/evaluate.py` is unchanged. The DAgger design below is informed by these scenes, so the report must state that the test set was viewed before DAgger, and any DAgger model should be selected on validation only.

Run: `python -m evaluation.test_diagnosis --models <R4 s0, s1, s2 .npz> --out results/diagnostics/2026-10-29__r4-test-diagnosis` (640 rollouts plus 13 takeovers, about 15 min on 22 workers). The evaluator predicates are the standard ones (reach < 10 mm after 8 s, then 3 s more, all of the final 1 s < 10 mm, 40 s cap).

## 1. Headline

| Policy (64 test scenes) | s0 | s1 | s2 | Pooled hold (Wilson 95%) |
|---|---|---|---|---|
| Teacher A | 64/64 | | | 1.00 |
| **Full student** (R4) | 60 | 60 | 59 | **0.93 (0.89–0.96)** |
| Student arm + teacher wheels ("arm") | 61 | 60 | 60 | 0.94 (0.90–0.97) |
| Student wheels + teacher arm ("base") | 63 | 64 | 64 | 0.99 (0.97–1.00) |

- On test, R4 does at least as well as on validation (0.86, 0.78–0.92). Its median final error is 1.0–1.7 mm, against 0.67 mm for the teacher.
- **The arm is the bottleneck.** Swapping in the teacher's wheels changes almost nothing (0.93 → 0.94). Swapping in the teacher's arm removes almost every failure (0.99). Navigation (time to gate opening: median 6.14 s for the student, 6.14 s for the teacher) is solved.
- There were no numerical failures or blow-ups.

## 2. Where it fails

The 13 failed full-student episodes are concentrated in **7 of 64 scenes**, all with **low or near targets**:

| Target height | Scenes | Failed runs (3 seeds) |
|---|---|---|
| 0.10–0.30 m | 11 | **10/33** |
| 0.30–0.50 m | 11 | 2/33 |
| 0.50–0.80 m | 17 | 1/51 (test-0031, 0.65 m away) |
| 0.80–1.25 m | 25 | 0/75 |

- They are **not out-of-coverage scenes.** The distance to the nearest fitting scene in (distance, azimuth, height) space is the same for failing and passing scenes (median 0.092 vs 0.089). Low targets are simply the hard part of the in-range space.
- Failure types: 9 reached after 8 s but drifted out of the 10 mm hold, 2 stalled 20–27 mm short, and 2 were inside 10 mm only before 8 s and never again. Every failure was a **near miss** (minimum error 1.4–27 mm).
- **Teacher label chatter is ruled out.** In 6 low-height test scenes, the teacher's own hold labels hit ±30 N·m for 30–100% of the time. Those scenes produced only 1 failed run out of 54 (all modes), against 24 out of 522 elsewhere.

## 3. Mechanism

### 3.1 Compounding error in the arm near the target (primary)

| Arm torque RMSE vs teacher label (N·m), pooled over 64 scenes | Navigation | Approach | Hold |
|---|---|---|---|
| On the teacher's states (s0 / s1 / s2) | 0.03 / 0.03 / 0.03 | 1.26 / 1.28 / 1.28 | 0.27 / 0.31 / 0.31 |
| **On the student's own states** | 2.22 / 2.44 / 3.45 | 5.16 / 6.85 / 4.33 | **1.78 / 2.27 / 2.15** |
| On its own states, failed episodes only | 6.0 / 10.0 / 10.1 | 6.6 / 9.5 / 5.4 | 4.9 / 10.1 / 9.1 |

- The arm error on the student's own states is **4–8× larger in approach and hold** than on the teacher's states. This is the covariate-shift signature DAgger targets.
- At the stalled states, the teacher's label differs from the student's output by **7–18 N·m on joints 3 and 4** for tens of seconds (for example test-0015 s0, test-0031 s1). The teacher would push hard toward the target, but the student settles 10–40 mm short. Its position feedback near low targets is too weak.
- This matches the teacher's own operating limits: teacher A has no integrator and misses the 10 mm hold under persistent torque errors above about 0.5–1 N·m. The student's own-state hold error (about 2 N·m RMS) is well above that.

### 3.2 The base leaves the reach-gate band during long episodes (amplifier)

- After the gate opens, the base slides passively at about **2.3 mm/s for both teacher and student**, even with zero wheel commands. The student's wheels are not biased: their mean is 0.003 rad/s where the teacher commands 0.
- The teacher drives the base back once the distance error exceeds 0.08 m while the gate is still open (it stays open up to 0.12 m). The student outputs about 0 rad/s there (wheel RMSE on its own hold states 0.19–0.23 vs 0.02 on the teacher's states). In the DART data, wheel noise is switched off once reaching starts, so these states are almost absent from training.
- The teacher never meets this situation, because it finishes 3 s after reaching. A student stalled short keeps the episode running for 20–40 s, the base drifts out of the band, and the gate closes. The arm then returns home, so the final error jumps to 400–650 mm. **5 of 13 failures end this way** (test-0015 s0, test-0031 s1, test-0032 s0/s1/s2). Test-0032 starts 0.2 m too close, so the base must reverse, and the student stops reversing as soon as the gate opens. Its s1 run flips the gate 109 times.

### 3.3 The failed states are recoverable

Each of the 13 failed full-student episodes was handed to teacher A at the 40 s cap: **13/13 recovered and held**. Near misses settled in 0.16–0.62 s, and the gate-closed cases (arm home, 400–650 mm away) in 1.0–2.5 s. The states the student reaches are inside the teacher's recovery region, so teacher labels there are valid and corrective.

## 4. Implications for DAgger

1. **DAgger fits this bottleneck.** The error is on-policy (3.1) and the teacher can label and recover every failed state (3.3). Neither more capacity nor more DART noise targets this specifically. More DART noise is also limited by the teacher's 0.5–1 N·m hold tolerance.
2. **Let the student run long stretches.** The earlier DAgger R1 used β = 0.75 with 10 ms chunks, so student stretches were at most 70 ms and it never visited these states. R4 already succeeds 86–93% of the time, so roll out the student alone (β = 0, or a short expert prefix only), label every visited state with teacher A, and keep the full 40 s episode. The stall-and-drift states only appear after seconds.
3. **Over-sample low and near targets** (z < 0.5 m, distance < 1 m) in DAgger scenes, drawn from fitting scenes or fresh development seeds, never test or validation. Uniform Sobol puts about 35% of scenes below 0.5 m and 20% within 1 m.
4. **Keep the wheel loss on DAgger rows**, so the student learns to correct the base while the gate is open (3.2).
5. **Success criterion:** own-state hold arm RMSE well below 1 N·m (currently about 2), measured with this script on validation scenes, alongside sustained hold on the 32 validation scenes.

## 5. Limitations

- 64 scenes × 3 seeds; the 13 failures come from 7 scenes, so per-scene conclusions rest on few cases.
- Action RMSE on the student's states compares the clipped student output with the teacher's label at the same state. It measures disagreement, not the causal contribution of each error.
- Latencies from this run include the teacher and true-FK calls inside the recording wrapper. They are not the student's inference latency.
