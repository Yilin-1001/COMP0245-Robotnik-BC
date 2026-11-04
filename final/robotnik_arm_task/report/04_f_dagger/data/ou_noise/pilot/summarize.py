"""Verify completed pilots and write an evidence report; no simulator runs."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
TASK = Path('/media/yilin/data/ML_Final/COMP_0245_2026_Final/final/robotnik_arm_task')
FOLDERS = {'smoke':24, 'tuning':96, 'confirmation':192, 'd2_sampled':64}
all_rows = {}
checked = 0
for folder, expected in FOLDERS.items():
    path = ROOT / folder
    with (path/'results.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    rows = [json.loads(x) for x in (path/'results.jsonl').read_text().splitlines()]
    assert len(rows) == len(csv_rows) == expected, (folder,len(rows))
    assert len({r['name'] for r in rows}) == expected
    manifest = json.loads((path/'manifest.json').read_text())
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((TASK/name).read_bytes()).hexdigest() == digest, name
    assert manifest['protocol']['sensor_noise'] == 0
    assert manifest['protocol']['push'] is None
    for r in rows:
        with np.load(path/'traces'/f'{r["name"]}.npz') as data:
            t=data['trace']
        assert np.isfinite(t).all()
        assert np.all(t[t[:,6]==1,15:19]==0), r['name']
        assert np.all(t[t[:,7]==1,8:19]==0), r['name']
        if r['sigma_t']==0:
            assert np.all(t[:,8:19]==0), r['name']
        if r['reached']:
            assert abs(r['end_s']-r['first_reach_s']-3.)<.0011
            assert r['noise_off_s'] is not None
        checked += 1
    all_rows[folder]=rows

def tally(rows):
    n=len(rows)
    return f'{sum(r["reached"] for r in rows)}/{n}', f'{sum(r["held_last1s"] for r in rows)}/{n}'

lines = [
    '# D2 OU Validation Report', '', 'Date: 26 October 2026.', '',
    '## Scope', '',
    'This is an independent simulator pilot, not training-dataset collection. The repository controllers and configurations were not modified. No Claude implementation prompt has been generated.', '',
    '- Markovian: repository default `lead_time=0.15 s`. The older `0.50 s` override was checked only in the four-target smoke comparison.',
    '- Stateful: original `base_controller` and `arm_controller`, original world-origin navigation-goal rule, 25 s navigation timeout, base update before arm update, full true orientation as in the original template.',
    '- Nominal masses and friction; fixed arm/finger initialization; no sensor noise, external pushes or parameter randomization.',
    '- 1 kHz control. OU is the only added action perturbation: independent channels, zero initial noise, exact discrete OU update, correlation time 0.2 s. Sigma denotes stationary standard deviation.',
    '- Arm actions clipped to +/-30 Nm; wheel actions clipped to the controller-derived bound of about 5.273 rad/s.',
    '- Wheel noise is permanently gated off at the first reach-active flag (Markovian) or navigation-done flag (Stateful). This latch belongs to the noise scheduler, not the teacher.',
    '- First task reach requires reach-active, time >8 s, and true end-effector error <10 mm. All action noise switches off 1.5 s after that reach. Recording ends 3 s after reach or at 40 s, whichever comes first.',
    '- Sustained-hold pass requires reaching and every 1 kHz sample in the final 1 s to be below 10 mm. This measures recovery after noise removal, not continuous disturbance rejection.',
    '- As in the proposed collector, early stall detection is recorded but not used to end these pilot episodes. Reaching counts therefore use the declared 40 s collection protocol, not necessarily an unmodified original `check_stop` run.', '',
    '## Scenarios and separation', '',
    'The tuning cohort contains 16 scrambled Sobol scenarios (seed 427601). The separate confirmation cohort contains 32 scenarios (seed 429913), with target distance 0.5–3 m, relative azimuth +/-pi, height 0.1–1.25 m, start x/y +/-0.5 m and initial yaw +/-pi. Targets are transformed from robot-relative coordinates into world coordinates. Neither cohort uses the repository training or final test scenario lists.', '',
    'Each confirmation scenario has one clean rollout and two OU rollouts with different noise seeds, per teacher. A further rollout on the same 32 scenarios tests the actual D2 episode-scale draw, alpha uniformly sampled from 0.25 to 1. It is a paired check, not another independent set of targets.', '',
    '## Tuning results', '',
    '| Teacher | Arm sigma / wheel sigma | Reached | Sustained hold |',
    '| --- | --- | --- | --- |',
]
for teacher in ['Markovian','Stateful']:
    for sigma in [0.,.75,1.5]:
        rs=[r for r in all_rows['tuning'] if r['teacher']==teacher and r['sigma_t']==sigma]
        reach,hold=tally(rs)
        lines.append(f'| {teacher} | {sigma:g} Nm / {sigma*.2:g} rad/s | {reach} | {hold} |')
lines += ['',
    'The stronger 1.5 Nm / 0.3 rad/s candidate was not selected. The lower 0.75 Nm / 0.15 rad/s candidate was frozen before the confirmation runs. Clean failures are retained rather than removed from the denominator.', '',
    '## Confirmation results', '',
    '| Teacher | Mode | Trials | Reached | Sustained hold | Median final error |',
    '| --- | --- | --- | --- | --- | --- |',
]
summary=[]
for teacher in ['Markovian','Stateful']:
    groups=[('Clean',[r for r in all_rows['confirmation'] if r['teacher']==teacher and r['sigma_t']==0]),
            ('OU at fixed maximum',[r for r in all_rows['confirmation'] if r['teacher']==teacher and r['sigma_t']>0]),
            ('D2 alpha-sampled',[r for r in all_rows['d2_sampled'] if r['teacher']==teacher])]
    for mode,rs in groups:
        reach,hold=tally(rs)
        lines.append(f'| {teacher} | {mode} | {len(rs)} | {reach} | {hold} | {np.median([r["final_error_mm"] for r in rs]):.2f} mm |')
        summary.append(dict(teacher=teacher,mode=mode,n=len(rs),reached=sum(r['reached'] for r in rs),held=sum(r['held_last1s'] for r in rs)))
lines += ['', '## Failures and paired interpretation', '',
          '| Cohort | Teacher | Scenario | Noise seed | Reached | Last-1-second max error | Clean on same scenario | Navigation timeout |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
clean={(r['teacher'],r['scenario_id']):r for r in all_rows['confirmation'] if r['sigma_t']==0}
for folder in ['confirmation','d2_sampled']:
    for r in sorted(all_rows[folder],key=lambda r:(r['teacher'],r['scenario_id'],r['noise_seed'],r['sigma_t'])):
        if not r['held_last1s']:
            c=clean[(r['teacher'],r['scenario_id'])]
            label='Clean' if r['sigma_t']==0 else folder
            lines.append(f'| {label} | {r["teacher"]} | {r["scenario_id"]} | {r["noise_seed"]} | {r["reached"]} | {r["last1s_max_mm"]:.2f} mm | {"held" if c["held_last1s"] else "failed"} | {r["nav_timeout"]} |')
lines += ['', '## Original early-stop diagnostics', '',
          'For transparency, the original `StopCriteria` was evaluated alongside collection. The table below counts whether its first stop message was a reach, and whether the collection episode also passed the final hold check. Early stall messages can occur before an eventual reach within the collector time budget.', '',
          '| Teacher | Mode | First original stop was reach | Original reach plus final hold |',
          '| --- | --- | --- | --- |']
for entry in summary:
    teacher,mode=entry['teacher'],entry['mode']
    source=all_rows['d2_sampled'] if mode=='D2 alpha-sampled' else all_rows['confirmation']
    rs=[r for r in source if r['teacher']==teacher and (mode=='D2 alpha-sampled' or (r['sigma_t']==0)==(mode=='Clean'))]
    lines.append(f'| {teacher} | {mode} | {sum(r["original_stop_reached"] for r in rs)}/{len(rs)} | {sum(r["original_stop_reached"] and r["held_last1s"] for r in rs)}/{len(rs)} |')
lines += ['', '## Reproducibility and limits', '',
          f'All {checked} episode traces passed finite-value, clean-zero-noise, wheel-gating and all-noise-off checks. Source SHA-256 hashes in every manifest still match the repository. These observations support only the tested protocol and range, not a guarantee on every target or random seed. Repeated noise trials on the same target are paired observations, not independent target samples.', '',
          'The candidate D2 maximum is 0.75 Nm for arm torque and 0.15 rad/s for wheel speed, with tau=0.2 s and per-episode alpha in [0.25,1]. Further collection should retain all failures and keep Clean and DART/OU data in separate folders. The OU variant should be described as DART-style unless the full DART noise-selection procedure is implemented.', '',
          'Diagnostic traces are decimated to 100 Hz to keep this pilot small; simulation, success checks and noise updates ran at 1 kHz. These traces are not substitutes for full-rate training recordings.', '',
          '- [Pilot script](validate_ou.py)',
          '- [Summary and trace checks](summarize.py)',
          '- [Tuning manifest](tuning/manifest.json) and [CSV](tuning/results.csv)',
          '- [Confirmation manifest](confirmation/manifest.json) and [CSV](confirmation/results.csv)',
          '- [Actual D2 sampling manifest](d2_sampled/manifest.json) and [CSV](d2_sampled/results.csv)',
          '- Per-episode traces and worker logs are retained inside each run folder.', '']
(ROOT/'report.md').write_text('\n'.join(lines))
(ROOT/'summary.json').write_text(json.dumps(dict(episodes_checked=checked,results=summary),indent=2))
print(json.dumps(dict(episodes_checked=checked,results=summary),indent=2))
print('Report:', ROOT/'report.md')
