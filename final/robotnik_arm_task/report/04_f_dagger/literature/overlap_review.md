# OU noise injection + DAgger: targeted overlap review

Search date: 2026-11-04. This is a targeted public-literature search, not a systematic review or a proof of novelty. Research skill used with public web search because no Consensus connector was available. Technical findings below were checked against original papers or author/publisher pages.

## Conclusion

The broad combination of disturbed expert data and DAgger has prior art. Explicit OU noise and DAgger also occur in published work, but noise location and purpose matter. The reviewed material did not establish a paper with the complete recipe of this project's gated expert-action OU collection, clean action labels, retained noisy-expert dataset, and subsequent beta=0/no-additional-noise DAgger rounds for base-and-arm control. This absence of an exact match is not evidence that the recipe is novel.

## Project recipe being compared

- DART-style: execute clip(clip(expert action) + gated OU noise); save the clean expert action on the current pre-action observation.
- Main protocol: tau=0.2 s, per-episode alpha in [0.25,1], arm stationary std alpha*0.75 N m and wheel stationary std alpha*0.15 rad/s; independent channels.
- Wheel noise latches off at the first reach-active step. All noise shuts off 1.5 s after first task reach. The observation-derived gate itself can return from 1 to 0.
- Combined R8: initialise the collector using a Clean + DART-style + gate student; retain these base data and aggregate DAgger rounds.
- DAgger: student executes, beta=0, no added OU action noise, expert labels every pre-action state at 1000 Hz of simulation time; train each subsequent policy from scratch on the aggregated data.
- DART construction retains sustained-hold-passing episodes. DAgger retains task failures but excludes numerical failures and empty episodes.

Local implementation: `COMP_0245_2026_Final/final/robotnik_arm_task/data_collection/collect_dart.py`, `data_collection/collect_dagger.py`, `experiments/configs/dart/d2-s075-x2.yaml`, `experiments/dagger_rounds.py`, `training/datasets/build.py`, `core/feature_sets.py`.

## Verified relevant work

### 1. Closest temporal expert-action perturbation precedent

Yasasa Abeysirigoonawardena, Florian Shkurti, Gregory Dudek. **Generating Adversarial Driving Scenarios in High-Fidelity Simulators.** ICRA 2019, pp. 8271-8277.

- Author PDF: https://www.yasasa.me/assets/pdf/icra2019.pdf
- Author bibliographic metadata: https://www.yasasa.me/
- Evidence location: PDF page 6, Section VI-D.
- Verified: they perturb expert steering with temporally correlated noise and use DAgger to obtain labels on states outside the driving policy's training distribution.
- Important limit: the inspected passage does not identify that noise as OU or specify this project's exact two-stage training/data retention schedule. Do not equate all temporally correlated processes with OU.
- Comparison judgment: strong overlap in the high-level combination of temporally correlated expert-action perturbation and DAgger, but different task and unverified exact noise process / schedule.

### 2. Explicit disturbed-expert + DAgger combination

Yusuf Umut Ciftci, Darren Chiu, Zeyuan Feng, Gaurav S. Sukhatme, Somil Bansal. **SAFE-GIL: SAFEty Guided Imitation Learning for Robotic Systems.** ICRA 2025; arXiv initial submission 2024.

- Full text: https://arxiv.org/html/2404.05249
- Author project and venue: https://y-u-c.github.io/safegil/
- Evidence location: Section V-A, “Mitigating Covariate Shift”, Figure 4.
- Verified: an explicit experiment combines DAgger with adversarial disturbances during expert rollouts.
- Difference: reachability-guided adversarial disturbances, rather than a fixed stochastic OU action process. Its guidance targets safety-critical states.
- Comparison judgment: direct prior art for the general idea that expert perturbation and on-policy aggregation can be complementary. It does not establish equivalence to the project's phase-gated OU protocol.

### 3. Explicit OU + DAgger, but observation perturbation

Tong Zhang et al. **HuB: Learning Extreme Humanoid Balance.** CoRL 2025, PMLR 305:686-704.

- Proceedings metadata: https://proceedings.mlr.press/v305/zhang25b.html
- Full text checked: https://arxiv.org/html/2505.07294v2
- Evidence locations: Section 3.1 (DAgger student distillation); Section 3.4 and Appendix B.4 (OU noise).
- Verified: a PPO-trained privileged teacher is distilled into a deployable student with DAgger. During student training, OU noise perturbs observed root orientation; related observation quantities are recomputed from that noisy orientation.
- Difference: sensor/observation corruption, not perturbing expert torques and wheel commands to create a separate recovery-demonstration dataset.
- Comparison judgment: falsifies an unrestricted claim that nobody has combined OU noise and DAgger. Does not reproduce this project's action-noise/data-collection mechanism.

### 4. Explicit OU and DAgger in a hybrid IL/RL system

Qijie Zou, Kang Xiong, Qiang Fang, Bohan Jiang. **Deep imitation reinforcement learning for self-driving by vision.** CAAI Transactions on Intelligence Technology, 2021, 6(4):493-503. DOI: 10.1049/cit2.12025.

- Publisher full text: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12025
- PDF checked: https://pdfs.semanticscholar.org/ae84/ede1344c2747d1aa7ffdc60e3e969b1ca68c.pdf
- Evidence location: Section 4.2, exploration factors / data collection, Eq. (11), Table 4; Section 4.3 and Table 6.
- Verified: DAgger improves the imitation component; OU is used for DDPG exploration in the hybrid framework.
- Difference: OU serves actor exploration in reward-based learning; this is not the project's expert-action augmentation followed by purely supervised DAgger.
- Comparison judgment: component co-occurrence, not a close pipeline match.

### 5. Recent noise-augmentation + aggregation work

Jenny Wang, George Kantor. **Policy-Calibrated DAgger: Offline Calibrated Noise Injection for Imitation Learning.** arXiv:2609.30462, submitted 24 September 2026. Treated here as a preprint; peer-reviewed venue not verified.

- Metadata: https://arxiv.org/abs/2609.30462
- Full text: https://arxiv.org/html/2609.30462v1
- Evidence locations: Section IV-E; Section V; Table I and associated training description.
- Verified: diffusion-policy error calibration sets augmentation noise; interventions are augmented with corrective joint-position action chunks. Training retains base data, intervention rounds and augmented intervention data.
- Difference: offline augmentation and calibrated noise around HG-DAgger interventions, with heuristic corrective chunk generation; not expert-action OU execution and dense Markovian oracle labels. No OU process was identified in the inspected full text.
- Comparison judgment: related at the combined augmentation/aggregation level and relevant to a current related-work section, but different collection and relabeling mechanism.

### Lower-priority candidate inspected

Megan Tjandrasuwita, Jie Xu, Armando Solar-Lezama, Wojciech Matusik. **MeMo: Meaningful, Modular Controllers via Noise Injection.** NeurIPS 2024.

- Original paper: https://www.proceedings.com/content/079/079017-0594open.pdf
- Verified: DAgger training with Gaussian noise on the output of an internal boss module.
- Difference: latent-interface regularisation for modular transfer, rather than OU noise on expert execution commands. Not a close match for this project.

## What the project can defensibly claim

A conservative description is: “We evaluate a task-specific combination of phase-gated, temporally correlated expert-action augmentation and DAgger for mobile manipulation.” Describe empirical findings and the actual gate/noise schedule. Cite the temporal-noise-plus-DAgger and adversarial-disturbance-plus-DAgger precedents where discussing the general combination.

Do not claim “first combination of OU noise and DAgger” or imply that replacing another noise process with OU alone establishes a new algorithm. The distinctive implementation details are candidates for evaluation, not established novelty. Controlled comparisons can test whether temporal correlation, phase-dependent noise scheduling, retaining the OU dataset, and adding DAgger each matter in this task.

## Search audit

23 targeted keyword queries were submitted in this turn, plus full-text reading and bibliographic verification. Six core full-text papers were inspected; five form the main overlap assessment and one (MeMo) was classified as lower priority. Search returned further candidates that were not fully validated; they are not used as evidence for the conclusion. Search result counts are not treated as a publication census.

Query groups:

1. DAgger + Ornstein / Ornstein-Uhlenbeck / Uhlenbeck / OU noise (spelling and Unicode variants; some queries returned irrelevant game results).
2. Dataset aggregation + OU noise.
3. DART + DAgger + combination / pretraining.
4. DAgger + expert action noise / noise injection / temporally correlated / gated.
5. Candidate title and metadata verification for SAFE-GIL, HuB, Policy-Calibrated DAgger, the ICRA 2019 paper and the 2021 DIRL paper.

Scope limits: public web index; no authenticated scholarly citation database or exhaustive code search; exact-match absence is not proof of novelty. Three OpenReview PDF opens returned browser verification pages; accessible arXiv/author versions were used instead. Publication dates come from paper metadata, not search engine crawl timestamps.
