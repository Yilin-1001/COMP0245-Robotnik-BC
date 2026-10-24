> Exported from the Claude Doc https://claude.ai/artifact/5JgovWLGx7jmuEjpTkmAGT on 2026-10-24. Two diagrams (pipeline, timeline) are interactive in the doc only.

# COMP0245 Final Project Plan

Oct 24, 2026 · @Yilin

## Overview

Yilin, the only person coding, has until **Friday 6 November 2026** to submit code and a 6-page IEEE report. That is 14 days from today, Saturday 24 October. The project is behaviour cloning for the Robotnik mobile manipulator. The policy maps base pose (`base_pos`, `base_bearing`), arm state (`q_mes`, `qd_mes`) and `TARGET_EE_POS` to `torque_arm` (7) and `wheel_cmds` (4), and is trained on expert trajectories from `robotnik_arm_control.py`.

A working prototype must also be shown in person at a lab session between Fri 30 Oct and Fri 6 Nov. The demo is unmarked but required to complete the assessment.

**Success means:**

- The learned policy reaches the target within 10 mm on the in-distribution test set (T1). The expert's own success rate is the ceiling it is compared against.
- The results answer the brief's four evaluation questions: accuracy, generalisation, robustness and efficiency.
- Every design choice in the report is backed by our own ablation or validation result. The marking scheme rewards specific justification over generic statements.
- Every number comes from `results.csv` and can be reproduced from a clean clone by following the README.

## Alignment with the brief

Every question the brief asks has a decision, the experiment that justifies it, and the figure or table where the marker sees it. The tables follow the brief's own slides: the task outline (slide 4), Methodology (slide 5) and Evaluation (slide 6).

### Task outline (slide 4)

| Brief task | What we build | Code |
| --- | --- | --- |
| Implement a regressor: base pose, arm joints, target → `torque_arm`, `wheel_cmds` | BC-MLP policy at 100 Hz; ACT as an optional comparison | `train_mlp.py`, `policies.py` |
| Generate training data (expert trajectories) | Markovian version of the template controller, run on randomised scenarios with DART noise, then DAgger relabelling | `expert.py`, `scenarios.py`, `collect.py`, `dagger.py` |
| Design an evaluation methodology | Closed-loop runs on three frozen scenario sets (T1–T3), plus offline and efficiency metrics | `evaluate.py`, `test_sets.py` |
| Justify your choices | Each design choice below is paired with an ablation or a validation result | `results.csv`, `plots.py` |

### Methodology (slide 5)

| Brief question | Our decision | Justified by | Shown in report |
| --- | --- | --- | --- |
| Quantity and quality of data needed? | About 2,000 expert episodes (1–2 M samples at 100 Hz). DART noise on executed actions with clean labels, then 4+ DAgger rounds. Split by scenario seed, never by timestep. | Learning curve at 500 / 1,000 / 2,000 episodes; ablations without DART and without DAgger | Validation success vs episodes and vs DAgger round; dataset size per round |
| Would extra sensors or input data help? | Add derived inputs to the raw state: target in the base frame, sin/cos of bearing error, end-effector error from forward kinematics (pinocchio), phase flag. 27 inputs in total. | Raw inputs only vs derived inputs; world frame vs base frame | Ablation rows on T1 and T2 |
| Single instant vs temporal interval? | Single state. The Markovian expert's action depends only on the current state, so history should be unnecessary. | Single state vs a 5-step stacked history (and ACT chunking if time allows) | Ablation row plus error-over-time curves |
| ML regressor architecture? | MLP, 3×256 with LayerNorm, separate heads for wheels (4) and arm torques (7) | Width and depth sweep (2×128, 3×256, 4×512) on validation success | Success vs parameter count |
| Training method and hyperparameters? | Supervised BC with Huber loss, AdamW and a cosine schedule. Near-target samples weighted up. Normalisation from training data only. Models chosen on closed-loop validation success, not offline loss. | Near-target weighting on vs off; offline loss vs closed-loop success compared across checkpoints | Hyperparameter table; loss curves next to validation success |

### Evaluation (slide 6)

| Brief question | Test setup | Metrics | Shown in report |
| --- | --- | --- | --- |
| How accurate? | T1: 200 in-distribution scenarios with unseen seeds; the expert as a reference | Success within 10 mm, final EE error (mm), time to reach (s). Offline torque and wheel RMSE on held-out validation episodes. | Main results table, every policy against the expert |
| How generalisable to targets and initial conditions? | T2: a 72-target grid (8 directions × 3 distances × 3 heights). T1's randomised start poses. Targets beyond the training range (up to 3.5 m) in T3. | Success and final error per grid cell; success vs distance | Workspace heatmap |
| How robust to adversities? | T3: 100 scenarios with friction ×0.5 / ×1.5, link masses ±25%, double sensor noise, a mid-episode push, unreachable targets | Success drop from nominal; recovery after the push; behaviour on unreachable targets | Robustness bar chart per perturbation |
| How efficient (memory, parameters, compute)? | Every model on the same laptop, CPU inference | Parameter count, model size (MB), inference time per step against the 10 ms budget, training time, data-collection time | Efficiency table; success vs inference time |

Every success rate is reported with a 95% Wilson confidence interval. The main MLP is trained with 3 random seeds, so differences between ablations can be told apart from noise.

## Reaching the 80–100% band

The 80–100% band requires everything in the 70–79% band plus well-justified novel ideas. The plan targets each criterion with something a marker can check in the report or the repo.

| Marking criterion | What the plan does | What the marker sees |
| --- | --- | --- |
| **70–79:** only extremely minor technical issues | Splits by scenario seed, so no timestep leaks between sets. Normalisation comes from training data only. Models are chosen on validation, and the test set runs once. The expert is verified at 19/20 before any data is collected. | Splits, seeds and the selection rule stated in Methodology; one test run per model in `results.csv` |
| **70–79:** well documented and reproducible | Fixed seeds, configs in Git, every run logged with its checkpoint path. The README gives a command and the expected outputs for each script, and it is checked on a clean clone on Thu 5 Nov. | README, repo link, a hyperparameter table that matches the configs |
| **70–79:** excellent understanding of the method and of how code resources are used or modified | The report explains why the template controller cannot be cloned or relabelled as-is: it stores `q_des_arm` and a phase between steps. It then shows how it was made Markovian, and how pinocchio forward kinematics produces the EE-error input. The README separates external, modified and from-scratch code. | A Methodology subsection, "From the template controller to a Markovian expert"; the README's code-provenance list |
| **80–100:** novel ideas in method design, well used and justified | Plain behaviour cloning drifts once the policy leaves the expert's states (covariate shift). DART noise and DAgger relabelling target that drift, and the Markovian expert is what makes relabelling possible. Derived base-frame inputs encode the task geometry. | Each idea has its own ablation (no DART, no DAgger, raw vs derived inputs), so its value is a measured number, not a claim |
| **80–100:** novel ideas in experimental validation | The T2 workspace grid shows where failures occur, not only how often. The T3 suite tests adversities outside training. Offline loss is compared with closed-loop success, and accuracy is set against inference cost. Every rate carries a confidence interval over 3 seeds. | Workspace heatmap, robustness chart, success-vs-cost plot, intervals in every table |

**Lower-band traps to avoid**

- **Generic justification (caps the mark at 60–69%).** Every claim in the discussion cites a number from our own results, for example "DAgger raised T1 success from x% to y%". Never write "DAgger is known to help".
- **Code and report disagree (caps the mark below 60%).** Report tables are generated from `results.csv`, and every stated hyperparameter is checked against the configs on Thu 5 Nov.
- **Using the template as a black box (50–59%).** The Methodology must say what the template controller does and what was changed, not just that it was used.

The band is the markers' call, and novelty only counts when results back it. If an idea shows no gain, for example DAgger adds little, report and explain that honestly. A well-analysed negative result still counts as critical thinking.

## Study guide

Fourteen topics underpin the plan. They are ordered by the date each is first needed, so reading can keep pace with coding. The first six are needed before the expert is verified on Mon 26 Oct. Each row names the specific part to read, and full citations follow the table.

| Needed by | Topic | What to understand | Why this project uses it | Read |
| --- | --- | --- | --- | --- |
| Sat 24 Oct | Behaviour cloning and covariate shift | BC is supervised regression from states to expert actions. A cloned policy visits states the expert never did, and its errors compound: Ross et al. bound BC's cost as growing with the square of the horizon. | It is the brief's core task, and the reason the plan adds DART and DAgger | [Pomerleau 1988](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html); [Osa et al. 2018](https://arxiv.org/abs/1811.06711), behavioural cloning chapter; [Ross et al. 2011](https://arxiv.org/abs/1011.0686), Sections 1–2 |
| Sat 24 Oct | Markovian policies | A policy whose action depends only on the current state can label any state it is shown. The template stores `q_des_arm` and a phase between steps, so it cannot. | Needed for clean labels and for DAgger relabelling. This is the template change the report must explain. | [Osa et al. 2018](https://arxiv.org/abs/1811.06711), policy representation; `robotnik_arm_control.py` |
| Sat 24 Oct | PyBullet and Pinocchio | Stepping the simulation, headless DIRECT mode, joint motor control; computing link placements and Jacobians from joint angles | Every rollout, data run and evaluation runs on them | [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit); [Pinocchio docs and tutorials](https://stack-of-tasks.github.io/pinocchio/) |
| Sat 24 Oct | Frames and forward kinematics | Homogeneous transforms, moving a point from the world frame to the base frame, end-effector position from joint angles | The derived inputs: the target in the base frame and the EE error | [Lynch and Park](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 3–4 |
| Mon 26 Oct | Arm dynamics and torque control | How joint torques relate to q, qd and acceleration; PD control plus gravity compensation | `torque_arm` is a torque output, which explains why qd must be an input | [Lynch and Park](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 8 and 11 |
| Mon 26 Oct | Wheeled mobile base | How four wheel speeds move and turn the base; heading (bearing) error | `wheel_cmds` outputs and the sin/cos bearing inputs | [Lynch and Park](https://hades.mech.northwestern.edu/index.php/Modern_Robotics), Ch. 13 |
| Tue 27 Oct | Scenario and physics randomisation | Sampling targets, start poses and physics parameters so training covers the conditions seen at test time | Generalisation (T1, T2) and robustness (T3) | [Tobin et al. 2017](https://arxiv.org/abs/1703.06907) |
| Tue 27 Oct | DART noise injection | Inject noise into the executed action but store the clean expert action as the label, and set the noise level | Puts recovery states in the data without querying the expert online | [Laskey et al. 2017](https://arxiv.org/abs/1703.09327) |
| Wed 28 Oct | MLP training recipe | Input normalisation, LayerNorm, Huber loss (robust to large torque errors), AdamW's decoupled weight decay, cosine learning-rate schedule | Every hyperparameter in the report needs a reason | [Ba et al. 2016](https://arxiv.org/abs/1607.06450); [Loshchilov and Hutter 2019](https://arxiv.org/abs/1711.05101); [Loshchilov and Hutter 2017](https://arxiv.org/abs/1608.03983) |
| Wed 28 Oct | Offline loss vs closed-loop success | A low validation loss does not guarantee the robot reaches the target, so choose checkpoints by rollouts | The model-selection rule and one of the report's comparisons | [Mandlekar et al. 2021](https://arxiv.org/abs/2108.03298) |
| Thu 29 Oct | DAgger | Roll out the learner, have the expert relabel the visited states, aggregate and retrain; the β mixing schedule; why a few rounds fix covariate shift | The main method contribution | [Ross et al. 2011](https://arxiv.org/abs/1011.0686), Sections 3–4 |
| Sat 31 Oct | ACT (optional) | A transformer that predicts chunks of future actions, with temporal ensembling and a CVAE | The architecture comparison, and a second answer to the temporal-interval question | [Zhao et al. 2023](https://arxiv.org/abs/2304.13705) |
| Sun 1 Nov | History inputs and causal confusion | Adding past states can make a cloned policy copy its own previous actions instead of reacting to the task | Justifies single-state input as the default and frames the history ablation | [de Haan et al. 2019](https://arxiv.org/abs/1905.11979) |
| Tue 3 Nov | Reporting with uncertainty | Wilson intervals for success rates; results across seeds; not over-reading small differences | Rigour in every results table, aimed at the 80–100% band | [Binomial proportion intervals](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) (summarises Brown, Cai and DasGupta 2001); [Agarwal et al. 2021](https://arxiv.org/abs/2108.13264) |

**References**

1. D. A. Pomerleau, "ALVINN: An Autonomous Land Vehicle in a Neural Network," NeurIPS 1, 1988. [link](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html)
2. T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel, J. Peters, "An Algorithmic Perspective on Imitation Learning," Foundations and Trends in Robotics, 2018. [arXiv:1811.06711](https://arxiv.org/abs/1811.06711)
3. S. Ross, G. J. Gordon, J. A. Bagnell, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning," AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
4. M. Laskey, J. Lee, R. Fox, A. Dragan, K. Goldberg, "DART: Noise Injection for Robust Imitation Learning," CoRL 2017. [arXiv:1703.09327](https://arxiv.org/abs/1703.09327)
5. T. Z. Zhao, V. Kumar, S. Levine, C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
6. P. de Haan, D. Jayaraman, S. Levine, "Causal Confusion in Imitation Learning," NeurIPS 2019. [arXiv:1905.11979](https://arxiv.org/abs/1905.11979)
7. A. Mandlekar et al., "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation," CoRL 2021. [arXiv:2108.03298](https://arxiv.org/abs/2108.03298)
8. J. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," 2017. [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
9. K. M. Lynch, F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*, Cambridge University Press, 2017. Free preprint: [link](https://hades.mech.northwestern.edu/index.php/Modern_Robotics)
10. J. L. Ba, J. R. Kiros, G. E. Hinton, "Layer Normalization," 2016. [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)
11. I. Loshchilov, F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
12. I. Loshchilov, F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
13. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, M. G. Bellemare, "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS 2021. [arXiv:2108.13264](https://arxiv.org/abs/2108.13264)
14. L. D. Brown, T. T. Cai, A. DasGupta, "Interval Estimation for a Binomial Proportion," Statistical Science 16(2), 101–133, 2001. Summary: [Wikipedia](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
15. [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit) and [Pinocchio documentation](https://stack-of-tasks.github.io/pinocchio/)

## Scope and deliverables

The must-haves meet the brief by themselves, so work can stop at any priority level and still hand in a complete project. The should-haves answer the brief's methodology questions directly, so they come before ACT. ACT starts only if the best MLP is tagged by Sat 31 Oct.

| Priority | Item | Brief question it answers |
| --- | --- | --- |
| Must | Markovian expert, DART data, BC-MLP, DAgger, frozen T1–T3 test sets | Regressor, training data, evaluation methodology |
| Must | Ablations: without DART, without DAgger | Quality of data needed |
| Must | Efficiency metrics for every model: parameters, size, inference ms per step, training time | How efficient is the solution |
| Should | Learning curve: 500 / 1,000 / 2,000 expert episodes | Quantity of data needed |
| Should | Single state vs 5-step history as input | Single instant vs temporal interval |
| Should | Engineered inputs vs raw inputs; world frame vs base frame | Would extra input data help |
| Could | ACT policy with 2–3 DAgger rounds | Architecture comparison |

**Deliverables at submission (Fri 6 Nov):**

- [ ] Report, max 6 pages in the LaTeX IEEE template:
  - Introduction: problem, assumptions, constraints
  - Methodology: what we implemented and why
  - Results: experiments, tables and plots, with interpretation
  - Conclusion: limitations and improvements
- [ ] Contribution statement, max 1 page, added as page 7
- [ ] Code on Moodle, or a GitHub link in the report if it exceeds the size limit
- [ ] README covering:
  - How to run each script and what outputs it writes
  - Which external code was used: the course template, RoboEnv/`simulation_and_control`, and any reference implementations
  - Which parts were modified and which were written from scratch
- [ ] In-person demo at a lab session between Fri 30 Oct and Fri 6 Nov (required, unmarked)
- [ ] `results.csv` committed. Data and checkpoints stay out of Git; Moodle does not need them.

## Technical approach

The method is behaviour cloning from a Markovian expert, with DART noise during collection and DAgger rounds until validation success plateaus.

*[Diagram in the online doc: reach pipeline · 7 stages, 1 loop]*

1. **Prepare the expert** (`expert.py`). Make the action depend only on the current state.
   - Replace the stored `q_des_arm` with `q + qd_des·τ`.
   - Choose the base phase from distance and heading error.
   - Run at 100 Hz and hold each action for 10 physics steps (physics stays at 1 kHz).
   - Gate: at least 19 of 20 random targets reached within 10 mm.
2. **Sample scenarios** (`scenarios.py`). Draw the target in polar form: 0.5–3.0 m, azimuth −π to π, height 0.1–1.25 m.
   - Start pose: base ±0.5 m, bearing −π to π, arm joints within ±0.2 rad of home.
   - Physics: friction ×0.7–1.3, motor gains ±20%, link masses ±15%, plus joint sensor noise.
   - Split by scenario seed, never by timestep.
3. **Collect data** (`collect.py`). Run the expert headless on 500 training scenarios, scaling to about 2,000.
   - DART: add Gaussian noise to the executed action and store the clean expert action as the label.
   - Compute normalisation statistics on training data only.
4. **Train the baseline MLP** (`train_mlp.py`). Use 3×256 layers with LayerNorm, separate wheel and arm heads, Huber loss, AdamW and a cosine schedule.
   - Weight samples near the target more heavily.
   - Offline loss is used only for early stopping.
5. **Run DAgger rounds** (`dagger.py`). Roll out the policy on about 100 new scenarios per round, mixing in the expert with probability β (0.5 decaying to 0).
   - The expert relabels every visited state, the new samples join D, and the MLP retrains on all of D.
   - Stop when validation success stops rising, usually after 5–10 rounds.
6. **Train ACT** (optional, `train_act.py`). Use a 5–10 step state history, predict chunks of 10–20 actions with temporal ensembling, and set a small KL weight.
7. **Final test** (`evaluate.py`). Run once on Tue 3 Nov (day 11) for the expert, best MLP, ACT and the ablations.

**Interface contract.** `interfaces.py` is fixed on day 1 and changed only in a commit of its own. The expert, MLP and ACT all implement the same `Policy` class, so `evaluate.py` can score the expert before any model exists.

```python
OBS_LAYOUT = ["target_dx", "target_dy", "target_dz", "sin_bearing_err", "cos_bearing_err",
              "q[7]", "qd[7]", "ee_err[3]", "phase_flag"]       # 27 values, base frame
ACT_LAYOUT = ["wheel_cmds[4]", "torque_arm[7]"]                  # 11 values
CONTROL_HZ = 100

class Policy:
    def reset(self, scenario): ...
    def act(self, obs) -> np.ndarray: ...                          # shape (11,)

SEEDS = {"train": range(0, 10_000), "val": range(10_000, 10_300), "test": range(20_000, 20_372)}
# .npz: obs (N,27) float32, act (N,11) float32, episode_id (N,), source ("expert"|"dart"|"dagger_k")
```

**Evaluation.** Every policy runs in closed loop on scenario lists, not on stored samples. Each run records success within 10 mm, time to reach, final error, base path length, torque jerk and finger drift. For efficiency, each model also records its parameter count, size on disk, inference time per control step (the budget is 10 ms at 100 Hz) and training time.

| Test set | Contents | Episodes | Measures |
| --- | --- | --- | --- |
| T1 In-distribution | Training ranges, unseen seeds | 200 | Generalisation to new targets |
| T2 Structured grid | 8 directions × 3 distances × 3 heights, nominal physics | 72 | Where in the workspace it fails |
| T3 Robustness | Friction ×0.5/×1.5, masses ±25%, double sensor noise, a mid-episode push, targets up to 3.5 m, a few unreachable targets | 100 | Behaviour outside training ranges |

Models are chosen only on the 300 validation scenarios. The test seed range is run only once, on Tue 3 Nov.

## Workstreams

Yilin writes all the code, so the three workstreams run one after another along the critical path, not in parallel. Long runs (data collection, DAgger retraining, ablations) are left running while the next script is written. ACT drops to stretch: it starts only if the best MLP is tagged by Sat 31 Oct.

| Workstream | Files | Delivers | Writes in report |
| --- | --- | --- | --- |
| 1 · Sim and expert | expert.py, env.py, scenarios.py, collect.py, dagger.py | Markovian expert, D0 with DART, every DAgger round's data, T3 perturbations, learning-curve datasets | Methodology: expert, scenario sampling, DART, DAgger |
| 2 · Learning | dataset.py, train\_mlp.py, train\_act.py, policies.py | Normalisation stats, π₀, retrained MLP each round, history and input ablations, ACT if time allows | Methodology: models, inputs, training and hyperparameters |
| 3 · Eval and report | interfaces.py, evaluate.py, test\_sets.py, plots.py, results.csv, README, the report | Expert baseline, frozen T1–T3, validation score after each round, efficiency metrics, the single test run, all figures | Introduction, Results, Conclusion, contribution statement, final cut to 6 pages |

The evaluate.py skeleton comes first, on day 1, so the expert is scored before any model exists and every later stage has a pass/fail number.

## Timeline and milestones

The plan runs from Sat 24 Oct (day 1) to the deadline on Fri 6 Nov (day 14). The brief's suggested week-1 milestone, data collected by 23 Oct, has already passed, so week 1 here compresses expert, data and baseline into five days. The other two brief milestones are kept: a prototype and an early report draft by Fri 30 Oct, then optimisation and the finished report by Fri 6 Nov.

*[Diagram in the online doc: 14-day solo plan · code, runs, report · 7 milestones]*

The Code row is what Yilin writes; the Runs row is compute left running meanwhile, so coding and data collection overlap without a second person. The shaded column is day 11, the only day the test set is touched.

**Milestone checks.** If one fails, fixing it is the next day's only task.

1. **Sat 24 Oct, Contract:** `interfaces.py` is merged, and `robotnik_arm_task.py` runs headless.
2. **Mon 26 Oct, M1 expert ok:** the expert reaches 10 mm on at least 19 of 20 random targets. `evaluate.py --policy expert --split val` prints a row.
3. **Wed 28 Oct, M2 π₀ scored:** D0 has 500+ episodes, and π₀ has a closed-loop validation score in `results.csv`. π₀ runs in the GUI, so there is something to demo.
4. **Fri 30 Oct, report draft:** the IEEE skeleton has drafts of the Introduction and Methodology. The demo window opens; book the first available lab session.
5. **Sat 31 Oct, M3 best MLP:** at least 4 DAgger rounds are done, validation success has plateaued, and the best checkpoint is tagged.
6. **Mon 2 Nov, M4 freeze:** ACT, the ablations and the learning curve are scored on validation, and efficiency metrics are logged. No training happens after this.
7. **Tue 3 Nov, final test:** the expert, best MLP, ACT and ablations each run once on T1–T3. Results are committed and never rerun.
8. **Fri 6 Nov, submit:** the report is at most 6 pages plus the contribution page, the README is checked on a clean clone, and the demo is done.

## Workflow

Work happens in one Git repo, `COMP_0245_2026_Final`. Code and configs live in Git; data and checkpoints stay outside it in git-ignored folders. `main` must always run `evaluate.py --policy expert`.

**Repo layout**

```
final/robotnik_arm_task/
  robotnik_arm_task.py  robotnik_arm_control.py    # original task and controller
  configs/robotnik_arm.json                        # only config the task loads
  models/summit_xl_arm_description/                # only robot model the task loads
  interfaces.py  expert.py  scenarios.py  env.py  collect.py  dagger.py
  dataset.py  train_mlp.py  train_act.py  policies.py
  evaluate.py  test_sets.py  plots.py  settings.yaml
  results/results.csv                              # committed: every run, one row
  data/  checkpoints/                              # git-ignored, kept locally
```

**Environment.** Build a Python 3.10 virtual environment and installs RoboEnv into it. RoboEnv provides `simulation_and_control`, `pinocchio` and `pybullet`. The current setup uses uv with a `.venv` outside the repo. RoboEnv itself is git-ignored, and pixi or conda also work.

- [ ] Day 1: run `robotnik_arm_task.py` headless
- [ ] Day 1: pin package versions in `requirements.txt`

**Git rules**

- Work on a short-lived branch per task, such as `expert-markovian` or `train-mlp`, and merge into `main` at least daily.
- Before each merge, check that `evaluate.py --policy expert` still runs. Changes to `interfaces.py` go in a commit of their own.
- Ignored by `.gitignore`: `*.npz`, `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, `runs/`, `wandb/`, `__pycache__/`. Record the path of every dataset and model in `results.csv`.

**Data files**

- Name files by round: `D0.npz`, `D_round3.npz`, `mlp_r3.pt`, `act_r2.pt`.
- Every evaluation appends one row to `results.csv` with these columns: `run_id`, `policy`, `ckpt`, `split`, `n`, `success_10mm`, `time_to_reach`, `final_err`, `path_len`, `torque_jerk`, `finger_drift`.
- Report numbers and plots come only from `results.csv`.

**Daily log.** Each evening, note what finished, what is blocked and today's validation score. A failed milestone check becomes the next day's only task until it is fixed.

## Risks and open questions

The biggest risk is the expert, because every later stage depends on it. Each risk below has an early sign visible in the daily log and a day by which the fallback is decided.

| Risk | Early sign | Fallback | Decide by |
| --- | --- | --- | --- |
| Markovian expert misses 10 mm | Fewer than 19/20 targets reached on Mon 26 Oct | Spend one more day on the expert. If it still fails, keep the original expert for D0, add its phase as an input, and compute the phase from distance and heading when relabelling. | Tue 27 Oct |
| Data collection too slow | Under 30 episodes/min headless | Run `collect.py` as parallel processes, one seed range each. Stop at 500 episodes and spend the time on DAgger rounds. | Tue 27 Oct |
| π₀ barely moves | Validation success near 0 on Wed 28 Oct | Check normalisation and the base-frame transform, then raise the near-target sample weight. | Thu 29 Oct |
| Nothing to show at the lab demo | No policy runs in the GUI by Thu 29 Oct | Demo the expert pipeline and whichever π exists. The brief accepts work in progress, but skipping the demo fails the assessment. | Thu 29 Oct |
| ACT not trained in time | No ACT validation score by Sun 1 Nov | Report ACT as partial. MLP with DART and DAgger is the main result, and the should-have ablations still stand. | Sun 1 Nov |
| Report over 6 pages | Draft past 5 pages on Wed 4 Nov | Move secondary plots to fewer, denser figures; cut generic background before results. | Wed 4 Nov |
| Code and report disagree | Report states a setting not in the configs | Report tables are generated from `results.csv` and configs in Git; check each stated hyperparameter against them on Thu 5 Nov. | Thu 5 Nov |
| Test set leaks into tuning | Any script imports the test seed range before Tue 3 Nov | The test runs once, on Tue 3 Nov. Results are committed and never rerun. | Ongoing |
| Sole coder loses 2 days (illness, other deadlines) | A milestone check slips a day | Cut to the must-haves: drop ACT first, then the history and input ablations. Keep the DART and DAgger ablations. | Same day |

**Open questions**

- [ ] Which lab session between Fri 30 Oct and Fri 6 Nov is the demo in?
- [ ] Where are data and checkpoints backed up?
- [ ] Where is the Git remote? The repo's `origin` still points at the course repo, so it can't be pushed there. A GitHub link can also go in the report if the code exceeds Moodle's size limit.
