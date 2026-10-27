# Research References and Design Rationale

Reviewed: 26 October 2026.

This document consolidates the research discussed for the Robotnik imitation-learning project and the supporting references in the earlier project plan. It separates published findings, available research software, and their proposed application to this robot. It is an annotated reference guide, not a claim that the methods have already been implemented or validated here.

The [expert dataset specification](../specs/2026-10-26__spec__expert-datasets.md) defines the current collection contract: Dataset A uses the Markovian teacher with 21 default inputs; Dataset B uses the original stateful teacher with 26 default inputs. Both have 11 action labels. The body-frame unit gravity direction remains deferred. Earlier feature lists and methodological shorthand in the [project plan](../plans/2026-10-26__plan__project-overview.md) should be read in light of that specification and the qualifications below.

## Contents

1. [Problem formulation and theoretical distinctions](#problem)
2. [DART and DAgger: learning under distribution shift](#distribution-shift)
3. [Learning internal memory from history](#history)
4. [Estimating missing information and using privileged supervision](#privileged-information)
5. [Copycat behavior and targeted mitigation](#copycat)
6. [Supporting references: robotics, training and evaluation](#supporting)
7. [Research projects and implementation resources](#projects)
8. [Research-derived guidance for this project](#guidance)

<a id="problem"></a>

## 1. Problem formulation and theoretical distinctions

### 1.1 Physical state, teacher memory and student memory

The following notation is a project-level formulation of the controller-learning problem:

| Symbol | Meaning | Available to the deployed student? |
| --- | --- | --- |
| `x_t` | Full physical simulator state | Only through the declared observation interface |
| `o_t` | Deployable observation and known task references at step `t` | Yes |
| `z_t` | Original teacher memory before its update: joint reference, navigation phase and completion latch | No |
| `H_t` | Available history through `o_t`, optionally including previously executed actions | Yes, if retained |
| `m_t` | Student's own recurrent state or explicit reference estimate | Yes; maintained by the student |
| `a*_t` | Clean expert action label | During supervised collection or training-time queries only |

The original teacher can be represented schematically as:

```text
(a*_t, z_(t+1)) = Expert(x_t, z_t, task_context)
```

Identical current observations can correspond to different teacher memories and therefore different action labels. A single-observation regressor may average those labels. A recurrent student instead maintains:

```text
m_t       = GRU(m_(t-1), o_t)                # optional previous action is an ablation
a_hat_t   = ActionHead(m_t, o_t)
z_hat_t   = AuxiliaryHead(m_t)              # optional training supervision
```

The auxiliary head must use the recorded before/after convention consistently; it may predict the teacher's post-update reference even though `z_t` above denotes pre-update memory. Neither a GRU hidden vector nor its predictions need equal the teacher's exact internal representation. Learning useful behavior is the primary objective. See [R03](#r03), [R06](#r06) and [R10](#r10) for the research motivations and limitations.

The simplified teacher in Dataset A is a deterministic mapping from its chosen current features to an action. This makes a reactive baseline easier to define. It does not establish that the observation vector is the complete Markov state of the physical robot.

### 1.2 Three problems that require different responses

| Problem | What goes wrong | Relevant research | What the method does not establish |
| --- | --- | --- | --- |
| Distribution shift | Student errors lead to trajectories absent from demonstrations | DART, DAgger, sequence-model IL | That missing information becomes observable |
| Missing information or memory | Current observations do not determine the desired action | Sequence models, privileged reconstruction, RMA, A2D | That every hidden teacher variable can be recovered from history |
| Copycat or causal shortcut | History predicts expert actions offline, but the learned dependence fails in feedback | Causal Confusion, Fighting Copycat Agents, PrimeNet | That all past-action information is harmful |

History helps only when it contains relevant evidence. If two possible situations produce the same available history but require different teacher actions, no deterministic student using only that history can exactly reproduce both. More history, auxiliary labels and DAgger do not remove this identifiability limit. The practical response is to measure the ambiguity, improve deployable observations where permitted, or change the learning target or teacher. [R03](#r03), [R10](#r10).

### 1.3 Corrections to earlier planning shorthand

- A memoryless expert is convenient for isolated-state labels; it is **not a general requirement of DART or DAgger**. A stateful expert can label complete trajectories with a defined initialization and memory-update rule.
- Adding arbitrary Gaussian action noise is **DART-style collection**. Claiming a reproduction of DART requires implementing and reporting its noise-selection procedure.
- DART still requires the supervisor to respond during perturbed demonstration collection. It avoids requiring the supervisor to label learner rollouts in the same manner as DAgger; it does not generate corrective labels without a supervisor.
- DAgger's no-regret analysis is conditional. It does not guarantee that a few rounds solve the task, that an insufficient student policy class can match the teacher, or that any fixed success rate follows.
- A navigation phase reconstructed only from current distance and heading need not match the original controller's phase history or latched completion flag.
- The project's 1 kHz collection convention and reported torque-hold failures are local implementation and experimental facts, not universal requirements proved by a robotics textbook.

<a id="distribution-shift"></a>

## 2. DART and DAgger: learning under distribution shift

<a id="r01"></a>

### R01. DART

**Reference:** Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan and Ken Goldberg. *DART: Noise Injection for Robust Imitation Learning*. CoRL 2017, PMLR 78:143–156. [Paper and citation metadata](https://proceedings.mlr.press/v78/laskey17a.html). [Author explanation](https://bairblog.github.io/2017/10/26/dart/).

**Published method:** Perturb supervisor actions during demonstration collection so that the supervisor visits and corrects deviations. Select the noise distribution to approximate errors made by the learner, rather than treating arbitrary noise as the complete algorithm.

**Project application:** Collect clean and perturbed trajectories for both teachers. Save the clean expert label, sampled noise and final executed command separately. For B, update teacher memory along the actual perturbed trajectory. Pilot the perturbation scale and verify that the teacher still provides useful recovery behavior.

**Limit:** Action perturbations do not resolve hidden-state ambiguity or guarantee robustness to a different disturbance family. Noise on arm torques and wheel velocity targets also needs different units and scales. A fixed-noise baseline should be named accurately and compared at a controlled data budget.

<a id="r02"></a>

### R02. DAgger

**Reference:** Stephane Ross, Geoffrey J. Gordon and J. Andrew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011, PMLR 15:627–635. [Paper and citation metadata](https://proceedings.mlr.press/v15/ross11a.html).

**Published method and theory:** Iteratively collect trajectories under the current learner or an expert/learner mixture, query expert labels, aggregate the data and retrain. The analysis connects imitation learning to no-regret online learning under policy-induced distributions. Bounds depend on the loss, available policy class, optimization and assumptions connecting imitation error to task cost; a blanket “quadratic becomes linear” statement omits those conditions.

**Project application:** For A, label visited observations directly. For B, maintain a separate training-time teacher state along the same learner-driven episode, querying and advancing it once per control step. Store the sequence and replay it through the updated student when retraining.

**Limit:** Resetting the teacher at every query or borrowing `q_des` from a separate expert rollout defines different labels. DAgger also cannot make an inadequate observation history sufficient. Expert availability for training-time labels does not imply running the expert at deployment.

### Collection roles in this project

| Stage | Who drives the environment? | What is the supervised action target? |
| --- | --- | --- |
| Clean BC data | Expert | Clean expert action at the current pre-action observation |
| DART-style data | Expert plus recorded action perturbation | Clean expert action before the perturbation |
| DAgger data | Learner or declared expert/learner mixture | Expert action evaluated on that trajectory |

For B, all three rows require a consistent teacher-state update. A teacher with a latched navigation completion state may fail to recover after a later base displacement. That behavior must be assessed before describing its labels as corrective. This is a controller-specific constraint, not a limitation unique to DAgger.

<a id="history"></a>

## 3. Learning internal memory from history

<a id="r03"></a>

### R03. Sequence Model Imitation Learning with Unobserved Contexts

**Reference:** Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell and Steven Z. Wu. *Sequence Model Imitation Learning with Unobserved Contexts*. NeurIPS 2022. [Paper](https://papers.neurips.cc/paper_files/paper/2022/hash/708e58b0b99e3e62d42022b4564bad7a-Abstract-Conference.html). [Author code](https://github.com/gkswamy98/sequence_model_il).

**Published finding:** History can help infer unobserved context, but offline sequence imitation can learn to perpetuate earlier mistakes. The paper studies conditions for successful sequence imitation and supports collecting supervision on learner-induced histories. Its theoretical conclusions depend on assumptions such as realizability and recoverability.

**Project application:** Retain full B episodes and compare single-step BC, sequence BC and sequence DAgger with the same teacher and observations. Record transitions and recovery trajectories, not only smooth nominal reaching.

**Limit:** This is not a proof that a GRU reconstructs `q_des`, nor a prescription to use a fixed number of frames at any control frequency. A short frame stack at 1 kHz covers only milliseconds and may omit navigation history.

<a id="r04"></a>

### R04. Offline robot imitation and robomimic

**Reference:** Ajay Mandlekar et al. *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation*. CoRL 2021. [Paper](https://arxiv.org/abs/2108.03298). [Project](https://robomimic.github.io/). [Author code](https://github.com/ARISE-Initiative/robomimic).

**Published finding and resource:** The study examines how demonstration datasets, policy choices and evaluation affect offline robot imitation. The accompanying framework provides recurrent BC implementations and configuration examples.

**Project application:** Use its sequence batching, recurrent-policy configuration and evaluation patterns as implementation references. Train on ordered chunks, handle episode resets explicitly, and select models using closed-loop validation as well as action loss. The [BC-RNN example](https://github.com/ARISE-Initiative/robomimic/blob/master/examples/train_bc_rnn.py) is a practical starting point.

**Limit:** A reusable training framework does not establish suitable memory horizons, control rates or hyperparameters for this torque-controlled mobile manipulator. Its recurrent support is not evidence that recurrence alone solves partial observability or copycat behavior.

<a id="r05"></a>

### R05. GRU architecture

**Reference:** Kyunghyun Cho et al. *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP 2014. [Paper and citation metadata](https://aclanthology.org/D14-1179/).

**Role:** Foundational reference for the gated recurrent unit. Gates allow a learned hidden state to retain and update information over a sequence.

**Project application and limit:** A GRU supplies a trainable memory mechanism for B. It does not specify what that memory represents, guarantee long-term retention, or provide a robotics stability result. Expert-state prediction losses and closed-loop experiments are separate design choices.

<a id="privileged-information"></a>

## 4. Estimating missing information and using privileged supervision

<a id="r06"></a>

### R06. Recurrent belief learning with privileged reconstruction

**Reference:** Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun and Marco Hutter. *Learning robust perceptive locomotion for quadrupedal robots in the wild*. Science Robotics, 2022. [Paper](https://arxiv.org/abs/2201.08117). [Publisher DOI](https://doi.org/10.1126/scirobotics.abk2822). [Project](https://leggedrobotics.github.io/rl-perceptiveloco/). [Authors' workshop methods description](https://leggedrobots.org/lrws_2022/assets/pdfs/paper7.pdf).

**Published method:** A recurrent student builds a belief from available sensor histories. Training combines teacher-action imitation with reconstruction of privileged information, helping structure the learned representation. The deployed policy uses its own observations and recurrent state.

**Project application:** Train B with action loss plus optional prediction of `q_des_after - q`, navigation phase and completion. Store these as supervised targets, while excluding the true values from deployed inputs.

**Limit:** The paper concerns perceptive quadrupedal locomotion. Reconstructing this controller's joint reference and discrete memory is our adaptation, not a published result for Robotnik. Auxiliary prediction accuracy must be checked alongside control performance.

<a id="r07"></a>

### R07. Rapid Motor Adaptation (RMA)

**Reference:** Ashish Kumar, Zipeng Fu, Deepak Pathak and Jitendra Malik. *RMA: Rapid Motor Adaptation for Legged Robots*. RSS 2021. [Paper](https://arxiv.org/abs/2107.04034). [Project and supplementary material](https://ashish-kmr.github.io/rma-legged-robots/).

**Published method:** A privileged encoder provides an environment latent during base-policy training. An adaptation module learns to estimate that latent from recent proprioceptive and action history. Deployment uses the estimator and policy without privileged environment inputs.

**Project application:** This is a useful precedent for training an estimator with privileged targets and consuming its predictions at runtime. It motivates a separate explicit-state experiment in addition to a GRU action policy.

**Limit:** RMA estimates factors affecting dynamics; the original controller's `q_des` is evolving controller memory. These are different estimation problems. RMA also uses action history, so it does not support a universal ban on previous-action inputs.

<a id="r08"></a>

### R08. Adapting RMA for bipedal robots

**Reference:** Ashish Kumar, Zhongyu Li, Jun Zeng, Deepak Pathak, Koushil Sreenath and Jitendra Malik. *Adapting Rapid Motor Adaptation for Bipedal Robots*. IROS 2022. [Paper](https://arxiv.org/abs/2205.15299). [Project](https://ashish-kmr.github.io/a-rma/).

**Published method:** The approach includes policy fine-tuning with the learned adaptation module, addressing the effect of using an imperfect estimated latent.

**Project application:** Train and evaluate an explicit-state student with its own imperfect reference estimates. Training the action head exclusively with true teacher references would create a deployment mismatch.

**Limit:** Its reinforcement-learning fine-tuning procedure is not part of the current supervised dataset plan. Using predicted-state sequence training or DAgger here is a project-specific response to the analogous mismatch, not an A-RMA reproduction.

<a id="r09"></a>

### R09. Learning by Cheating

**Reference:** Dian Chen, Brady Zhou, Vladlen Koltun and Philipp Krahenbuhl. *Learning by Cheating*. CoRL 2019; proceedings published in 2020, PMLR 100:66–75. [Paper and citation metadata](https://proceedings.mlr.press/v100/chen20a.html). [Author code](https://github.com/dotchen/LearningByCheating).

**Published method:** A privileged driving agent supervises a vision-based student that does not receive the privileged inputs at deployment.

**Project application:** Supports separating teacher-only training information from deployable student observations in the dataset schema.

**Limit:** This is a general asymmetric-supervision precedent. It neither solves this controller's internal-memory estimation problem nor guarantees that a less-informed student can reproduce every teacher action.

<a id="r10"></a>

### R10. Robust Asymmetric Learning in POMDPs (A2D)

**Reference:** Andrew Warrington, Jonathan W. Lavington, Adam Scibior, Mark Schmidt and Frank Wood. *Robust Asymmetric Learning in POMDPs*. ICML 2021, PMLR 139:11013–11023. [Paper and citation metadata](https://proceedings.mlr.press/v139/warrington21a.html). [PDF](https://proceedings.mlr.press/v139/warrington21a/warrington21a.pdf). [Author code](https://github.com/plai-group/a2d).

**Published finding:** A privileged expert can choose behavior unsuitable for a partially informed student. Adaptive Asymmetric DAgger (A2D) jointly adapts expert and trainee instead of assuming a fixed privileged teacher always supplies an appropriate target. Partial observability must therefore be considered when interpreting imitation performance.

**Project application:** Dataset B omits true orientation and teacher memory. Evaluate whether its deployable student completes tasks, separately from matching every expert action. Investigate information mismatch alongside estimation and policy errors.

**Connection to relative-position inputs (project inference):** Relative navigation-goal and heading features can express B's known world-origin-based task reference without directly feeding world x/y coordinates. These features do not supply hidden `q_des`, phase or completion memory. A current-observation MLP may still learn useful task behavior when available features suffice in the sampled scenarios; this remains an empirical hypothesis. Generalization to new geometry and exact imitation under hidden-state ambiguity are different questions.

**Limit:** The paper does not validate our 26 inputs or guarantee memoryless BC success. Our fixed-teacher MLP baseline and hand-designed Markovian teacher are not implementations of A2D.

<a id="copycat"></a>

## 5. Copycat behavior and targeted mitigation

<a id="r11"></a>

### R11. Causal Confusion in Imitation Learning

**Reference:** Pim de Haan, Dinesh Jayaraman and Sergey Levine. *Causal Confusion in Imitation Learning*. NeurIPS 2019. [Paper](https://arxiv.org/abs/1905.11979). [Author code](https://github.com/pimdh/causal-confusion).

**Published finding:** Additional inputs can create spurious predictive dependencies that work on demonstrations but fail when the policy controls the system. The proposed approach uses structured hypotheses and interventions to identify relevant causal relationships.

**Project application:** Treat extra history and previous actions as hypotheses to test. Include controlled disturbances and feature ablations, measuring feedback behavior rather than interpreting a lower offline regression loss as sufficient evidence.

**Limit:** The paper does not show that all history is harmful. State estimation can require history. Its causal identification procedure is more involved than simply dropping one input feature.

<a id="r12"></a>

### R12. Fighting Copycat Agents

**Reference:** Chuan Wen, Jierui Lin, Trevor Darrell, Dinesh Jayaraman and Yang Gao. *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1b113258af3968aaf3969ca67e744ff8-Paper.pdf). [Author code](https://github.com/AlvinWen428/fighting-copycat-agents).

**Published method:** Observation histories can reveal previous actions, encouraging a policy to copy them. A target-conditioned adversarial objective discourages excess previous-action information in the learned representation while preserving information useful for predicting the current action.

**Project application:** First compare a current-observation policy, a history policy and a previous-action-copy baseline. If shortcut dependence is supported by recovery and transition tests, the adversarial method is a candidate targeted intervention.

**Limit:** The target-conditioned adversary uses current target actions during training; these are not deployed policy inputs. Suppressing all past-action information could also remove useful dynamics evidence. Adding a generic adversarial loss without reproducing its conditioning is not the published method.

<a id="r13"></a>

### R13. PrimeNet: Fighting Fire with Fire

**Reference:** Chuan Wen, Jianing Qian, Jierui Lin, Jiaye Teng, Dinesh Jayaraman and Yang Gao. *Fighting Fire with Fire: Avoiding DNN Shortcuts through Priming*. ICML 2022, PMLR 162:23723–23750. [Paper](https://proceedings.mlr.press/v162/wen22d.html). [Project](https://sites.google.com/view/icml22-fighting-fire-with-fire/). [Author code](https://github.com/AlvinWen428/fighting-fire-with-fire).

**Published method:** Generate a coarse prediction from task-relevant key inputs and provide it to the main model as a priming variable. This encourages a useful predictive dependence instead of an easier but unreliable shortcut.

**Project application:** A current-observation branch using joint state and task geometry could prime a history-based action model. This is a later ablation if copycat behavior is observed.

**Limit:** The choice of key inputs requires domain knowledge. A primed policy is not necessarily an additive residual policy, and the method does not reconstruct `q_des` by definition. The linked repository exposes image-classification code and lists CARLA imitation code as forthcoming at review time; it is not a complete released robot imitation implementation.

### Interpreting torque and action history

The following are project-specific timing and representation rules motivated by the concerns above:

| Candidate input | Interpretation |
| --- | --- |
| Current expert torque `a*_t` | The supervised target itself; using it to predict that action leaks the answer |
| Previous executed command `a_executed[t-1]` | Causally available at deployment, but its predictive value may become a shortcut |
| Measured torque available before the decision | A possible sensor observation with its own latency and actuator semantics; not automatically equal to the commanded torque |
| Torque sampled after executing `a_t` | Future information relative to the action being predicted; cannot be silently attached to `o_t` |
| Expert `q_des` | Teacher memory label; unavailable as a normal deployment input |

A previous torque generally does not uniquely identify a joint reference: desired velocity, gravity compensation, clipping and timing also affect the command. High action autocorrelation alone is not proof of copycat behavior; smooth control naturally produces correlated actions. Use perturbations and phase-transition tests to determine whether the policy reacts to new evidence or merely repeats its previous command.

<a id="supporting"></a>

## 6. Supporting references: robotics, training and evaluation

These entries preserve the remaining bibliography from the project plan. They provide background or implementation choices, rather than direct solutions to hidden teacher memory.

| ID | Reference and primary source | Role and qualification |
| --- | --- | --- |
| S01 | D. A. Pomerleau. *ALVINN: An Autonomous Land Vehicle in a Neural Network*. NIPS 1988. [Proceedings](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html) | Historical neural driving reference; not a modern solution to distribution shift. |
| S02 | T. Osa, J. Pajarinen, G. Neumann, J. A. Bagnell, P. Abbeel and J. Peters. *An Algorithmic Perspective on Imitation Learning*. Foundations and Trends in Robotics, 2018. [Paper](https://arxiv.org/abs/1811.06711) | Broad foundation for policy representations, BC and interactive imitation learning. |
| S03 | K. M. Lynch and F. C. Park. *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press, 2017. [Author book site and free preprint](https://hades.mech.northwestern.edu/index.php/Modern_Robotics) | Frames/FK: Chapters 3–4; dynamics/control: 8 and 11; mobile robots: 13. Robot-specific calibration and stability still require validation. |
| S04 | J. Tobin et al. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. IROS 2017. [Paper](https://arxiv.org/abs/1703.06907) | Visual domain-randomization precedent. It does not justify a particular dynamics-noise range, Sobol sampler or dataset size here. |
| S05 | J. L. Ba, J. R. Kiros and G. E. Hinton. *Layer Normalization*. 2016. [Paper](https://arxiv.org/abs/1607.06450) | Architecture option; input normalization and LayerNorm are different operations. |
| S06 | I. Loshchilov and F. Hutter. *Decoupled Weight Decay Regularization*. ICLR 2019. [Paper](https://arxiv.org/abs/1711.05101) | Basis for AdamW; not evidence for a particular learning rate or weight decay on this task. |
| S07 | I. Loshchilov and F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts*. ICLR 2017. [Paper](https://arxiv.org/abs/1608.03983) | Cosine scheduling and warm restarts. A cosine schedule without restarts should be described as such. |
| S08 | R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville and M. G. Bellemare. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS 2021. [Paper](https://arxiv.org/abs/2108.13264) | Supports uncertainty-aware comparisons and careful aggregation across runs. Applying those ideas to imitation learning is a methodological transfer. |
| S09 | L. D. Brown, T. T. Cai and A. DasGupta. *Interval Estimation for a Binomial Proportion*. Statistical Science 16(2), 2001. [Publisher DOI](https://doi.org/10.1214/ss/1009213286); [author-hosted manuscript](https://www.stat.purdue.edu/~dasgupta/publications/tr99-19.pdf) | Foundation for Wilson/other binomial intervals. Correlated timesteps are not independent success trials; repeated scenarios or seeds require an appropriate grouping model. |
| S10 | T. Z. Zhao, V. Kumar, S. Levine and C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*. RSS 2023. [Paper](https://arxiv.org/abs/2304.13705); [ALOHA project](https://tonyzhaozh.github.io/aloha/) | ACT predicts action chunks and uses temporal ensembling. Deferred in this project; action chunking is distinct from inferring internal memory with a GRU. |

### Software documentation

| Resource | Official source | Use |
| --- | --- | --- |
| PyBullet | [Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit) | Simulation stepping, motor commands and sensor semantics |
| Pinocchio | [Documentation](https://stack-of-tasks.github.io/pinocchio/), [repository](https://github.com/stack-of-tasks/pinocchio) | Kinematics, Jacobians and rigid-body dynamics |
| rliable | [Author repository](https://github.com/google-research/rliable) | Implementations accompanying S08; choose estimators appropriate to the actual evaluation design |

<a id="projects"></a>

## 7. Research projects and implementation resources

The links below distinguish research papers from reusable code. Linked availability was reviewed on the date above; no repository was installed or reproduction run as part of this documentation task. Record the exact revision, dependencies and license before incorporating code.

| Topic | Project or repository | Scope |
| --- | --- | --- |
| DART | [Project](https://berkeleyautomation.github.io/DART/), [BerkeleyAutomation/DART](https://github.com/BerkeleyAutomation/DART) | Author project and research implementation for R01 |
| DAgger and history | [gkswamy98/sequence_model_il](https://github.com/gkswamy98/sequence_model_il) | Author implementation for R03, including sequence imitation comparisons; not the original R02 code release |
| Recurrent BC | [ARISE-Initiative/robomimic](https://github.com/ARISE-Initiative/robomimic) | Robot imitation framework and BC-RNN examples |
| Privileged recurrent belief | [Perceptive locomotion project](https://leggedrobotics.github.io/rl-perceptiveloco/), [author method explanation](https://takahiromiki.com/explainers/perceptive-locomotion/) | R06 paper, demonstrations and method details; a complete original policy-training code release was not verified |
| RMA | [RMA project](https://ashish-kmr.github.io/rma-legged-robots/), [project-linked code](https://github.com/antonilo/rl_locomotion) | Locomotion training/adaptation resource linked by the authors; inspect coverage before assuming full reproduction |
| A-RMA | [Bipedal adaptation project](https://ashish-kmr.github.io/a-rma/) | R08 paper and demonstration material |
| Privileged driving teacher | [dotchen/LearningByCheating](https://github.com/dotchen/LearningByCheating) | Author implementation for R09 in CARLA |
| Adaptive asymmetric IL | [plai-group/a2d](https://github.com/plai-group/a2d) | Author implementation for R10 |
| Causal confusion | [pimdh/causal-confusion](https://github.com/pimdh/causal-confusion) | Author research code; check benchmark coverage rather than assuming a full robot stack |
| Copycat mitigation | [AlvinWen428/fighting-copycat-agents](https://github.com/AlvinWen428/fighting-copycat-agents) | Target-conditioned adversary and single-observation/history BC baselines |
| Priming | [AlvinWen428/fighting-fire-with-fire](https://github.com/AlvinWen428/fighting-fire-with-fire) | Partial release: NICO image-classification code; CARLA code listed as forthcoming |
| Action chunking | [ALOHA project](https://tonyzhaozh.github.io/aloha/) | Paper, demonstrations and linked ACT resources; deferred here |

<a id="guidance"></a>

## 8. Research-derived guidance for this project

### 8.1 Two student designs supported by the collection plan

| Design | Training | Deployment | Main risk to evaluate |
| --- | --- | --- | --- |
| GRU with optional auxiliary heads | Ordered B trajectories; action targets; optional reference/phase/completion targets | Current deployable observation plus the student's own recurrent state | Shortcut dependence, insufficient memory, ambiguous observations |
| Explicit reference-state student | Predict a reference update or reference estimate; supervise with recorded teacher state; train sequences using the student's own estimates | Observation plus the student's own reference and any required learned phase/completion state | Accumulated estimation error and mismatch between training with true state and deployment with predicted state |

Both designs run without the expert at deployment. The second is a project proposal inspired by estimation and asymmetric-learning work; it is not an off-the-shelf algorithm established by RMA. A joint reference alone does not encode all of the original controller's navigation memory. If an explicit design also retains a fixed low-level controller, specify which part is learned and keep the comparison with an 11-output action policy explicit.

A deterministic external task reference is a separate case: if a known trajectory generator can run independently from task context at deployment, its output can be a legitimate input. The original feedback-integrated `q_des` must not be relabelled as such a freely available task reference.

### 8.2 Evidence-to-decision map

| Project decision | Research basis | Status and required check |
| --- | --- | --- |
| Keep separate A and B datasets | R02, R03, R10; different local teacher semantics | Agreed design. A-MLP versus B-GRU compares complete approaches, not recurrence alone. |
| Keep full ordered trajectories and episode resets | R03–R06 | Agreed storage rule. Validate sequence timing and hidden-state initialization. |
| Record `q_des`, phase and completion as labels | R06–R09 | Agreed collection rule; auxiliary losses are an experiment. Never expose true teacher memory as deployed input. |
| Use student-generated memory at runtime | R03, R06–R08 | Required deployment constraint. Test drift and recovery using predicted states. |
| Keep clean labels separate from executed commands | R01–R02 | Required collection semantics, including clipping and action-noise records. |
| Advance B's teacher on the actual rollout | R02–R03 plus original controller update order | Project-specific stateful labelling rule. Check exactly one update per step and correct before/after snapshots. |
| Log previous commands without making them default inputs | R07, R11–R12 | Agreed logging/ablation choice. Useful estimation evidence and shortcut risk must both be tested. |
| Add targeted copycat mitigation only after diagnosis | R11–R13 | Later experiment. Compare transition and recovery behavior against action-copy and current-observation baselines. |
| Defer body-frame gravity direction | User-selected scope; R10 explains the resulting information mismatch | B remains partially observed. Use teacher-side orientation diagnostics without silently adding them to student inputs. |
| Use nominal physics before broader randomization | R01, S04 and local teacher validation | Project experiment order, not a published optimal curriculum. Keep disturbance conditions identifiable. |
| Evaluate closed-loop behavior and uncertainty | R02, R04, S08–S09 | Required evaluation approach. Include sustained hold, recovery, saturation and variation across runs. |
| Test a relative-feature MLP with incomplete teacher state | R10; project-specific feature construction | Baseline hypothesis. Measure task success and action error separately; increase capacity only after checking information limitations. |

### 8.3 Minimal comparisons and limits of the evidence

Within Dataset B, compare a current-observation MLP, GRU with action loss, GRU with auxiliary state losses, and an explicit-state policy if implemented. Keep teacher, deployable information, scenario split and data budget controlled. Compare with/without previous executed action separately. For data collection, distinguish clean BC, fixed-noise BC and later DAgger; only use the DART name without qualification if its noise-selection method is implemented.

No cited paper establishes that 21 or 26 inputs are optimal, that 512 episodes are sufficient, that a GRU exactly reconstructs this teacher's memory, or that excluding gravity direction is harmless. Those are collection choices or hypotheses to assess locally. The literature supports the experimental structure and highlights failure modes; results for this robot must come from its own evaluations.

For an efficient reading order, start with R02 and R01 for collection, R03 and R12 for the benefits and risks of history, R06 and R07 for privileged estimation, then R10 for information mismatch. Use R04 for implementation patterns and R13 if targeted priming becomes an experiment.

### 8.4 Local sources and document maintenance

- [Expert dataset specification](../specs/2026-10-26__spec__expert-datasets.md): authoritative field roles, dimensions, timing and collection design.
- [Markovian expert](../../final/robotnik_arm_task/data_collection/expert.py): the actual reactive teacher definition.
- [Original controller](../../final/robotnik_arm_task/robotnik_arm_control.py) and [task initialization](../../final/robotnik_arm_task/robotnik_arm_task.py): reference integration, navigation memory and call order.
- [Observation/action interfaces](../../final/robotnik_arm_task/core/interfaces.py): deployment information and action ordering.
- [Expert model accuracy notes](../reports/2026-10-24__report__markovian-expert-model-accuracy.md): local model mismatch and calibration evidence.
- [Project plan](../plans/2026-10-26__plan__project-overview.md) and [ablation strategy](../plans/2026-10-26__plan__ablation-strategy.md): earlier planning context; dataset details should follow the current specification.

When a method is implemented, record its variant and source revision in the experiment configuration. Cite the original paper for the method and the software project for reused code. Report project-specific modifications explicitly, and update this guide when new evidence changes a design decision.
