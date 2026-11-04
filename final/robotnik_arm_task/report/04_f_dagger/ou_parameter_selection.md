# OU 噪声强度与参数选取：实验依据

整理日期：2026-11-04。范围：归档已有实验与配置，说明 f-DAgger 的 OU 参数来源。本次未新增仿真、数据采集或训练。

当前方案使用固定配置的分段 OU 动作扰动。参数由候选方案及专家恢复试验检查后固定；采集时每条轨迹随机采样一次幅度系数，训练过程中不会自动拟合或更新 OU 参数。

## 1. 当前参数，以及每个参数有哪些证据

| 参数 | 当前值 | 已有实验支持及其边界 |
| --- | --- | --- |
| 手臂基础稳态标准差 | 0.75 N·m / channel | D2 pilot 比较过 0.75 和 1.5 N·m，选定较温和的候选值；后续还比较过学生在 0.375、0.75、1.5 三档训练数据上的表现 |
| 轮速基础稳态标准差 | 0.15 rad/s / wheel | 与手臂强度成对比较：0.75/0.15 和 1.5/0.3；没有单独优化两者的比例 |
| 相关时间尺度 τ | 0.2 s | 另一组专家压力测试比较过 0.05、0.2、1.0 s；该测试的初始化、门控及成功判据与正式采集不同，不能据此认定 0.2 s 对最终学生最优 |
| episode 幅度系数 α | Uniform(0.25, 1)，每条轨迹固定 | D2 confirmation 之后检查过这一采样范围；没有找到不同 α 范围之间的对照 |
| OU 初值 | 全部通道为 0 | D2 pilot 和正式采集使用同一选择；压力测试采用稳态初始化，不能混为一组实验 |
| 轮速噪声关闭 | 首次 reach gate 激活后永久关闭 | D2 pilot 与正式采集使用此规则；压力测试中 A 的门控是随当前 gate 切换，并非永久关闭 |
| 全部噪声关闭 | 首次任务到达后 1.5 s | 已验证该调度可运行；没有找到 1.5 s 与其他关闭延迟的消融 |
| 控制步长 Δt | 0.001 s | 跟随任务的 1 kHz 控制频率；没有作为独立噪声超参数搜索 |

基础标准差乘以 α 后才是该 episode 的 OU 稳态标准差。它不是噪声样本的最大值。由 Δt 和 τ 计算的保留系数为 ρ = exp(−Δt/τ) ≈ 0.995012，不单独调节。

对应文件：[当前配置](data/ou_noise/protocols/d2-s075-x2.yaml)、[参数证据表](tables/ou_parameter_evidence.csv)、[采集实现快照](code/ou_noise_snapshot/data_collection/collect_dart.py)。

## 2. 噪声幅度的初步选择：16 个 tuning 场景

以下计数来自原始 pilot CSV，专家 A 使用 lead_time=0.15 s。两档非零噪声都同时扰动手臂和轮速。

| 手臂 σ / 轮速 σ | A 到达 | A sustained hold | B 到达 | B sustained hold |
| --- | --- | --- | --- | --- |
| 0 / 0 | 16/16 | 16/16 | 15/16 | 15/16 |
| 0.75 N·m / 0.15 rad/s | 16/16 | 15/16 | 15/16 | 15/16 |
| 1.5 N·m / 0.3 rad/s | 16/16 | 14/16 | 14/16 | 14/16 |

原始报告记录：较低的 0.75/0.15 方案在独立 confirmation 之前被固定。这里每种条件只有 16 个场景，15/16 与 14/16 的差别支持当时选择较温和的方案，不能证明该组合是唯一最优值。

来源：[tuning/results.csv](data/ou_noise/pilot/tuning/results.csv)、[tuning/manifest.json](data/ou_noise/pilot/tuning/manifest.json)、[原始 D2 报告](data/ou_noise/pilot/2026-10-26__report__ou-noise-validation.md)、[汇总表](tables/ou_pilot.csv)。

## 3. 独立 confirmation 与 α 采样检查

confirmation 使用另一组 32 个场景。固定强度方案在每个场景上使用两个噪声种子，共 64 次受扰执行；α 采样方案在同一组 32 个场景上各执行一次，并非另一组独立目标。

| 条件 | A 到达 | A sustained hold | B 到达 | B sustained hold |
| --- | --- | --- | --- | --- |
| Clean | 32/32 | 32/32 | 31/32 | 31/32 |
| 固定 0.75/0.15 | 64/64 | 62/64 | 60/64 | 59/64 |
| α × (0.75, 0.15)，α ~ U(0.25, 1) | 32/32 | 31/32 | 31/32 | 31/32 |

这里的 sustained hold 检查的是撤去噪声后的恢复：首次任务到达后继续记录 3 s，在到达后 1.5 s 关闭噪声，最后 1 s 的全部末端误差都必须低于 10 mm。它不表示持续受扰时始终保持 10 mm 精度。

来源：[confirmation/results.csv](data/ou_noise/pilot/confirmation/results.csv)、[d2_sampled/results.csv](data/ou_noise/pilot/d2_sampled/results.csv)。每条轨迹的 JSONL、NPZ trace 与日志均保存在相应 cohort 目录。Trace 为 100 Hz 诊断记录，原仿真、噪声更新与成功检查运行在 1 kHz。

## 4. τ 的比较来自另一套专家压力测试

压力测试共有 2,112 条记录：66 个条件 × 2 个专家 × 16 个场景。其中 1,248 条为 OU 条件，覆盖三个 τ、手臂五档 σ，以及轮速四档 σ 的有/无门控条件。

下表是专家 A 的部分阈值汇总。每个值表示满足至少 15/16 通过条件的最大已测 σ；“≥8”只表示测试网格最高的 8 N·m 仍通过。

| τ | 手臂完成任务 σ（N·m） | 手臂受扰保持 σ（N·m） | 轮速无门控保持 σ（rad/s） | 轮速有门控保持 σ（rad/s） |
| --- | --- | --- | --- | --- |
| 0.05 s | ≥8 | 1 | 0.3 | 1 |
| 0.2 s | ≥8 | 0.5 | 0.1 | 1 |
| 1.0 s | 4 | 0.5 | 0.1 | 0.1 |

这些结果说明时间结构和门控会影响控制表现。该试验分别扰动手臂与轮速；不能用其中的单通道阈值直接证明正式采集的联合噪声配置最优。

来源：[原始 2,112 条记录](data/ou_noise/expert_stress/expert_noise_robustness.csv)、[原阈值表](data/ou_noise/expert_stress/disturbance_thresholds.csv)、[新增 OU 条件汇总表](tables/ou_stress_by_condition.csv)、[原始压力测试报告](data/ou_noise/expert_stress/2026-10-26__report__expert-operating-limits.md)。

`ou_stress_by_condition.csv` 按 teacher/channel/τ/gated/σ 分组，每组 16 条。`completed` 统计在 30 s 内受扰到达且随后恢复成功的 episode；`held_under_noise` 统计受扰 hold 样本中至少 90% 低于 10 mm 的 episode。两者不能与 D2 的最终 sustained-hold 计数直接合并。

## 5. 三套试验设置必须区分

| 设置 | 专家压力测试 | D2 选参 pilot | 正式 OU 训练数据采集 |
| --- | --- | --- | --- |
| A 的 lead_time | 0.50 s | 0.15 s（另有四目标 smoke 比较） | 0.50 s |
| OU 初始化 | 稳态分布抽样 | 0 | 0 |
| 轮速执行限幅 | ±6 rad/s | 约 ±5.273 rad/s | ±6 rad/s |
| A 的轮速噪声门控 | 当前 gate 为真时关闭，可能恢复 | 首次激活后永久关闭 | 首次激活后永久关闭 |
| 扰动通道 | 手臂、轮速分别测试 | 联合扰动 | 联合扰动 |
| 主要判据 | 受扰完成、受扰保持、撤扰恢复分别记录 | 撤扰后的最终 sustained hold | 撤扰后的最终 sustained hold |

正式采集的原始 episode 索引显示：221 个 fitting 场景 × 2 次噪声抽样 = 442 条轨迹，441 条通过 sustained-hold 检查。这是选定配置在正式专家和采集协议下的运行结果。训练数据构建只保留通过检查的轨迹。

来源：[正式采集 manifest](data/collections/ou_d2-s075-x2_manifest.json)、[442 条 episode 索引](data/collections/ou_d2-s075-x2_episodes.csv)、[正式采集汇总](tables/ou_collection.csv)。

## 6. 对学生的后续强度消融

下列条件都带 reach-gate 输入，不执行 DAgger；验证集为 3 个训练种子 × 32 个场景。

| 条件 | 手臂 σ / 轮速 σ | OU 采集轨迹数 | 验证 sustained hold | 记录来源 |
| --- | --- | --- | --- | --- |
| Clean only | 0 / 0 | 0 | 10/96 | 本机 v1 |
| 半强度 | 0.375 / 0.075 | 442 | 69/96 | 队友机器，2026-10-30 |
| 当前基准 | 0.75 / 0.15 | 442 | 83/96 | 本机 v1 |
| 双强度 | 1.5 / 0.3 | 442 | 84/96 | 队友机器，2026-10-30 |
| 双采集量 | 0.75 / 0.15 | 884 | 82/96 | 队友机器，2026-10-30 |

这组记录提供了强度敏感性的后续证据，并非早期 pilot 选参时已知的结果。双强度与基准只相差 1/96，且部分条件来自另一台机器，不足以认定其中一档更优。队友条件的逐 seed 计数和模型文件不在本机，当前汇总保留其转录来源。

来源：[强度对照表](tables/noise_strength_controls.csv)、[对照原始报告快照](data/ou_noise/strength_controls/2026-10-31__report__dagger-ablation-and-final-model.md)、[四种 OU 配置](data/ou_noise/protocols/)。

## 7. 可用于英文报告的表述

> We used a fixed, phase-gated OU action-noise protocol with a correlation time of 0.2 s. Base stationary standard deviations were 0.75 N·m per arm joint and 0.15 rad/s per wheel, multiplied by an episode-wise scale sampled uniformly from [0.25, 1]. The base scale was selected in a 16-scenario pilot and checked on a separate 32-scenario cohort. Under the subsequent training-data collection protocol, 441 of 442 expert episodes passed the sustained-hold check. Earlier stress tests and later student ablations provide supporting sensitivity evidence, but do not establish global optimality of the chosen parameters. The pilot and final collection used different expert lead times, which are reported separately.

## 8. 文件完整性与复现范围

- [数据目录说明](data/ou_noise/README.md) 给出原始数据、配置、脚本和摘要的位置。
- [来源清单](data/ou_noise/source_inventory.csv) 记录复制文件的原路径、目标路径、字节数和 SHA-256；本次记录 521 个来源映射。
- [核查结果](data/ou_noise/verification_summary.json) 记录 376 条 pilot trace 的有限值、关闭门控后零噪声、Clean 零噪声和记录时长检查。
- 源脚本及历史报告按原样归档。原报告中的历史路径和状态说明保持不变；阅读当前归档请使用本文和数据目录说明中的链接。
- 原脚本中的绝对 TASK 路径仍指向本工作区。历史 manifest 的代码哈希描述当时版本；本次没有将其解释为当前控制器代码与历史版本一致，也未重新运行历史仿真。
