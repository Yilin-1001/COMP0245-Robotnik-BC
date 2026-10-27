# A D8-W256：DAgger 首阶段结果与效果有限的分析

日期：2026-10-27。用户已确认并完成预检、R1采集、20,000次GPU更新及预先指定的纯学生评估。**当前R1没有解决reaching：选中模型在32 validation与固定32 fitting场景中均0次reach、0次持续保持。** 尚未执行R2–R4、额外seed、B、C0/C1或新test评估。

## 结果与选择

R1的5k、10k、15k、20k四个checkpoint各做8场景screening，reach/hold全部为0/8；最终误差中位数分别1544.1、2457.3、1714.9、2474.8mm，按冻结规则选中5k。只有该候选进入完整32 validation和固定32 fitting诊断，没有用fitting结果选模型。

| 纯学生闭环指标 | R0：BC step84000 | R1：选中step5000 |
|---|---:|---:|
| Validation reach / hold | 0/32 / 0/32 | 0/32 / 0/32 |
| Validation 最终误差中位数 | 1750.7 mm | 1713.2 mm |
| 固定fitting32 reach / hold | 0/32 / 0/32 | 0/32 / 0/32 |
| 固定fitting32 最终误差中位数 | 1696.6 mm | 1773.5 mm |

Validation误差仅下降约2.14%，fitting32误差上升约4.54%，没有任务成功率收益。虽然2.14%的validation改进使R1按预先冻结的字典序规则成为下一轮候选parent，**这不代表模型已可用，也不代表它优于所有历史不同结构/目标的模型**。C0/C1尚未完成，不能将这点误差变化因果归于DAgger。

## 为什么效果有限

### 1. R1仍主要覆盖专家纠偏后的状态，学生独立执行片段过短

实际专家执行比例75.0036%，学生约25%。10ms块混合产生14,497个学生连续控制片段：中位10ms，95分位30ms，99分位40ms，最长70ms；arm已激活的片段最长也只有70ms。

因此，采集轨迹大部分时间由专家持续拉回，而评估要求学生连续控制最多40s。R1虽然确实新增了学生干扰下的状态，却较少覆盖学生连续控制几百毫秒到数秒后产生的偏离。**这是“已解决分布偏移”仍不能成立的直接覆盖证据**；它尚未定量证明这些缺失状态单独造成全部失败。

首轮β=0.75是已确认的温和起步策略。DAgger本身是迭代访问状态、重新标注、聚合再训练的方法；只完成这一轮不能视为完整低β日程已失败，参见[Ross等原论文](https://proceedings.mlr.press/v15/ross11a.html)。增加轮数、改变块长度或接管策略都应使用新的冻结配置，不能在本轮中途修改。

### 2. 模型仍没有学到专家的局部反馈关系

在原validation的固定4096个hold行上，用自动微分检查物理单位速度导数。A教师在未限幅关节的目标为`∂τ_i/∂q̇_j = -30 δ_ij`，因为其IK目标只依赖位置，速度进入PD阻尼项。限幅边界条目按既有规则排除。

| 反馈诊断 | R0 | R1选中5k | R1末尾20k | 专家未限幅目标 |
|---|---:|---:|---:|---:|
| 速度导数对角项中位数 (N·m·s/rad) | −7.15 | −6.03 | −6.87 | −30 |
| 非对角速度耦合RMS (N·m·s/rad) | 5.87 | 5.17 | 5.36 | 0 |
| 最大逐轨迹hold平均力矩偏差 (N·m) | 3.50 | 2.79 | 3.05 | 相对专家为0 |

这些结果支持“预测动作值改善，速度纠偏映射仍未学对”：学生局部阻尼显著弱于教师，且存在教师没有的跨关节速度耦合。**单独这组导数不构成完整机器人闭环稳定性证明**，位置反馈、阶段切换、惯性和底盘耦合也需检查。历史BC+J中即使速度导数接近教师仍有reaching失败，见[既有失败诊断](2026-10-27__report__reaching-failure-and-training-rollouts.md)，因此也不能断言只补阻尼就足够。

### 3. 新数据中的reaching学习信号仍较少，而且最难的阶段改善有限

轨迹等权采样下，新增64条只占全部285条的22.46%实际梯度样本，与新增行数占比22.49%以及学生执行比例25%是三个不同概念。对完整20,000×1024采样流在GPU重放，终点RNG与checkpoint完全相同，得到实际计数：

| 梯度样本来源 | Navigation | Approach | Hold |
|---|---:|---:|---:|
| 原始221条 | 7,694,774 | 4,167,225 | 4,019,374 |
| 新增64条 | 2,223,212 | 1,213,397 | 1,162,018 |

新增数据的approach行只占全部梯度样本约5.92%，其中又并非全是严重偏离后的恢复状态。arm权重0.8会提高所有阶段的手臂动作损失权重，并不会让reaching阶段获得80%的训练样本。

用每轨迹固定64行、先轨迹内平均再轨迹间等权的同口径探针比较R0与R1选中5k：

| 离线指标 | R0 | R1选中5k |
|---|---:|---:|
| D1 weighted Huber | 0.006825 | 0.003599 |
| D1 approach arm RMSE | 1.357 N·m | 1.256 N·m |
| D1 hold arm RMSE | 0.399 N·m | 0.131 N·m |
| 原validation approach arm RMSE | 0.776 N·m | 0.775 N·m |

新数据总体拟合确实改善，尤其hold；最困难的approach改善小，原validation的approach几乎不变。不能把总体loss下降等同于reaching控制已经学会。上节反馈诊断使用完整validation缓存与固定导数样本，统计权重与本表固定探针不同，不混比二者RMSE。

### 4. 更多相同目标更新没有显示持续收益，容量不足尚未被证实

本轮完整训练了20k，不是在5k提前停训；5k是由固定screening从四个候选中选出来的。20k的原validation固定探针Huber反而由R0的0.005628变为0.005829，screening最终误差也高于5k。现有证据不支持“只要继续同一数据和目标多训就会成功”。

D8-W256已有8×256隐藏层和472,585个参数。本轮没有改变容量，无法排除表达能力或优化的限制，但也没有证据把失败主要归于网络太小。更值得先区分的是：连续学生状态覆盖不足，以及动作值回归没有识别正确反馈。

## 已核对的替代解释

- 专家在同一64新目标独立执行时64/64持续保持；混合采集63/64。后者包含专家控制，不能当作模型成功率。
- D8预检：专家独立8/8；reaching交给学生0.25s或1s后恢复均8/8，3s为7/8；reset接管1s/3s均8/8。
- 唯一未持续保持的D1轨迹`dagger_train-0029`，重放3297和6186步的两个前缀后切回专家，均按原谓词持续保持成功。没有因任务失败删除该轨迹。
- 对D1抽查1620个原始执行前状态，重算专家标签与学生预测一致；部署特征与缓存特征一致。原始专家轮速标签与执行限幅动作分开保存，未将其他轨迹同时间索引的动作拿来标注。
- 新归一化未重新拟合。跨轮只加载R0权重，AdamW从零开始；GPU中断续训与连续训练的权重差为0。

因此，“专家在这些目标上不能完成”或已检查的标签一步错位/归一化漂移不是目前最有证据的解释；更严重、尚未访问的偏离状态仍可能超出专家可恢复范围。

## 本轮配置和执行

网络与R0相同：21输入、8个256宽隐藏层，每层Linear→LayerNorm→SiLU，输出7力矩+2独立轮速。数据为Clean_253中的221 fitting加64新场景，32 validation严格排除。原始2,658,124行，新增771,163行，累计3,429,287行。

保持arm/wheel=0.8/0.2标准化group-mean Huberδ1；新建AdamW，lr1e-4、weight decay1e-4、batch1024、warmup500+cosine20k，不使用Jacobian损失。1kHz重算专家/学生动作，每10步选择完整控制器。

训练和四次screening墙钟186.4s，GPU峰值约1.91GiB，主训练进程CPU峰值RSS约3.43GiB；专家参考和混合采集分别约102.6s、96.5s，另有预检、GPU smoke及最终闭环评估开销。

采集代码提交`0be02ab`，正式训练代码提交`407666f`，审计/绘图与旧证据校验修正提交`4fd9fe7`。汇总曾把旧报告driver的哈希误作共享物理evaluator哈希，现分别验证二者；共享evaluator与R0训练提交中的文件哈希完全一致。修正只影响旧证据复用和报告，没有重训或改写本轮checkpoint/回放。

全部217项测试通过，包括原clean回归、动作时序、失败前缀、冻结场景、数据混入拒绝、热启动/续训、采样流与证据校验。另有真实PyBullet采集smoke、GPU训练smoke和双模拟器headless smoke。

## 文件、曲线和物理demo

- [冻结执行协议](../../final/robotnik_arm_task/experiments/protocols/2026-10-27__protocol__dagger-a-r1.json)。256个候选训练场景的完整日程与64个sealed test几何已冻结，实际仅使用R1的64个训练场景；没有运行新test策略评估。
- [完整R1结果目录](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/)、[闭环汇总](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/evaluation_complete.json)、[控制导数诊断](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/control_diagnostics.json)。
- [训练/validation同口径曲线及LR](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/plots/2026-10-27__learning-curves__dagger-a-r1.png)，同目录有PDF/SVG；[四节点纯学生闭环曲线](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/plots/2026-10-27__closed-loop__dagger-a-r1.png)。开发曲线用validation，独立新test保持封存。
- [选中模型best.pt](../../final/robotnik_arm_task/checkpoints/dagger/A/bc/dagger-A-bc-r1-d08w0256-s0-1175a0015097/best.pt)、[部署inference.npz](../../final/robotnik_arm_task/checkpoints/dagger/A/bc/dagger-A-bc-r1-d08w0256-s0-1175a0015097/inference.npz)。best.pt SHA256 `049993cfa92c5d1a3c2eb36472d023c709fabd8bf5a6c80fdb82e493ba83b7fd`。step20000也保留，但不是选中的候选。
- [专用demo清单](../../final/robotnik_arm_task/experiments/demo_models/2026-10-27__demo__dagger-a-r1.yaml)，用于观察本轮失败表现，未替换其他实验的默认模型。

在`final/robotnik_arm_task/`运行两个真实模拟器窗口：

```bash
/media/yilin/data/ML_Final/.venv/bin/python -m evaluation.live_compare \
  --teacher A --objective bc \
  --protocol experiments/demo_models/2026-10-27__demo__dagger-a-r1.yaml \
  --scenario train-0143
```

这会加载R1选中5k模型与A专家；SPACE暂停、R重置、Q关闭，可用`--target X Y Z`指定目标。当前没有纯学生成功案例可作为本轮成功demo。

## 下一步建议（尚未启动）

优先做有限的、预算匹配的诊断：增加学生连续接管长度并降低专家占比，确认是否能覆盖真实失败区域；同时比较同D8下plain BC与反馈导数约束/显式阻尼残差控制。后者会改变目标或结构，应单独立项，并检查位置反馈与切换边界，不能许诺仅加入速度Jacobian就成功。

在归因“DAgger带来收益”之前完成C0仅继续BC与C1专家新场景数据的匹配对照。不要同时改网络深度、损失、接管策略与采样比例，否则无法区分哪项有用。当前结果未达到预定额外seed投入门槛，也不授权自动扩展后续轮次。
