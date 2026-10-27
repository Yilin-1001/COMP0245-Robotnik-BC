# DAgger A 方案复查与修订汇总

2026-10-27。本次为设计审查：检查已有方案、实际A教师/数据接口、训练循环及最新D8结果，未实现或启动DAgger。

建议采用刚完成的D8-W256（0.8/0.2、seed0、84k）作为R0，保持网络与损失，先完成新R0的恢复预检及R1，再审阅R2–R4。该初始化与预算仍为待确认建议。

## 核对后的事实

- D8-W256：8×256隐藏层，472,585参数；原221 fitting与32 validation闭环reach/hold均为0。完整证据见[D8报告](2026-10-27__report__d08w0256-single-run.md)。
- `data_collection/dagger.py`仍在main中抛出NotImplementedError；`training/dagger/`尚不存在，不能直接启动完整DAgger。
- clean配置校验只接受clean regime，clean来源校验排除dagger/noise；需要独立DAgger loader/validator与warm-start入口。
- A教师从当前Observation计算动作；collect与evaluation显式使用lead_time=0.50，类默认0.15不匹配本数据。
- 原clean监督标签来自限幅前的教师接口输出a_expert；环境执行a_executed才按执行限制裁剪，训练loader取a_expert的7力矩与FR/FL两轮速。
- 现有engine会新建AdamW，接收传入模型，但clean prepare重新估计normaliser、fit验证器仅支持clean；跨配置的latest恢复会被拒绝。DAgger须加载父模型权重与原normaliser，新轮update从0开始；同轮resume才恢复AdamW/scheduler/RNG。

## 复查发现与修改

| 项目 | 原方案/风险 | 本次修订建议 |
|---|---|---|
| R0 | 旧M6 BC，step6000、历史0.5/0.5 | 最新D8 BC，step84000、0.8/0.2；保持同结构便于比较增加访问状态数据的效果 |
| 动作loss | 仍写0.5/0.5 | Huber δ1，arm0.8/wheel0.2 |
| 专家恢复 | 旧M6 BC+J探针不能证明D8可恢复；从reset短时放手会偏向导航 | 新D8在专家reaching激活后放手0.25/1/3秒再接管，另做reset后1/3秒探针 |
| 新目标参考 | 几何在范围内不保证专家独立成功 | 本轮64个新场景先跑专家参考，保留失败；与偏离后恢复能力分别报告 |
| 标注对齐 | 容易误用执行动作或专家独立轨迹 | 同一s_t实时查询专家；原始教师标签与执行限幅分开，保留T+1/T对齐 |
| β含义 | 容易把专家执行比例误解为标签比例 | β只控制完整动作向量的执行来源；每1ms均查询专家，10ms仅固定控制器身份 |
| 采样 | 轨迹等权使短失败前缀单行权重较高；R1新轨迹仅约22.5% | 保留全部有效失败前缀，报告每阶段/来源/轨迹长度及实际采样占比；不自动加阶段权重 |
| 评估开销 | 每2k做8场景回放，每轮80个screening rollout | 改为5k/10k/15k/20k四节点，每轮32个screening rollout；每2k仍记离线指标 |
| 训练场景诊断 | 原方案主要看validation | 每轮选中候选加固定32 fitting；最终补齐221 fitting，均不用于晋级 |
| 内存与恢复 | 聚合数据增长，原engine全量上传GPU | 预检累计数据GPU/CPU峰值；原归一化冻结；跨轮只warm-start，同轮才resume |
| 归因 | 更多updates或更多新目标也可能改善 | 完成C0继续BC、C1新场景专家独立采集的匹配预算对照后再归因于DAgger |

## 建议的训练配方

数据：Clean_253中的221 fitting作为D0，32 validation排除；每轮64个新训练场景，原几何范围与nominal physics，冻结分格、Sobol索引与互斥检查。输入、1kHz接口、执行限制、教师版本及normaliser均固定。

每轮：从晋级模型权重开始，AdamW β=(0.9,0.999)、eps1e-8、weight decay1e-4、batch1024、峰值LR1e-4、warmup500+cosine、20,000 updates；动作项0.8/0.2标准化Huber δ1，纯BC。学习率1e-4是新增数据微调的建议峰值，和R0预训练3e-4区分，所有对照使用相同微调配方。

| 阶段 | 新场景数 | β：执行专家动作的概率 | 新增updates | 执行建议 |
|---|---:|---:|---:|---|
| 预检 | 8个固定fitting及独立短轨迹 | 1、0及混合 | 仅管线smoke | 先检查专家、标注和恢复；不进入D1 |
| R1 | 64 | 0.75 | 20,000 | 建议首个正式阶段 |
| R2 | 64 | 0.50 | 20,000 | R1报告后审阅 |
| R3 | 64 | 0.25 | 20,000 | 后续计划 |
| R4 | 64 | 0.00 | 20,000 | 学生全程执行，专家仍全程标注 |

累计Dr=D0∪D1∪…∪Dr，每轮保留所有历史有效数据；在全部累计轨迹中等权抽轨迹、再在该轨迹有效行中等权抽时间点。请求β与实际专家执行比例分别记录，不将混合采集成功率当作纯学生成功率。

## 新R0身份与预检门槛

- 父模型目录：`checkpoints/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/clean/A/bc/clean-A-bc-clean253-d08w0256-lr3e-4-s0-selection-ca837b8fa585`。
- best.pt SHA256：`a439ce195692874c6a712c05d2d0de6a13c22e8061ac5af2783805e5034dcd35`；inference SHA256：`2300123d9cd98e74f1db7db554789226f72e91fa8b8b377ff1166abd4e5520ad`；normaliser文件SHA256：`d69994cf4ac00b84213113aabc6ab51560df3ec6bcd430795c7cfc8357f1ea53`。
- normaliser内容SHA256保持`1ba73a11d1bb7d581a034d86600e8f92d78a76760118e3c76130d3718ea31ef4`；本次已核对父文件真实哈希。
- 固定恢复场景（只按ID选取，未按效果挑选）：`train-0001, train-0035, train-0067, train-0108, train-0142, train-0178, train-0216, train-0255`。
- 建议工程门槛：专家完整接管8/8通过持续保持；reaching后D8放手0.25秒和1秒各至少6/8可由专家恢复，3秒报告边界。这些新D8探针尚未执行；旧模型恢复8/8不替代新结果。
- β0/1控制器来源、10步块切换、同状态专家标签、超限标签/执行限幅区分、失败前缀、T+1/T、保留集互斥与同轮resume先通过单元/集成检查，再开始R1。

## 验收与证据

所有评估使用纯学生β0、无专家接管：原reach/hold谓词、40秒上限；以32 validation的hold、reach、末端误差晋级，完全相同则保留历史模型，记录实际采集parent。每轮固定32 fitting只诊断；最终模型221 fitting与R0比较。

继续画D0固定训练探针与validation同口径loss/RMSE、LR，以及每轮DAgger源数据各自的拟合误差、reaching覆盖、归一化后状态分位数、专家/学生动作差和饱和；不得把不断变化的聚合训练集loss当作跨轮同分布曲线。

旧64 test已查看，仅作历史回归；新独立64场景需训练前冻结、开发时封存，最终配方与模型锁定后一次评估。C0/C1对照需相同R0、normaliser、每轮20k优化与晋级规则；C1复用新场景专家参考，匹配标签行数时记录确定性重放重复比例，不能称为独立状态数匹配。

## 算法依据与限制

DAgger将学生访问状态与这些状态上的专家动作累积为监督数据；原算法允许混合执行策略并以验证选模型，理论依赖其在线学习等假设。[原论文，算法3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)。[imitation官方文档](https://imitation.readthedocs.io/en/stable/algorithms/dagger.html)也明确区分实际执行来源与始终保存的专家动作。

本方案的10ms块混合、变长轨迹等权、非凸网络有限步优化及历史最好策略采集均为工程选择；不能直接声称满足原论文全部理论条件，也不能保证4轮稳定。A教师从观测推算平面底盘姿态，严重偏离中的有限标签未必可恢复；DAgger也不会自动提供闭环稳定性或修复隐藏状态缺失。B仍需单独设计。

完整修订见[DAgger A设计](../specs/2026-10-27__design__dagger-a.md)。本次只改文档；未采集新数据、未运行新恢复仿真、未启动DAgger训练。
