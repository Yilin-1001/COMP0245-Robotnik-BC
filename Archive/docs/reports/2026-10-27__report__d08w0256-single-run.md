# D8-W256 单模型训练与闭环验证结果

2026-10-27。已完成用户授权的单个D8-W256训练与物理仿真评估，30组容量矩阵未启动。

选中84k模型在全部221个fitting场景与32个validation场景中，task reaching与持续保持均为0次成功。本次增加到8层、256宽并训练84k仍未得到可完成reaching的闭环控制器。结论范围为本次单结构、单学习率、单seed实验。

## 实际训练配置

| 项目 | 实际值 |
|---|---|
| 网络 | 21输入；8个256宽隐藏层，每层Linear→LayerNorm→SiLU；7力矩+2独立轮速输出 |
| 参数量 | 472,585 |
| 数据 | 教师A、Clean_253，原221 fitting /32 validation，归一化只用fitting |
| 目标 | 纯BC；标准化Huber δ1；0.8 × mean_arm + 0.2 × mean_wheel |
| 优化 | AdamW；β0.9/0.999，eps1e-8，weight decay1e-4，batch1024 |
| 学习率与预算 | 3e-4；warmup500+cosine；84,000 updates，seed0，从头初始化 |
| GPU与训练耗时 | RTX 5070 Ti Laptop GPU；训练循环210.863秒（约3分31秒，包含训练期指标与保存） |
| 全流程GPU峰值分配 | 1.83 GiB，含训练期离线验证 |

训练前以独立模型进行200次更新的GPU预检，预检权重随后丢弃；正式模型使用原seed初始化流。保存每7000步checkpoint、NumPy导出与latest恢复状态，共12个节点；每2000步记录完整验证离线指标，12个节点记录同口径训练/验证探针。

## 物理执行结果

主模型仅在预定42k、84k两点之间用8个固定screening场景按原rank_rollout选择，84k末端误差中位数更低，因此选中84k。未按fitting结果选择checkpoint。

| checkpoint / 场景集 | 场景数 | Task reaching | 持续保持 | 末端误差中位数 (mm) |
|---|---:|---:|---:|---:|
| 42k screening | 8 | 0/8 | 0/8 | 1813.0 |
| 84k screening | 8 | 0/8 | 0/8 | 1411.8 |
| 84k validation | 32 | 0/32 | 0/32 | 1750.7 |
| 84k fitting全部 | 221 | 0/221 | 0/221 | 1604.8 |
| 84k fitting固定子集 | 32 | 0/32 | 0/32 | 1696.6 |

任意时刻进入10mm（不限制t>8秒）：validation 0/32，fitting 0/221。所有回放无执行异常；任务未成功的场景均计入统计。完整221场景物理回放耗时约10.5分钟，使用16个CPU worker。

原物理判据保持不变：t>8秒后真实末端误差<10mm才算task reach，继续运行3秒，最后1秒全部1001个观测<10mm才算持续保持；40秒上限。学生独立控制，无专家接管。validation复用同checkpoint的8个screening结果并补齐24场景，未把复用当成独立样本；固定32 fitting诊断子集从全部221结果中提取。封存test未访问。

## 学习曲线与阶段误差

[训练/验证损失、RMSE及学习率曲线](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/2026-10-27__plot__d08w0256-learning-curves.png) · [PDF](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/2026-10-27__plot__d08w0256-learning-curves.pdf)。

两侧探针均从每条轨迹等距取64行，包含首末行；先逐episode取均值再episode等权汇总。fitting共14,144行，validation共2,048行。数值用于同口径曲线，和完整validation逐行汇总的指标归约不同。

| 84k探针指标 | Fitting | Validation |
|---|---:|---:|
| 加权标准化Huber | 0.001453 | 0.005628 |
| Arm RMSE (N·m) | 0.279418 | 0.477255 |
| Wheel RMSE (rad/s) | 0.018089 | 0.038388 |

[机械臂分阶段误差曲线](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/2026-10-27__plot__d08w0256-phase-errors.png)。阶段标签只做记录分析，不作为学生输入或接管条件。

| 84k阶段Arm RMSE (N·m) | Fitting | Validation |
|---|---:|---:|
| navigation | 0.055075 | 0.055255 |
| approach | 0.467925 | 0.776003 |
| hold | 0.080009 | 0.270013 |

训练探针损失继续下降，验证探针误差后期出现回升与明显训练/验证差距；approach阶段的机械臂误差尤其突出。离线拟合改善尚未转化成学生闭环任务成功，全部训练场景也未task reach。此结果同时保留预测拟合与控制表现，不用低loss代替成功率。

## 模型、证据与复现

- run_id：`clean-A-bc-clean253-d08w0256-lr3e-4-s0-selection-ca837b8fa585`。
- [模型目录](../../final/robotnik_arm_task/checkpoints/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/clean/A/bc/clean-A-bc-clean253-d08w0256-lr3e-4-s0-selection-ca837b8fa585/)：选中`best.pt`和`inference.npz`对应84k；原`step_84000.pt/.npz`及其他节点均保留。
- [运行配置与契约](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/run_contract.json)、[模型与日程验证记录](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/verification.json)、[GPU预检](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/preflight.json)。
- [评估索引](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/results.csv)、[最终评估汇总](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/evaluation_complete.json)、[逐场景fitting结果](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/rollouts__fitting__step84000.csv)、[逐场景validation结果](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/rollouts__validation__step84000.csv)。
- [探针索引](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/probe_indices.json)、[逐节点探针指标](../../final/robotnik_arm_task/results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/probe_metrics.jsonl)；同目录保存评估与绘图脚本及原始日志。
- 实现与配置提交`a9053a6`；9项新增用例经历红/绿检查，完整191项测试通过。正式训练未修改原数值训练循环。
- 检查通过：实际CUDA、8层472,585参数、0.8/0.2、Clean_253划分、原normaliser内容哈希、manifest/入口/probe哈希、12份checkpoint与导出、12个探针节点、AdamW状态与实际LR日程。NumPy/PyTorch物理预测最大绝对差1.0673674e-05。

方案见[单模型执行计划](../plans/2026-10-27__plan__d08w0256-single-run.md)及[容量设计](../specs/2026-10-27__design__network-capacity-a.md)。训练复现命令与manifest在执行计划中；结果目录的`evaluate_run.py`可复用已完成的闭环记录，`summarize_run.py`核对证据并生成曲线。原M1–M6模型及当前demo默认路径保持原设置。
