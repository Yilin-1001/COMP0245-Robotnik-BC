# A 已有网络：Arm 0.8 / Wheel 0.2 配对重训

日期：2026-10-27。用户已确认以下 A 范围；本轮不启动尚未批准的大网络容量矩阵或 DAgger。

训练协议：[冻结任务清单](../../final/robotnik_arm_task/experiments/reweight/2026-10-27__protocol__arm08-wheel02-a.yaml)。实现提交：`ab1c885`。182 项完整测试已通过。

| 网络 | 隐藏层 | 目标 | 原选定 LR | 新训练预算 | 固定配对评估步数 |
|---|---|---|---:|---:|---:|
| M1 | 64 → 64 | BC | 3e-4 | 42,000 | 2,000 |
| M2 | 128 → 128 | BC | 3e-4 | 42,000 | 4,000 |
| M3 | 256 → 256 | BC | 1e-4 | 42,000 | 30,000 |
| M4 | 128 → 128 → 128 | BC | 1e-3 | 42,000 | 40,000 |
| M5 | 256 → 256 → 256 | BC | 1e-3 | 42,000 | 6,000 |
| M6 | 256 → 128 → 64 | BC | 1e-3 | 42,000 | 6,000 |
| M6 | 256 → 128 → 64 | BC+J | 1e-3 | 42,000 | 30,000 |

全部从 seed0 初始化重新训练，不从旧模型继续优化。沿用原 AdamW、batch1024、weight decay1e-4、warmup500+cosine、标准化 Huber delta1、episode-uniform 采样；A 的 Clean_253 中 221 fitting/32 validation 不变，归一化内容哈希仍为 `1ba73a11d1bb7d581a034d86600e8f92d78a76760118e3c76130d3718ea31ef4`。

动作项为 `0.8 × mean_arm + 0.2 × mean_wheel`，未加权的两个 group loss 仍单独记录。新权重进入配置和 run identity；旧配置缺字段仍解释为0.5/0.5。BC+J 沿原10%初始化校准规则重新计算 λ_J，本轮为 `0.06036798253479663`；它是派生校准值变化，因此 BC+J 结果应与六组纯 BC 的单独权重对照区分。

输出相对 `final/robotnik_arm_task/`：

- `checkpoints/reweighted/2026-10-27__experiment__arm08-wheel02/clean/A/{bc,bcj}/<run_id>/`
- `results/reweighted/2026-10-27__experiment__arm08-wheel02/`

每组保留42k终端 checkpoint、可恢复的latest，以及每2000步的checkpoint与NumPy推理导出；每500步保存训练和验证离线指标。明确使用CUDA，GPU不可用时停止，不退回CPU。原模型、原协议和demo默认选择不覆盖。

完整训练后，在旧模型原先选中的相同步数上跑32个validation场景，比较相同训练阶段下的权重变化；42k终端另做8个screening场景检查。这里不是在新权重全部checkpoint中重新搜索最优点，不宣称这些固定配对点就是新模型的最优checkpoint。不访问test。

主要观察物理arm/wheel RMSE、reaching与持续保持、末端误差；不同权重下的加权总loss不能直接比较优劣。保持原reach/hold谓词与40秒仿真上限。

复现训练时，从任务根目录使用 `training.baseline.train` 或 `training.jacobian.train`，将协议每行的 `overrides` 转为 `--set key=value`。BC+J 校准文件已在协议中绑定哈希；需要从零复现时先用相同权重执行 `training.jacobian.train --calibrate`，再引用产生的记录。运行前核对协议列出的旧配置与checkpoint哈希，运行后核对新run_id。
