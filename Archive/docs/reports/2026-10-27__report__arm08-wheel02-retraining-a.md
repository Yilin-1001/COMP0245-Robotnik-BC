# A 模型 Arm 0.8 / Wheel 0.2 配对重训结果

2026-10-27。已完成用户确认的 7 个 A 模型：M1–M6 BC、M6 BC+J，全部从 seed0 初始化重新训练至 42,000 updates，使用 CUDA。

本轮权重调整没有解决 reaching。固定配对 checkpoint 的 32 场景验证中，六组 BC 均未 reaching；M6 BC+J 从旧模型的 4/32 降为 1/32。所有模型的持续保持均为 0/32。42,000 步终端模型在 8 场景 screening 中也均为 0/8 reaching、0/8 持续保持。

## 配方与比较方法

动作损失为 `0.8 × mean_arm_Huber + 0.2 × mean_wheel_Huber`。两个分组先分别取均值，均使用原标准化尺度和 Huber δ=1。原数据、221 fitting/32 validation 划分、网络结构、各自学习率、AdamW、batch1024、warmup500 与 cosine 预算保持不变。

主比较采用旧模型原先选定的相同步数，未在新 checkpoint 中重新搜索最优点。离线 RMSE 在专家轨迹的验证状态上计算；闭环指标由模型独立控制仿真得到，两者含义不同。单 seed、固定 checkpoint 的结果不能代表全部新 checkpoint 的最优性能。8 个 screening 场景来自 validation 的固定子集；本轮未访问 test。

BC+J 沿原校准规则重新计算 λ_J：0.06036798253479663，因此该组同时包含动作权重与派生校准值的变化。六组纯 BC 是动作权重的直接配对比较。

## 同步数结果

表中数值为「旧 0.5/0.5 → 新 0.8/0.2」。RMSE 为未加权的物理单位；reaching 使用 t>8秒后末端误差<10mm 的既定判据，随后继续3秒，最后1秒所有观测<10mm才算持续保持。

| 模型 | 配对 updates | Arm RMSE (N·m) | Wheel RMSE (rad/s) | Reaching /32 | 末端误差中位数 (mm) |
|---|---:|---:|---:|---:|---:|
| M1 BC | 2000 | 1.0453 → 0.9392 | 0.1100 → 0.1488 | 0 → 0 | 2195.8 → 40867.0 |
| M2 BC | 4000 | 0.7257 → 0.6621 | 0.0930 → 0.1124 | 0 → 0 | 1428.6 → 2292.2 |
| M3 BC | 30000 | 0.5532 → 0.5391 | 0.0528 → 0.0645 | 1 → 0 | 20583.8 → 44767.0 |
| M4 BC | 40000 | 0.5280 → 0.5804 | 0.0471 → 0.0441 | 0 → 0 | 1446.4 → 55884.6 |
| M5 BC | 6000 | 0.5422 → 0.5340 | 0.0493 → 0.0605 | 0 → 0 | 15972.0 → 14378.5 |
| M6 BC | 6000 | 0.5445 → 0.5290 | 0.0562 → 0.0692 | 0 → 0 | 1411.7 → 1408.9 |
| M6 BCJ | 30000 | 0.7247 → 0.6047 | 0.0443 → 0.0587 | 4 → 1 | 768.0 → 9028.0 |

纯 BC 中 5/6 组在配对点的 arm RMSE 下降，同时 wheel RMSE 上升；M4 方向相反。离线误差的改善未转化为 reaching 成功率改善。闭环末端误差包含模型控制失稳造成的大误差，没有裁掉失败场景。

## 曲线与模型位置

[验证集 Arm/Wheel RMSE 对比图](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/2026-10-27__plot__validation-rmse-old-vs-arm08.png) · [PDF](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/2026-10-27__plot__validation-rmse-old-vs-arm08.pdf)。每行一个模型，灰色为旧权重，蓝色为新权重，橙色虚线为配对 checkpoint。

[训练分组损失对比图](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/2026-10-27__plot__training-group-loss-old-vs-arm08.png) · [完整比较 CSV](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/2026-10-27__table__paired-comparison.csv)。训练图画未加权 group mean，不直接比较不同权重的总 loss。

[新模型路径与参数量索引](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/2026-10-27__index__retrained-models.csv)。索引同时列出配对点与42k终端的 `.pt`、`.npz` 路径；这些是固定评估点，未标为新权重的最优模型。

模型均保存在任务根目录的 `checkpoints/reweighted/2026-10-27__experiment__arm08-wheel02/clean/A/{bc,bcj}/<run_id>/`。每组保留21个checkpoint（每2000步）、NumPy导出、latest恢复状态和每500步的metrics.csv。旧模型与demo默认模型未覆盖。

## 验证与复现

- 源码及测试提交 `ab1c885`；任务清单提交 `6b8d847`。完整测试182项通过。
- 7/7终端checkpoint核对通过：42,000步、CUDA、0.8/0.2、对应run identity、原归一化内容哈希及模型结构。每组21份checkpoint和导出均存在。
- 旧配置、旧选中checkpoint与旧selection文件SHA256均与冻结清单一致；数值代码digest及BC+J校准文件SHA256核对通过。
- 14次闭环评估，共224次配对validation rollout及56次终端screening rollout；全部场景有记录，无执行异常。任务成功与执行异常分别统计。
- [评估索引](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/results.csv)、[训练进度与GPU环境](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/training_progress.json)、[模型校验记录](../../final/robotnik_arm_task/results/reweighted/2026-10-27__experiment__arm08-wheel02/verification.json)。逐场景JSON/CSV与汇总位于同一结果目录。

冻结配方见[协议](../../final/robotnik_arm_task/experiments/reweight/2026-10-27__protocol__arm08-wheel02-a.yaml)及[执行计划](../plans/2026-10-27__plan__arm08-wheel02-retraining-a.md)。从 `final/robotnik_arm_task/` 依次运行本轮结果目录内的 `train_queue.py`、`evaluate_pairs.py`、`summarize_pairs.py` 可复用已完成训练和评估并重新生成比较图表；训练使用GPU，评估使用16个CPU进程。
