# D8-W256 单模型训练与闭环验证

2026-10-27。用户已授权先单独训练D8-W256并进行测试；30组容量矩阵暂不启动。

执行状态：84k训练及完整32 validation、221 fitting闭环评估均已完成，见[结果报告](../reports/2026-10-27__report__d08w0256-single-run.md)。

- A教师，Clean_253，原221 fitting / 32 validation；归一化仅用fitting。
- 隐藏层`[256,256,256,256,256,256,256,256]`，472,585参数，LayerNorm+SiLU，21输入及7+2输出头。
- 纯BC，标准化Huber δ1；arm/wheel=0.8/0.2。
- 单次学习率采用前轮提出的3e-4，seed0，从头初始化，84,000 updates。
- 沿原AdamW（β0.9/0.999，eps1e-8，weight decay1e-4），batch1024，warmup500+cosine。
- 训练前用独立模型做200次GPU预检；随后丢弃预检模型，按原seed初始化正式训练。
- 每2000步记录原离线验证指标，每7000步保存checkpoint、NumPy导出、恢复状态，并记录同口径训练/验证探针误差。
- 探针每条轨迹等距最多64行，包含首末行，先逐episode取均值再等权汇总；分别记录Huber、物理RMSE和navigation/approach/hold分阶段误差。探针不计算末尾1秒hold bias。
- 在42k/84k各跑8个固定screening场景，按原rank_rollout及较早步数决胜选择一个checkpoint；完整训练和验证独立并行执行，不用专家接管学生。
- 选中checkpoint跑完整32 validation与全部221 fitting；固定的32 fitting诊断子集按2距离×4高度×4方位格中最小ID预先确定，从完整fitting结果中提取，不重复计算独立样本。
- “测试”沿用本轮容量plan的训练/验证闭环检查，不访问封存的test场景。

使用[容量设计](../specs/2026-10-27__design__network-capacity-a.md)中的原物理成功判据与40秒仿真上限，最终报告预测误差与任务成功率，二者分别解释。

冻结配置：[单模型manifest](../../final/robotnik_arm_task/experiments/capacity/2026-10-27__experiment__d08w0256-arm08-wheel02.yaml)。使用薄入口`experiments.capacity`复用原BC engine；架构仅注册到该run的配置，原M1–M6模板与矩阵不变。manifest、入口、探针索引与数据哈希进入run identity。

从`final/robotnik_arm_task/`启动：

```bash
python -m experiments.capacity --manifest experiments/capacity/2026-10-27__experiment__d08w0256-arm08-wheel02.yaml
# 相同配置中断后添加 --resume；检查配置而不训练添加 --dry-run。
```

独立输出目录（相对任务根目录）：

- `checkpoints/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/clean/A/bc/<run_id>/`
- `results/capacity/2026-10-27__experiment__arm08-wheel02-d08w0256/`

运行契约、probe_indices、preflight、训练/验证曲线、逐场景结果和模型校验记录均归档到本轮结果目录。旧模型与demo路径保持原有设置。
