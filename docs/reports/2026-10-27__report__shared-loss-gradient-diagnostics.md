# Arm与wheel两项损失是否妨碍reaching：当前模型梯度诊断

日期：2026-10-27。只做只读诊断，没有优化器更新、重新训练、改变损失或修改模型选择。

## 机制与检查对象

当前模型共享全部隐藏层，末端分别是7维torque head与2维wheel head。两项损失在共享层参数上的原始梯度为`0.8 g_arm + 0.2 g_wheel`。若两项梯度方向冲突或大小严重失衡，存在多任务干扰的可能；已经分开输出头，并不等于共享特征也已解耦。

在D0 fitting221与D1 mixed64的实际聚合数据上，用固定seed20261027按轨迹等权、轨迹内时间均匀抽8个batch，每个1024行。R0、R1选中5k、R1末尾20k使用完全相同的样本，标准化与训练一致。只检查共享trunk梯度，不包含独立head参数。

## 结果

| 共享层诊断：8个batch的均值 | R0 step84000 | R1选中step5000 | R1末尾step20000 |
|---|---:|---:|---:|
| arm与wheel梯度余弦 | +0.0928 | +0.0179 | −0.0245 |
| 加权wheel梯度范数 / 加权arm梯度范数 | 9.67% | 10.62% | 3.43% |
| arm/wheel余弦为负的batch数 | 0/8 | 2/8 | 5/8 |
| approach-arm与其他阶段arm的梯度余弦 | +0.0656 | +0.0384 | −0.00666 |
| 总原始梯度对approach-arm方向投影反向的batch数 | 0/8 | 0/8 | 0/8 |

加权范数比明确包含0.8/0.2：`||0.2 g_wheel|| / ||0.8 g_arm||`。比例不是由两个loss的标量比值推断。

另测approach阶段arm梯度`g_R`，并将总arm梯度拆为`p_R g_R + (1-p_R) g_other`；这8个batch的approach平均比例为26.23%。相对于`0.8 p_R ||g_R||²`，wheel项对approach方向投影的平均贡献，选中5k为+0.116%、末尾20k为−0.052%。因此这些样本上wheel梯度没有严重抵消reaching方向。

这些是有限checkpoint、有限batch的原始梯度一阶诊断，**不是实际AdamW步长分析，也不能代表整个训练历史或所有关键状态**。AdamW的动量、逐参数预条件和weight decay均未计入；没有因此证明完全不存在负迁移。

当前证据不支持将“wheel损失严重压制arm”定为主要原因，仍需要通过受控训练消融验证因果。此前将arm/wheel从0.5/0.5改为0.8/0.2，也没有带来reaching成功率改善。

## 推荐的验证与解决顺序（未启动）

1. **先做arm-only配对消融。** 从与R1相同R0、冻结归一化、D0+D1、batch流和预算开始，仅去掉wheel项，保留`0.8 L_arm`，尽量固定arm项原有尺度。与现有R1比较分阶段arm拟合与反馈导数。wheel head保持不训练，但共享层变化会影响它的预测，因此不能直接把该模型的全学生导航失败解释成arm学习失败。
2. **使用明确标注的混合控制诊断。** 两组都由专家先导航到reaching激活，再由学生控制7个arm torque、专家持续控制wheel；使用相同场景与原reach/hold谓词。该结果只能说明专家底盘条件下的arm能力，不能称为完整学生成功率。若arm-only仍失败，wheel损失就不能解释全部问题。
3. **若消融显示明显收益，再比较独立arm/wheel网络。** 两者不共享可训练隐藏层，即`tau=f_arm(x;theta_arm)`、`omega=f_wheel(x;theta_wheel)`，wheel损失无法更新arm参数。保留相同输入，控制或明确报告参数量变化，避免把额外容量当成解耦效果。当前仅拆最后输出头已经实现，不足以测试这一点。
4. **单独检查阶段与反馈问题。** arm0.8并未将80%样本分配给reaching。可分别比较阶段均衡采样、reaching加权与反馈导数目标，但不要同时改loss类型、阶段采样、网络与DAgger接管策略，避免无法解释收益来源。

分别调用两个backward、或使用两个optimizer而仍更新同一共享层，不等于参数解耦。

只有后续发现持续且有显著影响的梯度方向冲突时，才考虑[PCGrad](https://arxiv.org/abs/2001.06782)等投影方法；若主要问题是梯度尺度不平衡，可考虑[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)。二者不保证修复反馈控制或访问状态覆盖问题。

## 原始证据

- [完整逐batch与汇总JSON](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/arm_wheel_gradient_diagnostics.json)。
- [复现诊断脚本](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/2026-10-27__analysis__shared-loss-gradients.py)，脚本内容SHA与JSON绑定一致。
- [当前共享trunk与双输出头实现](../../final/robotnik_arm_task/training/common/models.py)。
- [已完成的0.8/0.2配对重训](2026-10-27__report__arm08-wheel02-retraining-a.md)、[R1闭环与反馈诊断](2026-10-27__report__dagger-a-r1.md)。
