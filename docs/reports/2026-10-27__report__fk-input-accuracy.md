# 学生输入中的FK末端位置：定义与直接精度检查

日期：2026-10-27。只做训练后诊断，没有修改输入、模型、物理、场景或模型选择，没有访问新test。

## FK表示什么

FK是Forward Kinematics，正向运动学：根据URDF中的连杆几何、关节轴和当前关节角，计算末端坐标系原点的位置。当前实现为`core.features.DeployableFK.ee_pos()`，使用Pinocchio输出`robot_j2s7s300_end_effector`坐标系的世界坐标。

使用的量包括观测底盘xyz、yaw和7个arm关节角。底盘roll固定0、pitch固定0.0022rad；6个手指角固定0.1rad。它是一个确定的几何计算，输入中的末端误差为`Rz(-yaw) × (target_world − ee_FK_world)`。

固定pitch是平地校准近似；观测没有真实roll/pitch，因此底盘姿态变化会带来位置估计误差。URDF质量修正主要影响重力力矩，不等于改变FK的连杆几何计算。

## 此次直接测量

预先选取此前8个冻结恢复探针中的前4个fitting场景：`train-0001`、`train-0035`、`train-0067`、`train-0108`。分别运行A专家与当前R1选中step5000学生。专家使用lead_time0.50；学生沿用原21维输入及冻结归一化。控制1kHz，每10步即10ms记录一次位置比较。

参考位置直接读取PyBullet `getLinkState(...)[4]`，即同一个EE link frame原点的世界坐标，避免将质心位置与link原点混比；同时核对`RobotEnv.ee_pos()`的真实完整底盘姿态FK。所有仿真真值只进入诊断日志，没有传入策略。

| FK相对PyBullet位置的欧氏误差 | 专家轨迹4条 | 当前R1学生轨迹4条 |
|---|---:|---:|
| 采样观测数 | 4604 | 16000 |
| 中位数 | 0.507 mm | 0.607 mm |
| 95分位 | 1.539 mm | 1.694 mm |
| 采样最大值 | 3.918 mm | 5.435 mm |
| 采样误差超过10mm的比例 | 0 | 0 |
| 各轨迹最后1s合并后的误差中位数 | 0.643 mm | 0.734 mm |

这里最大值仅指10ms采样点，不是对全部1ms物理状态的上界。该检查只有4个nominal场景，不能推广成任意扰动、倾斜或实体机器人的精度保证。

使用真实底盘完整姿态的`RobotEnv.ee_pos()`与PyBullet位置，全部记录中的差异小于0.000129mm；表中毫米级误差主要来自部署FK的固定roll/pitch近似，而非URDF几何映射不一致。

4条专家轨迹均持续保持成功；4条学生轨迹均未reaching，学生最终真实目标误差分别约2100、1962、1575、1090mm。FK位置估计误差远小于这些任务误差，因此这组证据不支持将这些失败主要归因于FK。毫米级近似仍会消耗10mm到达阈值的部分余量，需要与精确保持问题区分。

## 与旧精度报告的区别

[旧专家模型精度报告](2026-10-24__report__markovian-expert-model-accuracy.md)中的0.2–3.2mm是专家最终到达目标的控制误差，不是直接测量的FK位置误差。当前表格直接比较FK预测位置与真实link位置，二者统计对象不同。

## 结果与复现

- [原始逐点诊断和汇总](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/fk_accuracy.json)。
- [冻结探针、模型及代码哈希](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/fk_accuracy_contract.json)。
- [复现脚本](../../final/robotnik_arm_task/results/dagger/2026-10-27__experiment__d08w0256-a-r1/2026-10-27__analysis__fk-input-accuracy.py)。脚本正文与已记录source SHA一致；如需重跑，应使用新的结果目录，以保留原冻结证据。
- [FK实现](../../final/robotnik_arm_task/core/features.py)。

本次没有变更生产代码；验证为8次真实PyBullet完整闭环回放、PyBullet link frame与完整姿态FK的独立几何核对。
