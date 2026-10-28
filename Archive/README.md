# Archive

归档日期：2026-10-28。这里存放今后不再使用的旧程序，以及被训练方法消融实验（`checkpoints/ablation/`）取代的旧模型。

**归档后的位置与原位置保持相同的相对路径**：`Archive/<原路径>`。恢复时把文件原样移回即可，代码请用 `git mv`。

- 代码：随 git 跟踪，历史记录完整保留。
- 模型：`Archive/final/robotnik_arm_task/checkpoints/` 已加入 `.gitignore`。
- 测试：`pytest.ini` 设置了 `norecursedirs = Archive`，归档的测试不会被收集。

## 归档的程序

| 原路径（相对仓库根目录） | 原因 |
|---|---|
| `final/robotnik_arm_task/training/dataset.py`、`training/policies.py` | 早期占位文件，已由 `training/common/` 取代 |
| `final/robotnik_arm_task/training/train_act.py` | ACT 占位，已决定不做 |
| `final/robotnik_arm_task/training/train_mlp.py`；`tests/training/test_train_mlp_wrapper.py`（原在 `test_baseline_cli.py` 中） | 已废弃的兼容入口及其测试 |
| `final/robotnik_arm_task/experiments/capacity.py`、`experiments/capacity/`；`tests/training/test_capacity.py` | 已完成的 D8-W256 容量实验 |
| `final/robotnik_arm_task/experiments/reweight/` | 已完成的 arm 0.8 / wheel 0.2 配对重训 |
| `final/robotnik_arm_task/training/dagger/`、`data_collection/dagger.py`、`dagger_preflight.py`、`dagger_rollout.py`、`dagger_scenarios.py`、`evaluation/dagger_audit.py`、`evaluation/dagger_report.py`；`tests/dagger/` | DAgger A：R1 未改善 reaching，当前计划不再使用 |
| `final/robotnik_arm_task/experiments/protocols/2026-10-27__protocol__dagger-a-r1.json` | DAgger R1 冻结协议 |
| `final/robotnik_arm_task/experiments/demo_models/`（全部） | 指向已归档模型的演示清单 |

## 归档的模型（git 忽略）

| 原路径 `final/robotnik_arm_task/checkpoints/…` | 内容 | 大小 |
|---|---|---|
| `clean/A/`、`clean/B/` | 旧 clean 流程：A、B 的 BC 筛选与种子实验，M6/M1 BC+J | 797 MB |
| `reweighted/` | arm 0.8 / wheel 0.2 配对重训的 7 个 A 模型 | 105 MB |
| `capacity/` | D8-W256 | 77 MB |
| `dagger/` | DAgger R1 | 34 MB |
| `sanity/`、`calibration/` | 单轨迹过拟合检查、旧 λ_J 校准记录 | <2 MB |

## 对旧记录的影响

- `results/results.csv`、`experiments/protocols/p-*.yaml`、`docs/reports/` 和 `outputs/2026-10-27/organization/` 中记录的旧模型路径**没有改写**。要找到实际文件，在原路径前加上 `Archive/`（相对仓库根目录）。文件内容和 SHA-256 均未改变。
- `evaluation.live_compare` 的默认演示模型已归档；运行时会提示改用 `--protocol <模型清单>`。消融模型训练完成后，默认演示会切换到新模型。
- 专家 B 的代码（`data_collection/template_teacher.py` 等）、DAgger 原始数据（`data/Markovian/DAgger/`）、旧结果（`results/`）都未归档。

## 恢复示例

```bash
# 恢复 DAgger 代码（仓库根目录执行）
git mv Archive/final/robotnik_arm_task/training/dagger final/robotnik_arm_task/training/dagger
# 恢复旧模型
mv Archive/final/robotnik_arm_task/checkpoints/clean final/robotnik_arm_task/checkpoints/clean
```
