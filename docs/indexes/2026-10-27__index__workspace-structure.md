# 工作区文件导航与命名规范

整理日期：2026-10-27。时间显示采用 Europe/London。

本索引描述实际文件位置。旧路径为相对符号链接，内容只保存在规范路径中一份。

## 命名规则

`YYYY-MM-DD__类型__主题.ext`；文件夹优先使用 `日期/用途/主题/`。

文档日期取原文件最后一次 Git 作者日期；实验产物取实验记录或 provenance 日期。
清单中的 mtime 是文件修改时间，不当作文件创建时间。当前整理操作的日期不覆盖原实验日期。

源码采用功能名；原始数据、run_id、step_<k>.pt、best.pt、哈希协议和 results.csv 保持原有接口。
这些文件已有版本和关联语义，使用时间索引查找，不逐个改写其名称或内容。

## 主要结构

```text
ML_Final/
├── .venv/                            Python 环境
├── COMP_0245_2026_Final/              Git 项目（名称保持兼容）
│   ├── docs/{plans,specs,reports,research,references,indexes}/
│   ├── final/robotnik_arm_task/
│   │   ├── core/, training/, evaluation/, data_collection/   源码
│   │   ├── experiments/              配置、冻结协议、划分
│   │   ├── data/                     原始数据与特征缓存
│   │   ├── checkpoints/              模型与训练日志（保留 run_id）
│   │   └── results/
│   │       ├── figures/2026-10-24/expert-comparison/
│   │       ├── figures/2026-10-27/{baseline-evaluation,training-diagnostics,train-test-learning-curves}/
│   │       ├── tables/{2026-10-24,2026-10-25,2026-10-26}/
│   │       ├── live_compare/2026-10-27/
│   │       └── clean/, reference/, diagnostics/, results.csv   可追溯评估记录
│   └── tests/                        测试
└── outputs/
    ├── 2026-10-25/experiments/markovian-lead-time/
    ├── 2026-10-26/validation/ou-noise/
    ├── 2026-10-26/data-preparation/clean-subsets/
    └── 2026-10-27/organization/       全量清单、改名映射、模型时间索引
```

## 文档索引

| 日期 | 类型 | 文档 |
|---|---|---|
| 2026-10-24 | roadmap | [project-milestones](../plans/2026-10-24__roadmap__project-milestones.md) |
| 2026-10-26 | implementation-plan | [mlp-baseline-and-jacobian](../plans/2026-10-26__implementation-plan__mlp-baseline-and-jacobian.md) |
| 2026-10-26 | plan | [ablation-strategy](../plans/2026-10-26__plan__ablation-strategy.md) |
| 2026-10-26 | plan | [mlp-baseline-training](../plans/2026-10-26__plan__mlp-baseline-training.md) |
| 2026-10-26 | plan | [project-overview](../plans/2026-10-26__plan__project-overview.md) |
| 2026-10-24 | assignment | [comp0245-final-project](../references/2026-10-24__assignment__comp0245-final-project.pdf) |
| 2026-10-26 | reference | [research-bibliography](../references/2026-10-26__reference__research-bibliography.md) |
| 2026-10-24 | report | [markovian-expert-model-accuracy](../reports/2026-10-24__report__markovian-expert-model-accuracy.md) |
| 2026-10-26 | report | [clean-expert-data-collection](../reports/2026-10-26__report__clean-expert-data-collection.md) |
| 2026-10-26 | report | [expert-operating-limits](../reports/2026-10-26__report__expert-operating-limits.md) |
| 2026-10-26 | report | [paired-clean-dataset-subsets](../reports/2026-10-26__report__paired-clean-dataset-subsets.md) |
| 2026-10-26 | research | [mlp-training-recipe](../research/2026-10-26__research__mlp-training-recipe.md) |
| 2026-10-26 | design | [mlp-training-structure](../specs/2026-10-26__design__mlp-training-structure.md) |
| 2026-10-26 | spec | [expert-datasets](../specs/2026-10-26__spec__expert-datasets.md) |

project-overview、ablation-strategy 保留早期方案及后续修订；文中标记为 archived 的表格不是当前训练默认值。
mlp-training-recipe、mlp-baseline-training、mlp-training-structure 描述当前训练流程的设计来源；最终结果见项目 README 和冻结协议。

## 实验与图表入口

- [训练集／测试集学习曲线](../../final/robotnik_arm_task/results/figures/2026-10-27/train-test-learning-curves/2026-10-27__figure__train-vs-test-loss-detail.png)
- [学习率与训练 loss](../../final/robotnik_arm_task/results/figures/2026-10-27/training-diagnostics/2026-10-27__figure__baseline-learning-rate-and-loss.png)
- [网络结构对比](../../final/robotnik_arm_task/results/figures/2026-10-27/training-diagnostics/2026-10-27__figure__baseline-architecture-learning-curves.png)
- [Huber／MSE 对比](../../final/robotnik_arm_task/results/figures/2026-10-27/training-diagnostics/2026-10-27__figure__baseline-huber-vs-mse.png)
- [OU 扰动验证报告](../../../outputs/2026-10-26/validation/ou-noise/2026-10-26__report__ou-noise-validation.md)
- [Markovian lead_time 实验报告](../../../outputs/2026-10-25/experiments/markovian-lead-time/2026-10-25__report__markovian-lead-time.md)
- [项目文件清单（排除环境与第三方）](../../../outputs/2026-10-27/organization/2026-10-27__inventory__project-files.csv)
- [整理后的全部文件位置](../../../outputs/2026-10-27/organization/2026-10-27__inventory__current-file-locations.csv)
- [按日期和类型汇总](../../../outputs/2026-10-27/organization/2026-10-27__timeline__project-files.csv)
- [文件类型统计](../../../outputs/2026-10-27/organization/2026-10-27__summary__file-types.csv)
- [所有原始文件清单](../../../outputs/2026-10-27/organization/2026-10-27__inventory__all-original-files.csv)
- [旧名到规范名对照表](../../../outputs/2026-10-27/organization/2026-10-27__mapping__renamed-paths.csv)
- [按训练时间排序的模型索引](../../../outputs/2026-10-27/organization/2026-10-27__index__trained-models.csv)
- [完整性检查结果](../../../outputs/2026-10-27/organization/2026-10-27__verification__organization.json)

## 清单与兼容性

- 盘点原始文件 56,309 个，涵盖环境、第三方依赖、源码、数据、模型与产物；未读取密钥内容。
- 已组织 99 个路径；完整映射保留整理前后 SHA-256。
- 143 个原有运行文件逐个校验 SHA-256；4,680 个数据／checkpoint 文件校验大小和修改时间。
- 索引覆盖 56 次模型训练，字段包含时间、教师、目标、结构、seed、LR、更新次数与推理文件。
- 原路径的兼容链接供旧命令、冻结证据和之前聊天中的链接使用；日常从本页规范路径进入。
- 不删除重复实验或历史记录，不改动虚拟环境、RoboEnv、模型权重和数据字节。
- results/clean 下按 run_id 和 checkpoint step 组织的评估表属于既有证据，继续保留稳定命名。
