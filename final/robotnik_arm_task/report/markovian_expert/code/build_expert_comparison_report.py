import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-expert-boundaries'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK=Path('/media/yilin/data/ML_Final/COMP_0245_2026_Final/final/robotnik_arm_task')
OUT=TASK/'results/diagnostics/2026-11-03__expert-a-b-boundaries'
supplement=json.loads((OUT/'supplement_summary.json').read_text())
all_new=json.loads((OUT/'supplement_rollouts.json').read_text())
assert len(all_new)==112
assert not any(r['failure'] for r in all_new)
combined=[]
for teacher,suffix in [('A',''),('B','-B')]:
    source=TASK/f'results/diagnostics/2026-10-29__expert-workspace{suffix}/summary.json'
    data=json.loads(source.read_text())
    for sweep,items in data['sweeps'].items():
        for r in items:
            combined.append(dict(teacher=teacher,extended_B_timeout=False,sweep=sweep,value=r['target_z' if sweep=='height' else 'distance'],
                                 n=r['n'],hold=r['hold'],reach=r['reach'],median_final_err_mm=r['median_final_err_mm'],median_reach_t_s=r['median_reach_t_s'],source_date='2026-10-29'))
combined.extend(dict(r,source_date='2026-11-03') for r in supplement)
(OUT/'combined_summary.json').write_text(json.dumps(combined,indent=2))
with (OUT/'combined_summary.csv').open('w',newline='') as f:
    keys=['teacher','extended_B_timeout','sweep','value','n','hold','reach','median_final_err_mm','median_reach_t_s','source_date']
    w=csv.DictWriter(f,keys,extrasaction='ignore');w.writeheader();w.writerows(combined)

def table(sweep,values):
    lines=['| '+('目标高度（m）' if sweep=='height' else '起点到目标水平距离（m）')+' | Markovian A | Stateful B |','|---|---:|---:|']
    for v in values:
        entries=[next(r for r in combined if r['teacher']==t and not r['extended_B_timeout'] and r['sweep']==sweep and r['value']==v) for t in ('A','B')]
        lines.append(f"| {v:g} | "+' | '.join(f"{r['hold']}/{r['n']}" for r in entries)+' |')
    return '\n'.join(lines)

ext=[r for r in supplement if r['extended_B_timeout']]
ext_table='\n'.join(['| 距离 | 原始 B（25 s 导航超时） | 仅取消 B 导航超时 |','|---|---:|---:|']+[f"| {r['value']:g} m | 0/8 | {r['hold']}/{r['n']} |" for r in ext])

report=f'''# Markovian expert A 与原始 Stateful expert B：能力边界对比

日期：2026-11-03。

## 结论

两位专家的主要取舍是：A 的底盘具有持续纠偏和远距离导航能力；B 的机械臂保留积分参考，对持续力矩偏差的容忍度更好，但 25 秒导航超时和完成锁存限制了远距离、受扰动后的恢复。

这里给出有限测试中的经验工作范围和失败区间，不是任意姿态下的理论最大工作空间。所有几何测试使用配置中的固定初始机械臂姿态、平地、标称质量和摩擦；底盘起点、朝向和目标方向随机变化。比较的是两个实际实现，除记忆机制外，它们的目标几何、运动学信息和模型修正也不同，不能把所有差异归因于“是否 Markovian”。

## 证据与协议

- 历史配对几何测试：A、B 各 232 次，每个网格点各 8 个匹配场景。原始文件为 `results/diagnostics/2026-10-29__expert-workspace{{,-B}}/rollouts.csv`。
- 本次补充：112 次仿真，96 次配对边界测试和 16 次 B 导航超时干预；没有仿真异常。原始结果保存在本目录 `supplement_rollouts.json`。
- A 固定使用数据采集版本 `lead_time=0.50 s`。B 保留课程控制器、原点到目标方向的停车点、25 秒导航超时以及完成锁存。
- 几何测试判据：真实 EE 误差小于 10 mm 且时间超过 8 s；随后继续运行 3 s，最后完整 1 s 始终小于 10 mm。高度和近目标测试上限 40 s，远距离测试上限 80 s。
- 远距离 80 s 是外部评估时限；原始 B 的内部导航超时仍为 25 s。取消 demo 的 40 s 时限不会取消 B 内部的超时。
- 本次新网格沿用 `evaluation.expert_workspace.scene()` 的开发场景生成器，按网格值和样本索引确定随机数，两位专家使用同一场景。没有使用测试场景训练或调整控制参数。

## 1. 距离范围

近距离旧测试中，0.05、0.1、0.2、0.3、0.4、0.5 m 各点 A、B 均为 8/8。这表示很近的目标不一定困难：移动底盘可以重新建立伸手所需的停车距离；0.05 m 也不是测出的最小可行距离。

远距离比较：

{table('far',[6.,6.5,7.,7.5,8.,10.])}

A 在这些距离网格上没有观察到失败。另有现成的统一 T1 测试，A 在 3–6、6–10、10–15 m 三段各 24/24；对应外部时限为 80、80、120 s，不能与固定 40 s 任务时限混为一谈。测试只证明覆盖到 10–15 m 这一段，不能声称最大可行距离恰好为 15 m。

在原 40 s 任务上限内，还要预留伸手及 3 s 后续检查，A 的经验远距离预算约为 9 m。历史 10 m 测试中首个任务 reach 的中位时间约 39.645 s，所以需要超过 40 s 才能完成后续 hold 检查。这是时间预算限制，不是机械臂的水平伸展长度。

### 原始 B 的远距离失败是否由内部超时造成？

在与历史 8 m、10 m 完全相同的开发场景上，本次只取消 B 的导航超时，保留其原始阶段机、完成锁存、机械臂积分器和动作限幅。结果如下：

{ext_table}

这项干预把导航超时与其他实现因素分开了。取消超时后的版本属于修改后的 B，不能将它的结果记作原始 B 的能力。

原始 B 的工作距离取决于起始朝向、目标高度和转向耗时，不能用一个固定米数概括全部场景。应报告网格成功数及失败区间。

## 2. 高度范围

目标距初始底盘 1.5 m、每点 8 个匹配场景。0.1–1.30 m 的历史高度网格各点，两位专家均为 8/8。补测把高处边界进一步细化：

{table('height',[0.,.05,.1,1.3,1.31,1.32,1.33,1.35,1.4])}

因此，当前证据支持 A 在已测 0.1–1.33 m 高度网格上全部通过；B 在已测 0.1–1.32 m 高度网格上全部通过，1.33 m 开始出现样本相关失败。1.35 m 两者都失败。没有证明这些离散点之间、所有起点和姿态下都成功。

低处 B 在 z=0.05 m 仍有 4/8 成功，z=0 有 2/8 成功，而 A 在这两点均为 0/8。B 的成功直接说明不能把 A 的低目标失败解释为机器人绝对不可达。

两者都把目标限制在距肩部 0.85 m 内。A 模型给出的初始肩高约 0.495 m，所以肩高加这一限制约为 1.345 m；实际停车距离、关节限位、几何投影和控制精度会进一步影响结果。0.85 m 是控制器显式的目标投影半径，不是已证明的机器人全关节物理可达边界。两个共享约束的控制器都失败，也不足以单独证明目标物理不可达。

## 3. 扰动范围：沿用历史配对报告

来源：`docs/reports/2026-10-26__report__expert-operating-limits.md`，A 使用 lead_time=0.50 s。每种条件 16 个相同开发场景，同一条件下 A、B 使用相同噪声序列。

下表为满足至少 15/16 条件的最大已测试 sigma。sigma 是每个通道的标准差，不是噪声幅度上界；恒定偏差是在每条轨迹开始时抽取一个高斯偏差并保持，不是每个关节都施加相同的正向力矩。

“完成”指在噪声期间到达，并在停止噪声后恢复；“持续扰动下保持”指记录到的 noisy hold 样本中至少 90% 小于 10 mm。这与本报告几何测试的 sustained-hold 判据不同，不能合并成功率。

| 扰动 | A 完成 | B 完成 | A 持续扰动下保持 | B 持续扰动下保持 |
|---|---:|---:|---:|---:|
| 机械臂恒定力矩偏差，N·m | 1 | 4 | 0.5 | 2 |
| 机械臂 OU，相关时间 0.2 s，N·m | ≥8 | ≥8 | 0.5 | 0.5 |
| 轮速 OU，相关时间 0.2 s，不关闭噪声，rad/s | 1 | 0.3 | 0.1 | 0.1 |
| 轮速 OU，相关时间 0.2 s，按各自 gate 关闭噪声，rad/s | 1 | 0.3 | 1 | 0.3 |

≥8 表示测试网格只到 8，该点仍满足条件，不表示无限耐受。机械臂与轮噪声历史上分别测试，没有证明同时施加时具有相同边界。A、B 的 gated 条件还分别采用瞬时 reach gate 和锁存 done，不能把差异全部归因于控制增益。

本次未重跑历史扰动网格；该报告所引用的 `results/expert_noise_robustness.csv` 在当前工作区未找到，因此这些扰动数值是历史报告证据，本次不能从原始 CSV 独立复核。新几何边界测试的原始逐场景数据完整保存在本目录。

## 4. 为什么能力边界不同

- A 每步重算底盘距离和朝向误差，没有导航超时，被推离后会重新调整；B 保存阶段和 done，完成或超时后不再纠正底盘位置。
- B 积分关节参考，可以积累参考偏移来补偿一定的恒定力矩误差；A 每步用当前状态重建参考，没有这种累积补偿，因此可能稳定在有偏差的位置。
- A 使用固定 roll/pitch 近似并修正末端模型质量，B 使用真实模拟器姿态和原始动力学模型。比较实际能力是合理的，但不是只改变一个记忆开关的因果消融。
- A 的质量变化、侧向推力测试显示了额外弱点；没有同协议的 B 数据，本报告不据此给出 B 的质量或推力上限。

## 5. 报告用表述

> 标称平地、固定初始机械臂姿态下，Markovian expert 的优势在于持续底盘反馈和无内部导航超时，已在 10–15 m 区间完成测试；原始 Stateful expert 的远距离能力受 25 s 导航超时限制。两者在大部分正常高度具有相近的可达能力，高度边界约在 1.33–1.35 m 附近，Stateful expert 在低目标和恒定力矩偏差下更有优势。上述范围是给定控制器、协议和有限场景集的经验边界，不是全机器人工作空间的理论极限。

![配对高度和距离测试](expert_operating_boundaries.png)

## 数据与复现

- `compare_expert_boundaries.py`：本次补测脚本，使用当前两个专家；可在项目 Python 环境中运行。
- `run_manifest.json`：本次条件与控制器/评估代码 SHA256。
- `supplement_rollouts.json` / `.jsonl`：112 次新增仿真。
- `supplement_summary.json`：新增网格汇总。
- `combined_summary.csv` / `.json`：历史几何网格加本次补测，保留日期和 B 超时干预标记。
- 所有原始专家代码和现有模型均未修改。历史已存在的 `results/results.csv` 改动未触碰。
'''
(OUT/'comparison.md').write_text(report)

fig,axes=plt.subplots(1,2,figsize=(11,4.1))
for ax,sweep,xmin,xmax in [(axes[0],'height',0,1.4),(axes[1],'far',5.8,10.2)]:
    for t,color in [('A','tab:blue'),('B','tab:orange')]:
        rr=sorted([r for r in combined if r['sweep']==sweep and r['teacher']==t and not r['extended_B_timeout']],key=lambda r:r['value'])
        ax.plot([r['value'] for r in rr],[100*r['hold']/r['n'] for r in rr],marker='o',markersize=4,color=color,label=f'{t}: '+('Markovian' if t=='A' else 'original Stateful'))
    ax.set_xlim(xmin,xmax);ax.set_ylim(-5,105);ax.grid(alpha=.25);ax.set_ylabel('Sustained-hold success (%)')
axes[0].set(xlabel='Target world height (m)',title='Height sweep: 1.5 m initial distance; 40 s cap')
axes[1].set(xlabel='Initial planar distance to target (m)',title='Distance sweep: 80 s external cap')
axes[1].scatter([r['value'] for r in ext],[100*r['hold']/r['n'] for r in ext],s=95,marker='x',color='tab:green',label='B: navigation timeout disabled')
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
fig.suptitle('Paired development scenes: 8 per grid point; lines connect tested points only',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'expert_operating_boundaries.png',dpi=180)
print('Report:',OUT/'comparison.md')
print('Validated 112 new rollouts; no simulator exceptions.')
