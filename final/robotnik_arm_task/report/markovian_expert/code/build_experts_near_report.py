import os
os.environ['MPLCONFIGDIR']='/tmp/mpl-experts-near'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK=Path('/media/yilin/data/ML_Final/COMP_0245_2026_Final/final/robotnik_arm_task')
OUT=TASK/'results/diagnostics/2026-11-03__expert-a-b-near'
rows=json.loads((OUT/'rollouts.json').read_text())
summary=json.loads((OUT/'summary.json').read_text())
assert len(rows)==112 and len(summary)==14
assert all(r['failure'] is None for r in rows)
assert sum(r['sustained_hold_pass'] for r in rows if r['teacher']=='A')==56
assert sum(r['sustained_hold_pass'] for r in rows if r['teacher']=='B')==55
for d in sorted({r['initial_distance_m'] for r in rows}):
    for i in range(8):
        pair=[r for r in rows if r['initial_distance_m']==d and r['sample']==i]
        assert len(pair)==2 and pair[0]['target']==pair[1]['target'] and pair[0]['start']==pair[1]['start'] and pair[0]['yaw']==pair[1]['yaw']
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,list(summary[0]));w.writeheader();w.writerows(summary)
table=['| 初始水平距离 | A 成功 | B 成功 | A 首次到达中位数 | B 首次到达中位数 | A 最终误差中位数 | B 最终误差中位数 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for d in sorted({r['distance_m'] for r in summary}):
    a,b=[next(r for r in summary if r['distance_m']==d and r['teacher']==t) for t in ('A','B')]
    table.append(f"| {d:g} m | {a['holds']}/8 | {b['holds']}/8 | {a['median_first_arrival_t']:.2f} s | {b['median_first_arrival_t']:.2f} s | {a['median_final_err_mm']:.3f} mm | {b['median_final_err_mm']:.3f} mm |")
failure=next(r for r in rows if not r['sustained_hold_pass'])
report='''# 两位专家的近距离对比

日期：2026-11-03。比较实际 Markovian A（lead_time=0.50 s）与原始 Stateful B，未修改控制器。新增 112 次配对仿真，全部完成且无仿真异常。

## 协议

- 距离指目标与初始底盘中心的**水平距离**，不是目标与末端的三维距离。0 m 表示目标在底盘中心的正上方，并不代表目标已经在末端，也不代表机械臂必须保持在原位伸向目标。
- 距离点：0、0.01、0.03、0.05、0.1、0.3、0.5 m；每点 8 个配对上下文、两位专家，合计 112 次。
- 目标高度为 0.2、0.5、0.8、1.2 m，每种高度两个上下文；起点 x/y 在 ±0.5 m，底盘朝向与目标相对方位随机。各距离复用同一组上下文，只有目标的水平距离变化；在 0 m 时目标方位本身无定义。
- 固定初始机械臂姿态、标称物理参数，40 s 上限，1 kHz。
- 任务成功沿用 rollout-v1：8 s 后 EE 误差小于 10 mm，再运行 3 s，最终完整 1 s 始终小于 10 mm。
- 速度比较使用 `first_arrival_t`：首次真实 EE 误差进入 10 mm 的时间，避免被“8 s 后才计任务 reach”截断。首次进入不等于已完成连续保持。
- 终端误差按同一提前终止规则统计，不是两位专家都固定运行到 40 s 后比较；它不能直接代表无限时长的稳态误差。

## 结果

'''+ '\n'.join(table)+'''

B 在 1 cm 处的首次到达中位数只包含到达的 7 条轨迹，该点的失败不能被计为快速完成。

本组 A 为 56/56，B 为 55/56。样本量有限，且同一上下文跨距离复用，不能据此声称两者整体成功率存在显著差异。既有独立场景网格在 0.05–0.5 m 各 8 条，A、B 均为 48/48。

![近距离速度、精度和底盘路径](near_comparison.png)

## 为什么 A 更快？

A 使用有符号停车距离误差：

$$
d=\|p_{target,xy}-p_{base,xy}\|-r(z)
$$

当目标距离小于停车距离时，d<0，底盘速度可以为负。因此它能先面对目标、再直接后退，建立伸手所需的空间。目标恰好在底盘正上方时，代码在水平距离不超过 1 mm 时令 heading error 为零；这使 0 m 与 1 cm 的转向行为不同，不能期待耗时随距离严格单调变化。

B 先根据“世界原点到目标”的方向确定固定停车点，按 align→drive→orient 行进；drive 线速度没有负值分支。为了去目标另一侧的停车点，它往往先转身、向前移动，然后再转回来面对目标，路径和转向耗时更大。

新测试中，在 0–0.3 m 所有点，A 的 8/8 条轨迹都出现超过 50 ms 的后退命令；B 所有 56 条轨迹都没有后退命令。在 0.05 m，底盘累计路径中位数 A 为 0.593 m，B 为 1.022 m。这个路径长度按每个 1 ms 观测的水平位移累加，包含实际滑移和微小振动。

## 为什么 B 更精确？

本停止规则下，B 的终端误差中位数大约 0.13–0.25 mm，A 大约 0.77–1.44 mm；两者正常完成的轨迹均满足 10 mm 标准。

B 保留积分关节参考，可以积累补偿一定的模型偏差；A 每步从当前状态重建参考，没有累计补偿。这与 B 的较小终端残差相一致。但两个实际实现还在动力学模型、底盘目标、运动学姿态信息等方面不同，本比较不是只切换积分器的消融实验，不能把全部精度差异只归因于记忆机制。

## B 的 1 cm 失败样本

样本 `near-comparison-0.01-3`：目标高度 1.2 m，初始水平距离 0.01 m。

- 原始 B 导航跑偏，底盘累计路径约 5.96 m。
- 首次 reach-active 出现在 25.000 s，与内部导航超时相符；当时底盘离目标已约 1.188 m，并非正确到达停车位置。
- 超时将 done 锁存，随后零轮速命令不再提供导航位置纠偏。底盘还可能因惯性或滑移继续移动。
- 40 s 内从未进入 10 mm；最终 EE 误差约 1714 mm，最小误差约 546 mm。
- 同一上下文中的 0、3 cm、5 cm 目标均成功，因此没有证据把这次失败解释为“最小可达距离是 1 cm”或“目标太近必然不可达”。

这个样本确认的是导航失败和超时误激活；当前记录未进一步隔离最初触发跑偏的数值/阶段边界因素，不能声称已经定位到某一条分支作为唯一原因。

![失败场景的导航距离](near_failure_navigation.png)

## 结论

正常高度、当前固定初始机械臂姿态下，两位专家都能处理极近的水平目标，甚至水平距离为零。A 的优势是能反向调整停车距离、较快进入伸手条件；B 在本停止规则下终端误差更小，但阶段式导航可能出现个别跑偏和超时。近目标的任务不一定要求原地伸手，专家往往先让底盘离目标更远。

这与低目标高度是不同问题：本测试不覆盖 z=0 或 z=0.05 m，也没有证明任意初始关节姿态、任意障碍环境下都有相同能力。

## 文件

- `rollouts.json` / `rollouts.jsonl`：112 次逐场景数据与 50 ms 轨迹。
- `summary.json` / `summary.csv`：按距离、专家汇总。
- `run_manifest.json`：配对规则、参数、源代码哈希。
- `compare_experts_near.py`：仿真脚本。
- 历史近距离数据：`../2026-10-29__expert-workspace{,-B}/rollouts.csv`。
'''
(OUT/'comparison.md').write_text(report)
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
panels=[('median_first_arrival_t','First arrival below 10 mm (s)'),('median_final_err_mm','Terminal EE error (mm)'),('median_base_path_m','Base path length (m)')]
for ax,(key,title) in zip(axes,panels):
    for t,color in [('A','tab:blue'),('B','tab:orange')]:
        rr=sorted([r for r in summary if r['teacher']==t],key=lambda r:r['distance_m'])
        ax.plot([r['distance_m'] for r in rr],[r[key] for r in rr],marker='o',label=f'{t}: '+('Markovian' if t=='A' else 'Stateful'),color=color)
    ax.set(xlabel='Initial planar target distance (m)',ylabel=title);ax.grid(alpha=.25)
axes[0].legend(fontsize=8)
fig.suptitle('8 matched contexts per distance; nominal physics; conditional medians for arrival times',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'near_comparison.png',dpi=180)
fig,ax=plt.subplots(figsize=(7.5,4.1))
for d,t,label in [(.01,'B','B: 1 cm (failed)'),(.01,'A','A: 1 cm'),(.03,'B','B: 3 cm (same context)')]:
    r=next(r for r in rows if r['teacher']==t and r['initial_distance_m']==d and r['sample']==3)
    ax.plot([x['time_s'] for x in r['trace']],[x['target_distance_m'] for x in r['trace']],label=label)
ax.axvline(25,color='grey',linestyle='--',label='B navigation timeout')
ax.set(xlabel='Simulation time (s)',ylabel='Planar base-to-target distance (m)',title='One near-target failure: navigation drift, then timeout')
ax.grid(alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'near_failure_navigation.png',dpi=180)
print('Verified matched scenes and 112 completed rollouts without simulation exceptions.')
print(OUT/'comparison.md')
