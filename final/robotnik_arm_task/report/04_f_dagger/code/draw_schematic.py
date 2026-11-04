from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'figures'  # report/04_f_dagger/figures (originally outputs/2026-11-04/dagger_report_figure)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
C={'ink':'#152C42','muted':'#536779','line':'#D6E0E7','teacher':'#18806A','student':'#2868B2','noise':'#BF6A16','purple':'#7351A6'}
fig,ax=plt.subplots(figsize=(18,14.5))
fig.patch.set_facecolor('white'); ax.set(xlim=(0,180),ylim=(145,0)); ax.axis('off'); fig.subplots_adjust(0,0,1,1)

def txt(x,y,s,size=10,color=None,weight='normal',ha='left',va='center',**kw):
 return ax.text(x,y,s,fontsize=size,color=color or C['ink'],fontweight=weight,ha=ha,va=va,linespacing=1.38,**kw)
def box(x,y,w,h,fc='white',ec=None,r=1,lw=.8):
 p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',facecolor=fc,edgecolor=ec or C['line'],linewidth=lw);ax.add_patch(p);return p
def arrow(x1,y1,x2,y2,col=None,dash=False,lw=1.5,rad=0):
 p=FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,color=col or C['ink'],linestyle=(0,(4,3)) if dash else '-',linewidth=lw,connectionstyle=f'arc3,rad={rad}');ax.add_patch(p)
def state(x,y,label,color):
 ax.add_patch(Circle((x,y),2.2,facecolor='white',edgecolor=color,lw=1.2));txt(x,y,label,10,color,ha='center')
def robot(x,y,active):
 # Conceptual side view: base, fixed hold posture or extended reaching arm.
 ax.add_patch(Rectangle((x-5,y+3),10,3,fc='#DCE6EE',ec=C['ink'],lw=1))
 for dx in [-3,3]:ax.add_patch(Circle((x+dx,y+6.1),1,fc=C['ink']))
 pts=np.array([[x,y+3],[x-1,y-2],[x+3,y-4]]) if not active else np.array([[x,y+3],[x+2,y-2],[x+9,y-3]])
 ax.plot(pts[:,0],pts[:,1],color=C['ink'],lw=2.3,solid_capstyle='round')
 for p in pts:ax.add_patch(Circle(p,.55,fc='white',ec=C['ink'],lw=1))
 ax.plot([x+11],[y-3],marker='*',color=C['teacher'],ms=11)
 if active:arrow(x+7,y-5,x+10.3,y-3.7,C['teacher'],dash=True,lw=1)
 else:arrow(x-5,y+9,x+5,y+9,C['teacher'],lw=1.2)

txt(5,5,'Expert labels across the two gate states',21,weight='bold')
txt(5,10,'Implementation schematic | R4: OU + Gate   /   R6: DAgger + Gate   /   R8: f-DAgger + Gate (OU, then DAgger)',11,C['muted'])
box(5,15,170,12,fc='#F1F5F8')
txt(8,18.6,r'$g(o_t)=1$ iff $|d_t|<0.12$ m and $|h_t|<0.075$ rad; otherwise $g(o_t)=0$.',12,weight='bold')
txt(8,23.5,r'$d_t=$ planar target distance $-$ height-dependent standoff; $h_t=$ target heading error.',10,C['muted'])
txt(109,18.8,'Gate is recomputed at every step: 0 ↔ 1.',10,weight='bold')
txt(109,23.5,'One 22-input MLP; gate is an input, not a hard output switch.',9.5,C['muted'])

# Legend
arrow(6,30,14,30,C['student']);txt(15,30,'Executed action',9.5)
arrow(48,30,56,30,C['teacher'],dash=True);txt(57,30,'Teacher supervision (saved label)',9.5)
txt(110,30,'All collectors label each pre-action state at 1000 Hz.',9.5,C['muted'])

cols=[(33,46,'A  OU + Gate','R4: teacher-driven noisy collection',C['noise']),
      (81,46,'B  DAgger + Gate','R6: student-driven collection',C['student']),
      (129,46,'C  f-DAgger + Gate','R8: OU-initialised student collection',C['purple'])]
for x,w,title,sub,color in cols:
 box(x,34,w,9,fc=color,ec=color)
 txt(x+2,37.2,title,13,'white',weight='bold');txt(x+2,40.7,sub,9,'white')

for active,y in [(False,45),(True,80)]:
 box(5,y,26,32,fc='#F1F5F8',ec='#F1F5F8')
 txt(7,y+3.4,('01  GATE = 0' if not active else '02  GATE = 1'),12,weight='bold')
 txt(7,y+7.7,('Navigate + arm posture' if not active else 'Arm reach + hold'),10,weight='bold')
 robot(16,y+16,active)
 txt(7,y+28.5,('Teacher arm: fixed-pose PD' if not active else 'Teacher arm: Cartesian reach'),8.5,C['muted'])
 for i,(x,w,_,_,color) in enumerate(cols):
  dart=i==0
  box(x,y,w,32,fc='#FFFCF7' if dart else '#F8FBFF')
  txt(x+2,y+3,('State reached by teacher + OU noise' if dart else 'State reached by the current student'),9.2,C['muted'])
  state(x+4.5,y+9,r'$o_t$',color)
  box(x+12,y+6,23,6,fc='#FFF0DB' if dart else '#E6F0FC',ec=color)
  txt(x+23.5,y+9,('Teacher + OU' if dart else r'Student $\pi_k$'),11,color,weight='bold',ha='center')
  state(x+41,y+9,r'$o_{t+1}$',color)
  arrow(x+6.8,y+9,x+11.8,y+9,color)
  arrow(x+35,y+9,x+38.7,y+9,color)
  txt(x+23,y+14.2,(r'Execute $\mathrm{clip}(\mathrm{clip}(a_E)+n_t)$' if dart else r'Execute $\mathrm{clip}(\pi_k(o_t))$; $\beta=0$'),10,color,ha='center')
  # Same current observation supplies the expert label; dashed line never enters execution.
  ax.plot([x+4.5,x+4.5,x+10],[y+11.3,y+20,y+20],color=C['teacher'],ls=(0,(4,3)),lw=1.25)
  arrow(x+9,y+20,x+11,y+20,C['teacher'],dash=True)
  box(x+11,y+17.4,20,5.2,fc='#E8F4EF',ec=C['teacher'])
  txt(x+21,y+20,r'Teacher $a_E=\pi_E(o_t)$',9.7,C['teacher'],ha='center')
  arrow(x+31,y+20,x+35,y+20,C['teacher'],dash=True)
  box(x+35,y+17.4,9,5.2,fc='white',ec=C['teacher'])
  txt(x+39.5,y+20,r'Label $y_t$',9.4,C['teacher'],ha='center')
  if not active:
   guidance=('Label: wheel alignment / standoff correction;\narm torque restores the fixed hold posture.' if not dart else 'Noise: arm + wheels before first gate activation.\nClean labels teach recovery from noisy states.')
  else:
   guidance=('Label: arm torque corrects reaching / hold error;\nwheel labels still correct residual base errors.' if not dart else 'Wheel noise has latched OFF; arm OU continues.*\nClean labels teach recovery during reaching.')
  txt(x+2,y+27.6,guidance,9.25)

# Round-level training, visually separate from the instantaneous action path.
for i,(x,w,_,_,color) in enumerate(cols):
 box(x,115,w,22,fc='white',ec=color,lw=1.1)
 txt(x+2,118,'OFFLINE TRAINING',10,color,weight='bold')
 if i==0:
  txt(x+2,122.2,r'$D_{\mathrm{Clean}}\ \cup\ D_{\mathrm{OU}}$',13,weight='bold')
  arrow(x+2,126.5,x+10,126.5,color)
  txt(x+11,126.5,'Train student on clean teacher labels',9.4)
  txt(x+2,132,'OU data: sustained-hold passes only.\nPerturbations are configured OU noise.',9.1,C['muted'])
 else:
  dat=(r'$D_{\mathrm{Clean}}\ \cup\ \bigcup_{i=1}^{k}D_{\mathrm{DAgger},i}$' if i==1 else r'$D_{\mathrm{Clean}}\ \cup\ D_{\mathrm{OU}}\ \cup\ \bigcup_{i=1}^{k}D_{\mathrm{DAgger},i}$')
  txt(x+2,122.2,dat,12,weight='bold')
  arrow(x+2,126.5,x+10,126.5,color)
  txt(x+11,126.5,r'Train $\pi_{k+1}$ from scratch; collect again',9.3)
  txt(x+2,132,('Failed student episodes are retained.\nTeacher labels change the NEXT round\'s policy.' if i==1 else 'DAgger rollouts add NO OU noise.\nBoth data sources supervise the next policy.'),9.1,C['muted'])

txt(5,118,'WHEN THE LABEL\nTAKES EFFECT',10,weight='bold')
txt(5,126.5,'Saved target → loss\n→ updated policy',9.7,C['teacher'])
txt(5,133.5,'No teacher takeover\nduring DAgger rollout.',9.1,C['muted'])
txt(5,140.2,'* OU noise: wheels latch off at the first g=1 (even if g later returns to 0); all noise stops 1.5 s after first task reach.',9,C['muted'])
txt(5,143.2,'Gate=1 can include base micro-adjustments. DAgger excludes numerical-failure / empty episodes. Schematic: no measured trajectories or performance claims.',8.5,C['muted'])

stem=OUT/'gate_dart_dagger_comparison'
fig.savefig(stem.with_suffix('.pdf'))
fig.savefig(stem.with_suffix('.svg'))
fig.savefig(stem.with_suffix('.png'),dpi=300)
print(stem)
