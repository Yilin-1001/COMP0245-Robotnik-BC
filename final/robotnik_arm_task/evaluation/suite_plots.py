"""Publication/export figures from saved outcome records only."""
from pathlib import Path
import math
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"ml-final-matplotlib"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

from core.model_names import report_label
COLORS = {'A':'#1b7837','B':'#7fbf7b','C':'#fddc7a','D':'#ef8a62','E':'#b2182b','F':'#555555'}


def plot_results(directory,rows,summary):
    directory=Path(directory);out=directory/'plots';out.mkdir(parents=True,exist_ok=True)
    if not rows:return
    names=sorted({r['row'] for r in rows},key=lambda s:(s=='expert',s))
    groups=sorted({(r['suite'],r['band'] if r['suite'] in ('T1','T3') else 'all') for r in rows})
    palette={n:plt.get_cmap('tab10')(i%10) for i,n in enumerate(names)};palette['expert']='black'
    def panels(n):
        cols=min(3,n);fig,axes=plt.subplots(math.ceil(n/cols),cols,figsize=(5*cols,3.8*math.ceil(n/cols)),squeeze=False)
        for ax in axes.flat[n:]:ax.set_visible(False)
        return fig,list(axes.flat)
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    def data(group,name=None):
        suite,band=group
        return [r for r in rows if r['suite']==suite and (band=='all' or r['band']==band) and (name is None or r['row']==name)]
    def stat(group,name,subset='all'):
        suite,band=group
        return next((r for r in summary if r['level']=='row' and r['policy']==name and r['suite']==suite and r['band']==band and r['subset']==subset),None)
    # 1. Main ablation comparisons: all scenes and expert-success subset.
    for subset in ['all','expert_success']:
        fig,axes=panels(len(groups))
        for ax,g in zip(axes,groups):
            students=[n for n in names if n!='expert']
            for i,n in enumerate(students):
                s=stat(g,n,subset)
                if s and s['success_rate'] is not None:
                    y=100*s['success_rate']
                    ax.bar(i,y,color=palette[n],alpha=.65)
                    if s['ci_low'] is not None:
                        lo=100*s['ci_low'];hi=100*s['ci_high']
                        ax.errorbar(i,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],color='black',capsize=3)
                    seedstats=[x for x in summary if x['level']=='model' and x['row']==n and x['suite']==g[0] and x['band']==g[1] and x['subset']==subset and x['success_rate'] is not None]
                    ax.scatter(i+np.linspace(-.12,.12,len(seedstats)),[100*x['success_rate'] for x in seedstats],s=15,color='black',zorder=3)
            expert=stat(g,'expert',subset)
            if expert and expert['success_rate'] is not None:ax.axhline(100*expert['success_rate'],color='black',ls='--',label='Markovian expert A')
            ax.set_xticks(range(len(students)),[report_label(n) for n in students],rotation=45,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Success (%)');ax.set_title('/'.join(g));ax.grid(axis='y',alpha=.2)
            if expert and expert['success_rate'] is not None:ax.legend(fontsize=7)
        fig.suptitle('Continuous <10 mm task success — '+subset,y=1.01)
        save(fig,'01_success_'+subset)
    # 2. Terminal precision grades are independent of the successful-hold flag.
    fig,axes=panels(len(groups))
    for ax,g in zip(axes,groups):
        bottom=np.zeros(len(names))
        for grade in 'ABCDEF':
            values=[]
            for n in names:
                rs=data(g,n);values.append(100*sum(r['grade']==grade for r in rs)/len(rs) if rs else 0)
            ax.bar([report_label(n) for n in names],values,bottom=bottom,color=COLORS[grade],label=grade);bottom+=values
        ax.set_ylim(0,100);ax.set_title('/'.join(g));ax.set_ylabel('Rollouts (%)');ax.tick_params(axis='x',rotation=45)
    axes[0].legend(ncol=6,fontsize=7);save(fig,'02_terminal_grades')
    # 3. Heatmaps for both denominators.
    for subset in ['all','expert_success']:
        values=np.full((len(names),len(groups)),np.nan)
        for i,n in enumerate(names):
            for j,g in enumerate(groups):
                s=stat(g,n,subset)
                if s and s['success_rate'] is not None:values[i,j]=100*s['success_rate']
        fig,ax=plt.subplots(figsize=(max(8,len(groups)*1.2),max(4,len(names)*.5)))
        im=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(len(groups)),['/'.join(g) for g in groups],rotation=45,ha='right');ax.set_yticks(range(len(names)),[report_label(n) for n in names])
        for i in range(len(names)):
            for j in range(len(groups)):ax.text(j,i,'N/A' if np.isnan(values[i,j]) else f'{values[i,j]:.0f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>65 else 'black')
        fig.colorbar(im,ax=ax,label='Success (%)');ax.set_title('Task success — '+subset);save(fig,'03_heatmap_'+subset)
    # 4. Error ECDF: log-error axis, F outcomes explicitly excluded/annotated.
    fig,axes=panels(len(groups))
    for ax,g in zip(axes,groups):
        for n in names:
            rs=data(g,n);errors=sorted(max(r['final1s_max_err_mm'],1e-4) for r in rs if r['grade']!='F' and r['final1s_max_err_mm'] is not None)
            if errors:ax.step(errors,np.arange(1,len(errors)+1)/len(errors),where='post',label=report_label(n),color=palette[n])
        ax.set_xscale('log');ax.axvline(10,color='grey',ls='--');ax.axvline(20,color='grey',ls=':')
        ax.set_xlabel('Final-second maximum error (mm; zeros at 1e-4)');ax.set_ylabel('Conditional ECDF, non-F only')
        ax.set_title('/'.join(g)+f"; F={sum(r['grade']=='F' for r in data(g))}");ax.set_ylim(0,1.02);ax.grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7,ncol=3)
    save(fig,'04_terminal_error_ecdf')
    # 5. Generalisation curves: keep each suite protocol separate, annotate actual band counts.
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,suite,key in [(axes[0],'T1','target_dist_m'),(axes[1],'T3','target_z_m')]:
        relevant=sorted([g for g in groups if g[0]==suite],key=lambda g:np.mean([r[key] for r in data(g)]))
        for n in names:
            xs=[];ys=[]
            for g in relevant:
                rs=data(g,n)
                if rs:xs.append(np.mean([r[key] for r in rs]));ys.append(100*np.mean([r['success'] for r in rs]))
            if xs:ax.plot(xs,ys,'o--',label=report_label(n),color=palette[n])
        ax.set_ylim(0,105);ax.set_ylabel('Success (%)');ax.set_xlabel('Distance (m)' if suite=='T1' else 'Height (m)')
        ax.set_title(suite+' — separate frozen bands'+(' (caps 80 / 80 / 120 s)' if suite=='T1' else ' (cap 40 s)'));ax.grid(alpha=.2)
        ax.text(.02,.02,'; '.join(f"{g[1]}: n={len({r['scenario_id'] for r in data(g)})}" for g in relevant),transform=ax.transAxes,fontsize=7)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7,ncol=3)
    save(fig,'05_distance_height_success')
    # 6. Aggregate the worst seed explicitly, rather than hiding overlapping seed-colored points.
    cmap=ListedColormap([COLORS[g] for g in 'ABCDEF']);norm=BoundaryNorm(np.arange(-.5,6.5),6)
    for suite in sorted({r['suite'] for r in rows}):
        for view in ['xy','distance-height']:
            fig,axes=panels(len(names))
            for ax,n in zip(axes,names):
                rs=[r for r in rows if r['suite']==suite and r['row']==n]
                byscene={}
                for r in rs:
                    sid=r['scenario_id']
                    if sid not in byscene or 'ABCDEF'.index(r['grade'])>'ABCDEF'.index(byscene[sid]['grade']):byscene[sid]=r
                points=list(byscene.values())
                x=[r['target_bx'] if view=='xy' else r['target_dist_m'] for r in points]
                y=[r['target_by'] if view=='xy' else r['target_z_m'] for r in points]
                ax.scatter(x,y,c=['ABCDEF'.index(r['grade']) for r in points],cmap=cmap,norm=norm,s=18,alpha=.85)
                if view=='xy':ax.plot(0,0,'k+',ms=8);ax.set_aspect('equal',adjustable='datalim')
                ax.set_title(report_label(n)+f"; {len({r['model_id'] for r in rs})} controller seed(s)")
                ax.set_xlabel('Initial forward target x (m)' if view=='xy' else 'Initial horizontal target distance (m)')
                ax.set_ylabel('Initial left target y (m)' if view=='xy' else 'World target height (m)')
            fig.suptitle(suite+' worst terminal grade across available seeds; A green → E red, F grey',y=1.01)
            save(fig,'06_targets_'+suite+'_'+view)
    # 7. Completion curves include failures in the denominator; successful times only in table.
    fig,axes=panels(len(groups))
    for ax,g in zip(axes,groups):
        for n in names:
            rs=data(g,n)
            if not rs:continue
            times=sorted(r['completion_time_s'] for r in rs if r['success'])
            cap=max(r.get('time_limit_s',r['duration_s']) for r in rs)
            ax.step([0]+times+[cap],[0]+list(100*np.arange(1,len(times)+1)/len(rs))+[100*len(times)/len(rs)],where='post',label=report_label(n),color=palette[n])
        ax.set_ylim(0,105);ax.set_title('/'.join(g));ax.set_xlabel('Simulation time (s), including hold');ax.set_ylabel('All-scene completion (%)');ax.grid(alpha=.2)
    if axes[0].get_legend_handles_labels()[0]:axes[0].legend(fontsize=7,ncol=3)
    save(fig,'07_completion_efficiency')
    # Additional long-hold traces show the worst terminal-error trajectory per row (not a model-selection step).
    longs=[r for r in rows if r['suite']=='T5' and r.get('history')]
    if longs:
        fig,axes=panels(len(names))
        for ax,n in zip(axes,names):
            candidates=[r for r in longs if r['row']==n]
            if candidates:
                r=max(candidates,key=lambda r:float('inf') if r.get('final1s_max_err_mm') is None else r['final1s_max_err_mm'])
                h=np.array(r['history']);ax.plot(h[:,0],h[:,1],color=palette[n]);ax.set_title(report_label(n)+' / '+r['scenario_id'])
                ax.set_yscale('symlog',linthresh=1);ax.axhline(10,color='grey',ls='--');ax.set_xlabel('Time (s)');ax.set_ylabel('EE error (mm)')
        fig.suptitle('T5 worst terminal outcome per row — full error history',y=1.01);save(fig,'08_long_hold_error_traces')
    # T4 paired shift directly compares the same relative task in different absolute map locations.
    bykey={(r['model_id'],r['scenario_id']):r for r in rows}
    pairs=[]
    for r in rows:
        parent=r.get('parent_id')
        if parent and (r['model_id'],parent) in bykey:pairs.append((bykey[r['model_id'],parent],r))
    if pairs:
        fig,axes=panels(len(names))
        for ax,n in zip(axes,names):
            valid=[(a,b) for a,b in pairs if b['row']==n and a['grade']!='F' and b['grade']!='F']
            ax.scatter([max(a['final1s_max_err_mm'],1e-4) for a,b in valid],[max(b['final1s_max_err_mm'],1e-4) for a,b in valid],s=15,color=palette[n])
            ax.plot([1e-4,1e5],[1e-4,1e5],color='grey',ls='--');ax.set_xscale('log');ax.set_yscale('log');ax.set_title(report_label(n))
            ax.set_xlabel('T0 parent terminal error (mm)');ax.set_ylabel('T4 translated terminal error (mm)')
        save(fig,'09_paired_translation')
