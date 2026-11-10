"""D5: standalone PDF/PNG figures from versioned summary tables only.

Uses pandas/matplotlib in the plotting environment; no CTR modules are imported.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures/deep_dive_v1'
RESULTS=ROOT/'results/deep_dive_final_v1'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#777777'
inputs={}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    path=ROOT/path;inputs[str(path.relative_to(ROOT))]=digest(path)
    return pd.read_csv(path,dtype={'value':str,'repeat_band':str})


def save(fig,name,footnote):
    fig.tight_layout(rect=(0,.06,1,.93))
    fig.text(.02,.01,footnote,fontsize=8,color='#555555')
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    begin=time.perf_counter()
    if OUT.exists() or RESULTS.exists():raise FileExistsError('Preserve published figures')
    OUT.mkdir(parents=True);RESULTS.mkdir(parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
                         'axes.titleweight':'bold','axes.labelcolor':'#333333'})
    # 1: historical information layers and frozen family comparison.
    fig,ax=plt.subplots(1,2,figsize=(12,4.7));fig.suptitle('Information gain exceeds observed model-complexity gain',fontsize=15)
    for split,color in [('val',BLUE),('test',ORANGE)]:
        vals=[read(f'results/{r}/evaluation/overall_metrics.csv').query('split==@split').auc.iloc[0] for r in ['lr_f1','lr_f1_f2','lr_f1_f2_f3']]
        ax[0].plot(['F1','F1+F2','F1+F2+F3'],vals,'o-',label=split,color=color)
    ax[0].set(ylabel='AUC',title='Existing LR information ablations');ax[0].legend();ax[0].grid(axis='y',alpha=.2)
    names=['LR','LightGBM','Wide & Deep','DeepFM','Mean + F4'];runs=['lr_f1_f2_f3','lightgbm_f1_f2_f3','wide_deep_f1_f2_f3','deepfm_f1_f2_f3']
    vals=[read(f'results/{r}/evaluation/overall_metrics.csv').query('split=="test"').auc.iloc[0] for r in runs]
    seq=read('results/sequence_test_v2/test_metrics.csv').query('condition=="mean_matched"');vals.append(seq.auc.mean())
    ax[1].barh(names,vals,color=[BLUE]*4+[ORANGE]);ax[1].errorbar(vals[-1],4,xerr=seq.auc.std(),color='black',capsize=3)
    ax[1].set(xlim=(.59,.66),xlabel='Test AUC (axis starts at 0.59)',title='Frozen families; sequence receives extra F4');ax[1].invert_yaxis()
    save(fig,'01_information_and_models','Tabular models: one frozen seed each. Sequence: residual-seed mean ± std on one shared backbone; not a pure architecture comparison.')
    # 2: reuse frozen bootstrap intervals verbatim.
    f=read('results/sequence_posthoc_v1/bootstrap_intervals.csv')
    fig,ax=plt.subplots(1,2,figsize=(12,5));fig.suptitle('Frozen user-cluster intervals: small sequence increments',fontsize=15)
    labels=['Mean matched − zero','Mean matched − mismatch','Attention matched − zero','Attention matched − mismatch','Attention − Mean (matched)']
    for a,metric in zip(ax,['auc','gauc']):
        sub=f[f.metric==metric].sort_values('id')
        for i,r in enumerate(sub.itertuples()):
            a.errorbar(r.effect*1e4,i,xerr=[[max(0,r.effect-r.ci_low)*1e4],[max(0,r.ci_high-r.effect)*1e4]],fmt='o',color=BLUE if i<2 else GRAY,capsize=3)
        a.set_yticks(range(5),labels);a.invert_yaxis();a.axvline(0,color='black',lw=.8);a.set(xlabel=f'Δ {metric.upper()} × 10⁴',title=metric.upper())
    save(fig,'02_frozen_sequence_intervals','Reused badecf4 results: 1,000 paired user-cluster draws; marginal 95% CIs. Blue: primary; gray: exploratory. No bootstrap rerun.')
    # 3: operating-policy sensitivity and correlation.
    errors=read('results/deep_dive_d2_v1/common_errors.csv');e=errors.query('split=="test" and fifth=="mean_matched" and models_catching==0').groupby('policy').click_share.mean()
    fig,ax=plt.subplots(1,2,figsize=(12,4.8));fig.suptitle('Common misses depend on the operating policy',fontsize=15)
    policies=['p>=0.5','top_1pct','top_5pct','top_10pct'];v=[e[x]*100 for x in policies]
    ax[0].bar(['p ≥ 0.5','Top 1%','Top 5%','Top 10%'],v,color=[GRAY,BLUE,BLUE,BLUE]);ax[0].set(ylim=(0,108),ylabel='Clicks missed by all five models (%)')
    for i,x in enumerate(v):ax[0].text(i,x+1,f'{x:.2f}%',ha='center',fontsize=9)
    c=read('results/deep_dive_d2_v1/correlations.csv').query('split=="test" and fifth=="mean_matched" and kind=="pearson"').groupby(['left','right']).correlation.mean()
    keys=['LR','LightGBM','WideDeep','DeepFM','mean_matched'];matrix=np.eye(5)
    for (l,r),x in c.items():i=keys.index(l);j=keys.index(r);matrix[i,j]=matrix[j,i]=x
    im=ax[1].imshow(matrix,vmin=.85,vmax=1,cmap='Blues');short=['LR','LGBM','W&D','DeepFM','Mean+F4']
    ax[1].set_xticks(range(5),short,rotation=25);ax[1].set_yticks(range(5),short);ax[1].set_title('Saved probability Pearson correlation')
    for i in range(5):
        for j in range(5):ax[1].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white' if matrix[i,j]>.96 else 'black',fontsize=8)
    fig.colorbar(im,ax=ax[1],shrink=.8)
    save(fig,'03_common_errors','Five models: four tabular + Mean matched. Top budgets fixed in advance.')
    # 4: exact accounting for two pre-existing anomalies.
    l=read('results/deep_dive_d2_v1/long_history_loss.csv').query('split=="test" and reference=="mean_mismatched" and group=="category_hit"')
    label=l.groupby(['seed','label']).contribution_to_long_history_delta.sum().groupby('label').mean()
    cold=read('results/deep_dive_d2_v1/double_cold_composition.csv').query('split=="test" and model=="mean_matched" and stage=="calibrated" and group=="evidence_bits"')
    cs=cold.groupby('value').share_total_excess.mean().sort_values(ascending=False)
    fig,ax=plt.subplots(1,2,figsize=(12,4.8));fig.suptitle('Probability errors: loss balance and cold-history composition',fontsize=15)
    vals=[label[0]*1e4,label[1]*1e4,label.sum()*1e4]
    ax[0].bar(['Nonclicks','Clicks','Net'],vals,color=[ORANGE,BLUE,GRAY]);ax[0].axhline(0,color='black',lw=.8);ax[0].set(ylabel='Contribution to ΔLogLoss × 10⁴',title='Length 20: matched − mismatched')
    ax[1].bar(cs.index,cs.values*100,color=BLUE);ax[1].tick_params(axis='x',rotation=55);ax[1].set(ylabel='Share of excess expected clicks (%)',title='Double cold after saved isotonic calibration')
    save(fig,'04_loss_and_cold_evidence','Bits: F2 behavior / F3 user exposure / F3 ad exposure / F4 positives. Mean across residual seeds. Decomposition is descriptive, not causal.')
    # 5: additive eligible-user GAUC attribution.
    g=read('results/deep_dive_d2_v1/gauc_contributions.csv').query('split=="test" and reference=="zero"')
    fig,ax=plt.subplots(1,2,figsize=(12,4.5));fig.suptitle('Which user groups account for the GAUC increment?',fontsize=15)
    for a,group,order in [(ax[0],'history_bin',['0','1','2-5','6-10','11-19','20']),(ax[1],'f2_activity',['0','1-9','10-99','100-999','1000+'])]:
        v=g[g.group==group].groupby('value').contribution_to_global_delta.mean().reindex(order)*1e4
        a.bar(v.index,v.values,color=BLUE);a.axhline(0,color='black',lw=.8);a.set(xlabel='Retained length' if group=='history_bin' else 'F2 past 14-day events',ylabel='Contribution to overall ΔGAUC × 10⁴')
    save(fig,'05_gauc_contributions','Mean matched − zero; existing eligible-user AUC reused. Each panel sums to +0.000478364; 50.84% impression coverage. No new intervals.')
    # 6: validation-only diagnostics, not new trained models.
    v=read('results/deep_dive_d3_v1/validation_metrics.csv').set_index('diagnostic')
    fig,ax=plt.subplots(1,2,figsize=(12,4.8));fig.suptitle('Validation-only reliance of already fitted models',fontsize=15)
    for model,color in [('LR',BLUE),('DeepFM',ORANGE)]:
        names=['full','no_F2','no_F3','no_F2_F3'];a=[v.loc[model+'_'+x,'auc'] for x in names]
        ax[0].plot(['Full','F2 neutral','F3 neutral','Both neutral'],a,'o-',label=model,color=color)
    ax[0].set(ylabel='Validation AUC',title='Information neutralization');ax[0].legend();ax[0].tick_params(axis='x',rotation=15)
    names=['full','no_FM','no_deep','no_wide'];xs=[v.loc['DeepFM_'+x,'auc'] for x in names]
    ax[1].barh(['Full','Without FM','Without deep','Without wide'],xs,color=ORANGE);ax[1].set(xlim=(.55,.66),xlabel='Validation AUC (axis starts at 0.55)',title='DeepFM logit-component deletion');ax[1].invert_yaxis()
    save(fig,'06_validation_reliance','Frozen parameters; no refitting or test scoring. Neutralization/deletion may be off-distribution and is not a retrained component ablation.')
    # 7: descriptive traffic drift and left-censored repeats.
    daily=read('results/deep_dive_d4_v1/daily_summary.csv');repeat=read('results/deep_dive_d4_v1/repeat_counts.csv')
    fig,ax=plt.subplots(1,3,figsize=(14,4.5));fig.suptitle('Temporal evidence and repeated-exposure associations',fontsize=15)
    x=np.arange(8);labels=[s[-2:] for s in daily.day]
    ax[0].plot(x,daily.ctr*100,'o-',color=BLUE);ax[0].set_xticks(x,labels);ax[0].set(xlabel='May, Beijing date',ylabel='Observed CTR (%)',title='Daily traffic')
    for c,name,color in [('f4_present','F4 positive history',BLUE),('f3_user_present','F3 user history',ORANGE),('f3_ad_present','F3 ad history',GREEN)]:ax[1].plot(x,daily[c]*100,'o-',label=name,color=color,ms=3)
    ax[1].set_xticks(x,labels);ax[1].set(xlabel='May, Beijing date',ylabel='Exposure share (%)',title='History availability');ax[1].legend(fontsize=8)
    for split,color in [('val',BLUE),('test',ORANGE)]:
        r=repeat[repeat.split==split].set_index('repeat_band').reindex(['0','1','2-5','6-19','20+'])
        ax[2].plot(r.index,r.ctr*100,'o-',color=color,label=split)
    ax[2].set(xlabel='Strictly prior-day user/ad exposures',ylabel='CTR (%)',title='Repeat buckets');ax[2].legend();ax[2].text(.99,.95,'20+: only 4 / 27 rows',transform=ax[2].transAxes,ha='right',va='top',fontsize=8)
    save(fig,'07_temporal_repeats','Descriptive dates, not rolling-origin model evaluation. More history partly reflects window accumulation. Repeat association does not establish fatigue.')
    # 8: reuse previously saved calibration group results.
    cg=read('results/sequence_posthoc_v1/group_summary.csv').query('split=="test" and condition=="mean_matched" and group=="cold_start_segment"')
    fig,ax=plt.subplots(figsize=(9,4.7));fig.suptitle('Near-global calibration leaves substantial cold-group bias',fontsize=15)
    order=['known_user__known_ad','known_user__new_ad','new_user__known_ad','new_user__new_ad'];x=np.arange(4)
    for i,(stage,color) in enumerate([('raw',GRAY),('calibrated',BLUE)]):
        v=cg[cg.stage==stage].set_index('value').reindex(order)
        ax.bar(x+(i-.5)*.32,v.copc_mean,width=.32,label=stage,color=color)
    ax.set_xticks(x,['Known / known','Known user / new ad','New user / known ad','New / new']);ax.axhline(1,color='black',ls='--',lw=1)
    ax.set(ylabel='COPC = observed / predicted clicks',ylim=(0,1.2));ax.legend()
    save(fig,'08_calibration_group_bias','Saved validation-selected calibration from badecf4, not refitted. Overall calibrated COPC ≈ 0.986; double cold ≈ 0.581.')
    for path,h in inputs.items():assert digest(ROOT/path)==h
    manifest={'complete':True,'source':'versioned summary CSVs only','elapsed_seconds':time.perf_counter()-begin,
        'inputs_sha256':inputs,'figures_sha256':{str(p.relative_to(ROOT)):digest(p) for p in OUT.iterdir()},
        'versions':{'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__},
        'bootstrap_or_calibration_repeated':False,'model_inference':False,'script_sha256':digest(Path(__file__))}
    (RESULTS/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(list(OUT.glob("*.png")))} standalone figures (PNG + PDF)',flush=True)


if __name__=='__main__':main()
