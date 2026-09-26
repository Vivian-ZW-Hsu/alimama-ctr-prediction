"""Read-only release audit of saved summary arithmetic. No prediction/model loading."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHECKS = defaultdict(int)
READ = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    p = ROOT/path
    READ[path] = sha(p)
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        result = list(reader)
    assert all(None not in r and None not in r.values() for r in result), path
    return result


def eq(a, b, context, tol=3e-12):
    a, b = float(a), float(b)
    assert math.isclose(a,b,rel_tol=tol,abs_tol=tol) or math.isnan(a) and math.isnan(b), (context,a,b)
    CHECKS['numeric_equalities'] += 1


def grouped(data, keys):
    out = defaultdict(list)
    for r in data: out[tuple(r[k] for k in keys)].append(r)
    return out


def summary(raw_path, summary_path, keys):
    raw = grouped(rows(raw_path),keys)
    saved = rows(summary_path)
    assert len(raw) == len(saved), summary_path
    for r in saved:
        group = raw[tuple(r[k] for k in keys)]
        for col, value in r.items():
            if col.endswith(('_mean','_std')):
                metric, kind = col.rsplit('_',1)
                if metric not in group[0]: continue
                vals=[float(x[metric]) for x in group if x[metric] and math.isfinite(float(x[metric]))]
                if not vals: assert value in ('','NaN','nan'); continue
                if kind == 'std' and len(vals) == 1:
                    assert value in ('','0','0.0'), (summary_path,col,value)
                    continue
                eq(value, statistics.mean(vals) if kind=='mean' else statistics.stdev(vals),(summary_path,col))
            elif col in ('predictors','seeds'): eq(value,len(group),(summary_path,col))
            elif col in ('rows','clicks','actual_ctr'):
                for x in group: eq(value,x[col],(summary_path,col))
    CHECKS['summary_tables'] += 1


def metric_identity(path):
    for r in rows(path):
        if {'rows','clicks','actual_ctr'} <= r.keys() and float(r['rows']) > 0:
            eq(r['actual_ctr'],float(r['clicks'])/float(r['rows']),(path,'CTR'))
        if {'actual_ctr','mean_prediction','copc'} <= r.keys() and float(r['mean_prediction']) > 0:
            eq(r['copc'],float(r['actual_ctr'])/float(r['mean_prediction']),(path,'COPC'))


def main():
    post='results/sequence_posthoc_v1/'
    test='results/sequence_test_v2/'
    val='results/sequence_selection_v2/'
    d2='results/deep_dive_d2_v1/'
    d3='results/deep_dive_d3_v1/'
    d4='results/deep_dive_d4_v1/'
    summary(test+'test_metrics.csv',test+'seed_summary.csv',['condition'])
    summary(val+'validation_metrics.csv',val+'validation_seed_summary.csv',['condition'])
    metrics=['auc','logloss','pr_auc','gauc','copc','ece']
    for directory,prefix,raw_file in [(test,'','test_metrics.csv'),(val,'validation_','validation_metrics.csv')]:
        predictors={(r['condition'],r['seed']):r for r in rows(directory+raw_file)}
        diffs=rows(directory+('paired_seed_differences.csv' if directory==test else 'validation_paired_differences.csv'))
        for r in diffs:
            for m in metrics:
                eq(r[m],float(predictors[r['left'],r['seed']][m])-float(predictors[r['right'],r['seed']][m]),('paired',r['id'],r['seed'],m))
        for r in rows(directory+prefix+'comparisons.csv'):
            subset=[x for x in diffs if x['id']==r['id']]
            for m in metrics:
                values=[float(x[m]) for x in subset]
                eq(r['delta_'+m+'_mean'],statistics.mean(values),('comparison mean',m))
                eq(r['delta_'+m+'_std'],statistics.stdev(values),('comparison std',m))
    mappings=[('overall_before_after.csv','overall_summary.csv',['split','condition','stage']),
              ('group_metrics.csv','group_summary.csv',['split','condition','stage','group','value']),
              ('group_paired_differences.csv','group_difference_summary.csv',['split','left','right','group','value']),
              ('calibration_overall_differences.csv','calibration_overall_difference_summary.csv',['split','condition']),
              ('calibration_group_differences.csv','calibration_group_difference_summary.csv',['split','condition','group','value'])]
    for raw,saved,keys in mappings: summary(post+raw,post+saved,keys)
    summary(d2+'group_metrics.csv',d2+'group_summary.csv',['split','model','group','value'])
    for path in [test+'test_metrics.csv',val+'validation_metrics.csv',post+'overall_before_after.csv',post+'group_metrics.csv',d2+'group_metrics.csv',d3+'validation_metrics.csv',d4+'repeat_metrics.csv']:
        metric_identity(path)
    frozen=rows(post+'bootstrap_intervals.csv')
    comparisons={r['id']:r for r in rows(test+'comparisons.csv')}
    for r in frozen:
        eq(r['effect'],comparisons[r['id']]['delta_'+r['metric']+'_mean'],('bootstrap frozen point',r['id'],r['metric']))
        assert float(r['ci_low']) <= float(r['ci_high'])
        assert 0 <= float(r['p_centered_two_sided']) <= 1
    CHECKS['saved_bootstrap_rows_checked_without_resampling']=len(frozen)
    baseline_runs=['lr_f1','lr_f1_f2','lr_f1_f2_f3','lightgbm_f1_f2_f3','wide_deep_f1_f2_f3','deepfm_f1_f2_f3']
    documents=['README.md','docs/FINAL_REPORT.md','docs/FINAL_REPORT_ZH.md','docs/RELEASE_AUDIT.md']
    text={p:(ROOT/p).read_text() for p in documents}
    for run in baseline_runs:
        path=f'results/{run}/evaluation/overall_metrics.csv'
        metric_identity(path)
        result=next(r for r in rows(path) if r['split']=='test')
        gauc=next(r for r in rows(f'results/{run}/evaluation/gauc_summary.csv') if r['split']=='test')
        for doc in ['README.md','docs/FINAL_REPORT_ZH.md']:
            line=next(s for s in text[doc].splitlines() if f"{float(result['auc']):.6f}" in s and s.startswith('|'))
            for m in ['auc','logloss']: assert f'{float(result[m]):.6f}' in line,(doc,run,m)
            assert f"{float(gauc['gauc']):.6f}" in line,(doc,run,'gauc')
            CHECKS['document_table_cells'] += 3
    for r in rows(test+'comparisons.csv'):
        for doc in ['docs/FINAL_REPORT.md','docs/FINAL_REPORT_ZH.md']:
            line=next(s for s in text[doc].splitlines() if f"{float(r['delta_auc_mean']):+.9f}".replace('-','−') in s.replace('-','−'))
            assert f"{float(r['delta_logloss_mean']):+.9f}".replace('-','−') in line.replace('-','−')
            CHECKS['document_table_cells'] += 2
    gm=rows(d2+'group_metrics.csv')
    for key, group in grouped(gm,['split','model','seed','group']).items():
        eq(sum(int(r['rows']) for r in group),3234051 if key[0]=='val' else 3308665,('group rows',key))
        eq(sum(int(r['clicks']) for r in group),159437 if key[0]=='val' else 166490,('group clicks',key))
    for key,group in grouped(rows(d2+'common_errors.csv'),['split','seed','fifth','policy']).items():
        eq(sum(float(r['click_share']) for r in group),1,('common errors',key))
        for r in group: eq(r['click_share'],float(r['clicks'])/float(r['all_clicks']),('miss denominator',key))
    missed=[float(r['click_share']) for r in rows(d2+'common_errors.csv') if r['split']=='test' and r['fifth']=='mean_matched' and r['policy']=='top_10pct' and r['models_catching']=='0']
    assert f'{100*statistics.mean(missed):.2f}'=='71.38'
    for doc,s in text.items():
        for line in s.splitlines():
            if '71.38' in line: assert 'top-10%' in line.lower(),(doc,line)
    gauc=rows(d2+'gauc_contributions.csv')
    totals={key:sum(float(r['contribution_to_global_delta']) for r in g) for key,g in grouped(gauc,['split','seed','reference','group']).items()}
    for reference,ident in [('zero','1'),('mean_mismatched','2')]:
        for group in {r['group'] for r in gauc}:
            effect=statistics.mean(v for (sp,se,ref,gr),v in totals.items() if sp=='test' and ref==reference and gr==group)
            eq(effect,comparisons[ident]['delta_gauc_mean'],('GAUC total',reference,group))
    for r in gauc:
        eq(r['contribution_to_global_delta'],float(r['group_gauc_delta'])*float(r['eligible_rows'])/float(r['all_eligible_rows']),'GAUC weighted contribution')
    for key,group in grouped(rows(d2+'long_history_loss.csv'),['split','seed','reference']).items():
        sums=[]
        for part,sub in grouped(group,['group']).items():
            for r in sub: eq(r['contribution_to_long_history_delta'],float(r['loss_delta_sum'])/float(r['long_history_rows']),'label loss contribution')
            sums.append(sum(float(r['contribution_to_long_history_delta']) for r in sub))
        for x in sums: eq(x,sums[0],('long loss partitions',key))
    for key,group in grouped(rows(d2+'double_cold_composition.csv'),['split','model','seed','stage']).items():
        counts=[]
        for part,sub in grouped(group,['group']).items():
            eq(sum(float(r['share_total_excess']) for r in sub),1,('cold excess shares',key))
            counts.append([sum(float(r[m]) for r in sub) for m in ['rows','clicks','expected_clicks']])
        for c in counts:
            for a,b in zip(c,counts[0]): eq(a,b,('cold partition',key))
    daily=rows(d4+'daily_summary.csv')
    eq(sum(int(r['rows']) for r in daily),26557961,'daily total')
    for r in daily: eq(r['ctr'],int(r['clicks'])/int(r['rows']),'daily CTR')
    for split,group in grouped(rows(d4+'repeat_counts.csv'),['split']).items():
        eq(sum(int(r['rows']) for r in group),{'train':20015245,'val':3234051,'test':3308665}[split[0]],'repeat rows')
        for r in group: eq(r['ctr'],int(r['clicks'])/int(r['rows']),'repeat CTR')
    assert not re.search(r'/Users/|/home/|/opt/|file://|[A-Z]:\\',text['README.md'])
    for doc,s in text.items():
        for link in re.findall(r'\]\(([^)]+)\)',s):
            if not link.startswith(('http:','https:','#')):
                assert ((ROOT/doc).parent/link.split('#')[0]).exists(),(doc,link)
                CHECKS['local_links']+=1
    # Hash and syntax inventory includes historical/smoke tables, whose scopes stay separate.
    tracked=subprocess.check_output(['git','ls-files','results'],cwd=ROOT,text=True).splitlines()
    inventory={}
    for name in tracked:
        if '/release_audit_v1/' in name: continue
        p=ROOT/name
        if p.suffix=='.json':
            json.loads(p.read_text()); inventory[name]={'sha256':sha(p),'format':'json'}
        elif p.suffix=='.csv':
            data=rows(name); inventory[name]={'sha256':sha(p),'format':'csv','rows':len(data)}
    figures=json.loads((ROOT/'results/deep_dive_final_v1/figure_manifest.json').read_text())
    for field in ['inputs_sha256','figures_sha256']:
        for name,digest in figures[field].items(): assert sha(ROOT/name)==digest,name
    legacy=json.loads((ROOT/d2/'legacy_audit.json').read_text())
    assert legacy['legacy_arrays_available'] is False and legacy['sixty_percent_exact_reproducible'] is False
    freeze=json.loads((ROOT/'results/sequence_selection_v2/freeze_manifest.json').read_text())
    assert len(freeze['checkpoints'])==15
    plan=freeze['comparison_plan']
    assert plan['seeds']==[42,2026,3407] and plan['batch_size']==4096 and plan['max_length']==20
    eq(plan['learning_rate'],0.0005,'frozen LR')
    assert all(r['best_epoch']==1 for r in freeze['checkpoints'])
    assert json.loads((ROOT/'results/f4_click_daily_v1/build_config.json').read_text())['seed']==20260926
    protocol=json.loads((ROOT/post/'protocol.json').read_text())
    assert protocol['bootstrap']['seed']==20260927 and protocol['bootstrap']['replicates']==1000
    # Published rounded claims: arithmetic on existing additive summaries only.
    claim_checks={}
    def claim(name,value,display,places,docs):
        assert f'{value:.{places}f}'==display,(name,value,display)
        for doc in docs: assert display in text[doc],(name,doc)
        claim_checks[name]={'unrounded':value,'published':display,'documents':docs}
    main_docs=['README.md','docs/FINAL_REPORT.md','docs/FINAL_REPORT_ZH.md']
    means=next(r for r in rows(test+'seed_summary.csv') if r['condition']=='mean_matched')
    for m in ['auc','logloss','gauc']:
        for suffix in ['mean','std']:
            v=float(means[m+'_'+suffix]);claim('mean_matched '+m+' '+suffix,v,f'{v:.6f}',6,main_docs)
    known=statistics.mean(float(r['contribution_to_global_delta']) for r in gauc if r['split']=='test' and r['reference']=='zero' and r['group']=='known_user' and r['value']=='True')
    claim('known-user share of GAUC delta (%)',100*known/float(comparisons['1']['delta_gauc_mean']),'98.89',2,main_docs)
    lh=[r for r in rows(d2+'long_history_loss.csv') if r['split']=='test' and r['reference']=='mean_mismatched' and r['group']=='raw_history']
    claim('raw length >=50 share of capped loss worsening (%)',100*sum(float(r['loss_delta_sum']) for r in lh if r['value']=='50+')/sum(float(r['loss_delta_sum']) for r in lh),'58.64',2,main_docs[1:])
    cold=next(r for r in rows(post+'group_summary.csv') if r['split']=='test' and r['condition']=='mean_matched' and r['stage']=='calibrated' and r['group']=='cold_start_segment' and r['value']=='new_user__new_ad')
    claim('double cold calibrated COPC',float(cold['copc_mean']),'0.581043',6,main_docs[1:])
    calibrated=next(r for r in rows(post+'overall_summary.csv') if r['split']=='test' and r['condition']=='mean_matched' and r['stage']=='calibrated')
    for metric,display in [('logloss','0.192839887'),('ece','0.001043477')]:
        claim('mean matched calibrated '+metric,float(calibrated[metric+'_mean']),display,9,['docs/FINAL_REPORT_ZH.md'])
    for r in rows(d3+'validation_metrics.csv'):
        if r['diagnostic'] in ['LR_full','LR_no_F2','LR_no_F3','DeepFM_full','DeepFM_no_F2','DeepFM_no_F3','DeepFM_no_FM']:
            claim('validation sensitivity '+r['diagnostic'],float(r['auc']),f"{float(r['auc']):.6f}",6,main_docs[1:])
    for r in rows(post+'calibration_overall_differences.csv'):
        if r['split']=='test': assert all(float(r[m])<0 for m in ['auc','logloss','pr_auc','gauc','ece'])
    for r in rows(post+'training_seed_direction_consistency.csv'): assert r['favorable_seeds']=='3' and r['all_favorable']=='true'
    CHECKS['published_claims']=len(claim_checks)
    print(json.dumps({'passed':True,'scope':'saved summary arithmetic and file integrity only; no prediction reads, model inference, fitting or resampling',
        'checks':dict(CHECKS),'published_claims':claim_checks,'figures_hash_verified':16,'inventory':inventory,
        'document_sha256':{n:sha(ROOT/n) for n in documents},'auditor_sha256':sha(Path(__file__))},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
