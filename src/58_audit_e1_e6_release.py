"""E1-E6 saved-result release audit. Does not load predictions or execute models."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCOPE='saved-result arithmetic, cross-table consistency, protocol metadata and document claims; no model inference, no resampling and no causal validation'
READ={}; CHECKS=Counter()


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(name):
    p=ROOT/name;READ[name]=sha(p)
    if p.suffix=='.json':return json.loads(p.read_text())
    if p.suffix=='.csv':
        with p.open(newline='') as f:return list(csv.DictReader(f))
    return p.read_text()


def eq(a,b,context='arithmetic',tol=3e-12):
    assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(context,a,b)
    CHECKS['numeric_equalities']+=1


def holm(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);bound=0.
    for rank,i in enumerate(order):
        bound=max(bound,min(1.,values[i]*(len(values)-rank)));out[i]=bound
    return out


def check_summary(raw,saved):
    assert set(r['condition'] for r in raw)==set(r['condition'] for r in saved)
    for r in saved:
        group=[v for v in raw if v['condition']==r['condition']]
        assert sorted(int(v['seed']) for v in group)==[42,2026,3407]
        eq(r['seeds'],len(group))
        for col,val in r.items():
            if col.endswith(('_mean','_std')):
                metric,kind=col.rsplit('_',1);vals=[float(v[metric]) for v in group]
                eq(val,statistics.mean(vals) if kind=='mean' else statistics.stdev(vals),(r['condition'],col))
        if 'zero' in r['condition']:
            for m in ['auc','gauc']:
                # GAUC reduction order differs at ~1e-16 despite identical zero predictions.
                for v in group:eq(v[m],group[0][m],'deterministic zero metric',tol=1e-14)
    CHECKS['seed_summary_tables']+=1


def check_intervals(intervals,draws,primary,holm_col):
    """Arithmetic over saved replicates only; no RNG, cluster draws or predictions."""
    assert len(draws)==1000 and sorted(int(r['replicate']) for r in draws)==list(range(1000))
    family=[]
    for r in intervals:
        vals=np.array([float(d[f"comparison_{r['id']}_{r['metric']}"]) for d in draws])
        assert np.isfinite(vals).all()
        effect=float(r['effect']);lo,hi=np.quantile(vals,[.025,.975])
        eq(r['ci_low'],lo);eq(r['ci_high'],hi);eq(r['bootstrap_std'],np.std(vals,ddof=1))
        assert lo<=hi
        p=(1+int(np.sum(np.abs(vals-effect)>=abs(effect))))/(len(vals)+1)
        eq(r['p_centered_two_sided'],p)
        assert 0<=p<=1
        assert (r['excludes_zero']=='true')==bool(lo>0 or hi<0)
        if primary(r):family.append(r)
        else:assert r[holm_col]=='','unexpected correction outside frozen family'
    for r,adjusted in zip(family,holm([float(r['p_centered_two_sided']) for r in family])):
        eq(r[holm_col],adjusted)
    CHECKS['saved_interval_rows']+=len(intervals)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='New JSON destination; default stdout')
    args=parser.parse_args()
    if args.output and args.output.exists():raise FileExistsError(args.output)
    ns=read('results/behavior_namespace_audit_v1/namespace_audit.json')
    for r in read('results/behavior_namespace_audit_v1/namespace_coverage.csv'):
        n=ns['namespaces'][r['relation']]
        for k in ['left_distinct','right_distinct','shared_distinct']:eq(r[k],n[k])
        eq(r['forward_coverage'],int(r['shared_distinct'])/int(r['left_distinct']))
        eq(r['reverse_coverage'],int(r['shared_distinct'])/int(r['right_distinct']))
        eq(r['forward_coverage'],n['forward_coverage']);eq(r['reverse_coverage'],n['reverse_coverage'])
    t=ns['time_distribution'];daily=read('results/behavior_namespace_audit_v1/daily_events.csv')
    eq(sum(int(r['events']) for r in daily),t['rows'])
    eq(sum(int(r['events']) for r in daily if '2017-04-22'<=r['event_date_bj']<'2017-05-14'),t['formal_window']['events'])
    eq(sum(int(r['events']) for r in read('results/behavior_namespace_audit_v1/behavior_types.csv')),t['formal_window']['events'])
    threshold=ns['thresholds'];v=ns['verdict']
    cat=ns['namespaces']['category']['forward_coverage']>=threshold['ad_category_forward_coverage_min']
    usr=ns['namespaces']['user']['forward_coverage']>=threshold['impression_user_forward_coverage_min']
    brand=ns['namespaces']['brand']['forward_coverage']>=threshold['ad_brand_forward_coverage_min']
    assert v['category_precondition_met']==cat and v['user_precondition_met']==usr and v['f5_admissible']==(cat and usr)
    assert v['brand_token_admissible']==brand
    e1doc=read('docs/BEHAVIOR_NAMESPACE_AUDIT.md')
    for value in ['723,268,134','99.8966','99.2807','99.5280']:assert value in e1doc,value
    e3='results/frozen_model_bootstrap_v1/'
    e3rows=read(e3+'bootstrap_intervals.csv')
    check_intervals(e3rows,read(e3+'bootstrap_deltas.csv'),lambda r:r['id']=='1', 'holm_primary')
    protocol=read(e3+'protocol.json');e3manifest=read(e3+'bootstrap_manifest.json')
    eq(protocol['bootstrap']['seed'],e3manifest['seed']);eq(e3manifest['valid_replicates'],1000)
    models={m['key']:m['run_name'] for m in protocol['models']}
    e3points={}
    for key,run in models.items():
        e3points[key]=next(r for r in read(f'results/{run}/evaluation/overall_metrics.csv') if r['split']=='test')
        e3points[key].update(next(r for r in read(f'results/{run}/evaluation/gauc_summary.csv') if r['split']=='test'))
    for r in e3rows:
        eq(r['effect'],float(e3points[r['left']][r['metric']])-float(e3points[r['right']][r['metric']]))
        assert r['interpretation']==('confirmatory_primary' if r['id']=='1' else 'exploratory_secondary')
    val='results/f5_selection_v1/';test='results/f5_sequence_test_v1/'
    check_summary(read(val+'validation_metrics.csv'),read(val+'validation_seed_summary.csv'))
    raw=read(test+'test_metrics.csv');summary=read(test+'test_seed_summary.csv');check_summary(raw,summary)
    freeze=read(val+'freeze_manifest.json');plan={str(p['id']):p for p in freeze['comparison_plan']}
    assert freeze['seed_list']==[42,2026,3407] and freeze['max_length']==200
    eq(freeze['selected_learning_rate'],.0005)
    assert len(plan)==6 and [plan[str(i)]['priority'] for i in range(1,7)]==['primary']*2+['secondary']*4
    byseed={(r['condition'],r['seed']):r for r in raw};paired=read(test+'paired_seed_differences.csv')
    metrics=['auc','logloss','pr_auc','gauc','copc','ece']
    assert len(paired)==18
    for r in paired:
        for side in ['left','right']:assert r[side]==plan[r['id']][side]
        for m in metrics:eq(r[m],float(byseed[r['left'],r['seed']][m])-float(byseed[r['right'],r['seed']][m]))
    effects=read(test+'comparison_effects.csv');effectmap={}
    summaries={r['condition']:r for r in summary}
    for r in effects:
        assert all(r[k]==plan[r['id']][k] for k in ['left','right','priority'])
        eq(r['effect'],statistics.mean(float(x[r['metric']]) for x in paired if x['id']==r['id']))
        eq(r['left_mean'],summaries[r['left']][r['metric']+'_mean']);eq(r['right_mean'],summaries[r['right']][r['metric']+'_mean'])
        eq(r['effect'],float(r['left_mean'])-float(r['right_mean']));effectmap[r['id'],r['metric']]=float(r['effect'])
    intervals=read(test+'bootstrap_intervals.csv')
    for r in intervals:
        eq(r['effect'],effectmap[r['id'],r['metric']])
        assert all(r[k]==plan[r['id']][k] for k in ['left','right','priority'])
    check_intervals(intervals,read(test+'bootstrap_deltas.csv'),lambda r:r['priority']=='primary' and r['metric'] in ['auc','logloss'],'holm_joint_primary_auc_logloss')
    gauc={r['id']:r for r in intervals if r['metric']=='gauc'}
    assert all(gauc[i]['excludes_zero']=='false' for i in ['1','2','6'])
    assert all(gauc[i]['excludes_zero']=='true' for i in ['3','4','5'])
    donor='results/f5_donor_uniqueness_audit_v1/'
    dm=read(donor+'manifest.json');ds=read(donor+'daily_summary.csv');degrees=read(donor+'indegree_distribution.csv')
    assert len(ds)==8
    for r in ds:
        n=int(r['assignments']);dist=[v for v in degrees if v['date']==r['date']]
        eq(sum(int(v['donors'])*int(v['indegree']) for v in dist),n)
        eq(sum(int(v['donors']) for v in dist),r['eligible_donors'])
        distinct=sum(int(v['donors']) for v in dist if int(v['indegree'])>0)
        eq(r['distinct_donors'],distinct);eq(r['reuse_excess'],n-distinct)
        eq(r['reuse_excess_rate'],(n-distinct)/n);eq(r['unique_donor_rate'],distinct/n)
        eq(r['unused_eligible_donors'],n-distinct);eq(r['unused_eligible_rate'],(n-distinct)/n)
        eq(r['reused_donors'],sum(int(v['donors']) for v in dist if int(v['indegree'])>1))
        eq(r['targets_with_reused_donor'],sum(int(v['donors'])*int(v['indegree']) for v in dist if int(v['indegree'])>1))
        eq(r['max_indegree'],max(int(v['indegree']) for v in dist if int(v['donors'])>0))
        eq(r['affected_impression_share'],int(r['affected_impressions'])/int(r['day_impressions']))
        if r['split'] in ['val','test']:
            saved=dm['validation' if r['split']=='val' else 'test']
            for k,value in saved.items():
                if isinstance(value,(int,float)):eq(r[k],value)
    for r in read(donor+'integrity.csv'):
        assert all(int(v)==0 for k,v in r.items() if k!='date')
    for r in read(donor+'partition_summary.csv'):
        eq(r['reuse_excess'],int(r['assignments'])-int(r['distinct_donors']))
        eq(r['reuse_excess_rate'],int(r['reuse_excess'])/int(r['assignments']))
    for r in ds:
        parts=[v for v in read(donor+'partition_summary.csv') if v['date']==r['date']]
        for axis in ['fallback','length','activity_bucket','recency_bucket']:
            subset=[v for v in parts if v['axis']==axis]
            eq(sum(int(v['assignments']) for v in subset),r['assignments'])
            eq(sum(int(v['targets_with_reused_donor_globally']) for v in subset),r['targets_with_reused_donor'])
    store=read('results/f5_behavior_v1/manifest.json')['event_store']
    eq(dm['event_store']['unknown_share'],store['unknown_category_tokens']/store['events'])
    token=read('results/f5_behavior_v1/audit_2017-05-13.json')['token_stats']['matched']
    eq(token['unknown_share'],token['unknown_tokens']/token['tokens'])
    serving={r['model']:r for r in read('results/serving_benchmark_v1/serving_benchmark.csv')}
    deep,f5=serving['deepfm_f123'],serving['f5_mean_matched']
    ratios=dict(pure_slowdown=float(deep['pure_rps_batch_8192'])/float(f5['pure_rps_batch_8192']),
        prepare_plus_inference_slowdown=float(deep['end_to_end_rps_batch_8192'])/float(f5['end_to_end_rps_batch_8192']),
        peak_rss_ratio=float(f5['peak_rss_mib'])/float(deep['peak_rss_mib']))
    assert [f'{x:.2f}' for x in ratios.values()]==['2.32','3.27','7.48']
    registry=read('results/experiment_registry.csv');rm=read('results/experiment_registry_manifest.json')
    stages=Counter(r['stage'] for r in registry)
    assert len(registry)==rm['directories']==100 and len(stages)==16 and stages==rm['stages']
    assert not any(r['stage'] in ['unclassified','unknown',''] for r in registry)
    counts=Counter()
    for r in registry:counts[r['stage']]+=int(r['test_inference_count'])
    actual={k:v for k,v in counts.items() if v}
    expected={'f4_sequence_test':15,'f5_sequence_test':15,'tabular_information_ablation':3,'tabular_benchmark':3,'legacy':2}
    assert actual==expected;eq(sum(actual.values()),rm['total_test_inferences']);eq(sum(actual.values()),38)
    # Registry is a frozen snapshot, not an inventory of these new audit directories.
    qa=read('results/release_audit_v3/pdf_review_v3.json')
    assert qa['page_count']==qa['expected_pages']==len(qa['pages'])
    assert qa['pdf_sha256']==sha(ROOT/'docs/FINAL_REPORT_ZH.pdf')
    assert qa['source_sha256']==sha(ROOT/'docs/FINAL_REPORT_ZH.md')
    assert qa['missing_glyphs']==[] and qa['visual_review_complete']
    assert len(qa['page_reviews'])==qa['page_count']
    assert all(p['embedded_truetype'] and p['text_within_page'] for p in qa['pages'])
    assert all(p['passed'] for p in qa['page_reviews'])
    docs={p:read(p) for p in ['README.md','docs/FINAL_REPORT.md','docs/FINAL_REPORT_ZH.md','docs/F5_BEHAVIOR_TEST_RESULTS.md',
        'docs/F5_BEHAVIOR_VALIDATION.md','docs/SERVING_BENCHMARK.md','docs/RELEASE_AUDIT.md','docs/F5_DONOR_UNIQUENESS_AUDIT.md']}
    for name in ['README.md','docs/FINAL_REPORT.md','docs/FINAL_REPORT_ZH.md']:
        s=docs[name]
        for value in ['0.00655565','0.00596489','0.00721123','0.001998','0.00127246','0.00094211',
                      '0.00023967','0.078109%','0.080850%','0.475843%','0.388223%',
                      '0.04982','138.23','87.72','4.3029%','3.941954%','1,530,445','658,608','1/2.32','1/3.27','7.48']:
            assert value in s,(name,value);CHECKS['document_numeric_claims']+=1
    # Check published donor fractions against arithmetic, rather than literals alone.
    for r in ds[-2:]:
        for field in ['reuse_excess_rate','affected_impression_share']:
            rendered=f"{100*float(r[field]):.6f}%"
            for name in docs:
                if name!='docs/SERVING_BENCHMARK.md':
                    assert rendered in docs[name],(name,field,rendered)
                    CHECKS['document_donor_rates']+=1
    balance=read(val+'validation_control_balance.json')
    eq(balance['impression_weighted_category_total_variation'],.049817542132944356)
    for side,display in [('matched','138.23'),('mismatched','87.72')]:
        assert f"{balance['nonempty_exposure_weighted_past_activity_mean'][side]:.2f}"==display
    assert f"{100*token['unknown_tokens']/token['tokens']:.6f}%"=='3.941954%'
    assert f"{100*store['unknown_category_tokens']/store['events']:.4f}%"=='4.3029%'
    for row in summary:
        if row['condition']=='behavior_zero_same_pipeline':continue
        for name in ['docs/F5_BEHAVIOR_TEST_RESULTS.md']:
            for metric in ['auc','logloss','gauc']:
                assert f"{float(row[metric+'_mean']):.6f}" in docs[name],(name,row['condition'],metric)
                CHECKS['document_metric_means']+=1
    for row in read(val+'validation_seed_summary.csv'):
        for metric in ['auc','log_loss']:
            assert f"{float(row[metric+'_mean']):.10f}" in docs['docs/F5_BEHAVIOR_VALIDATION.md']
            CHECKS['document_metric_means']+=1
    for r in intervals:
        if r['metric'] in ['auc','logloss']:
            for field in ['effect','ci_low','ci_high']:
                value=f"{float(r[field]):+.8f}".replace('-','−')
                assert value in docs['docs/F5_BEHAVIOR_TEST_RESULTS.md'].replace('-','−'),value
                CHECKS['document_interval_cells']+=1
    for name,s in docs.items():
        for bad in ['95% user-specific','model* cost is negligible','not the binding constraint',
                    '身份特有','身份本身，而非','Parquet batch read, encoding','not an imbalance in the']:
            assert bad not in s,(name,bad)
        for line in s.splitlines():
            if '71.38' in line:assert 'top-10%' in line.lower(),(name,line)
        for link in re.findall(r'\]\(([^)]+)\)',s):
            if not link.startswith(('https:','http:','#')):
                assert ((ROOT/name).parent/link.split('#')[0]).exists(),(name,link)
                CHECKS['relative_links']+=1
    assert not re.search(r'/Users/|/home/|/opt/|file://',docs['README.md'])
    for name in ['README.md','docs/FINAL_REPORT.md','docs/SERVING_BENCHMARK.md']:
        for phrase in ['in-memory prepare + inference','end_to_end_*','deserialization','Cold-cache']:
            assert phrase.lower() in docs[name].lower(),(name,phrase)
    for name in ['README.md','docs/FINAL_REPORT.md','docs/F5_BEHAVIOR_TEST_RESULTS.md']:
        assert 'exploratory GAUC' in docs[name] and 'family-wise correction' in docs[name]
    assert f"{qa['page_count']} 页" in docs['docs/RELEASE_AUDIT.md']
    assert f"--expected-pages {qa['page_count']}" in docs['docs/RELEASE_AUDIT.md']
    result=dict(passed=True,scope=SCOPE,source_sha256=sha(__file__),checks=dict(CHECKS),serving_ratios=ratios,
        descriptive_auc_gap_ratio=effectmap['2','auc']/effectmap['1','auc'],
        event_unknown_share=store['unknown_category_tokens']/store['events'],
        test_matched_retained_unknown_share=token['unknown_tokens']/token['tokens'],
        registry_snapshot=dict(directories=100,stages=16,unclassified=0,test_composition=actual,total=38),
        pdf_pages=qa['page_count'],input_sha256=READ,
        limitations=['Checks establish the named arithmetic/metadata/wording properties, not causal validity.',
          'Saved bootstrap replicate quantiles/p-values are checked without generating any new replicate.',
          '38 is the frozen inventory of saved formal prediction outputs; legacy overwrite-free execution cannot be proved from file counts alone.',
          'The frozen registry predates new audit directories and is not overwritten.'])
    output=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    else:print(output)

if __name__=='__main__':main()
