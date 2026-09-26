"""1000 exact paired user-cluster replicates from frozen prediction ranks."""
import json
import subprocess
import time
import numpy as np
import polars as pl
from ctr_posthoc import ROOT,STUDY,DATA,holm,sha256,write_json

METRICS=('auc','logloss','pr_auc','gauc')


def main():
    if (STUDY/'bootstrap_manifest.json').exists():raise FileExistsError('Bootstrap already complete')
    inp=json.loads((STUDY/'input_manifest.json').read_text());protocol=json.loads((STUDY/'protocol.json').read_text())['bootstrap']
    for name,digest in inp['derived_sha256'].items():assert sha256(ROOT/name)==digest
    exe=DATA/'cluster_bootstrap_metrics'
    if exe.exists():raise FileExistsError('Existing bootstrap executable/output: inspect before resuming')
    subprocess.run(['clang++','-O3','-std=c++17','-pthread',str(ROOT/'src/cluster_bootstrap_metrics.cpp'),'-o',str(exe)],check=True)
    M=len(inp['models']);B=protocol['replicates'];seed=protocol['seed'];threads=protocol['threads']
    t=time.perf_counter()
    subprocess.run([str(exe),str(DATA),str(DATA/'bootstrap_pilot.bin'),str(protocol['pilot_replicates']),str(seed),str(threads)],check=True)
    pilot_seconds=time.perf_counter()-t
    print(f'Exact bootstrap pilot: {pilot_seconds:.2f}s; projected full <= {pilot_seconds*B/protocol["pilot_replicates"]:.1f}s',flush=True)
    pilot=np.fromfile(DATA/'bootstrap_pilot.bin',dtype='<f8').reshape(-1,M,4)
    original=pl.read_csv(ROOT/'results/sequence_test_v2/test_metrics.csv')
    for index,r in enumerate(inp['models']):
        expected=original.filter((pl.col('condition')==r['condition'])&(pl.col('seed')==r['seed'])).row(0,named=True)
        np.testing.assert_allclose(pilot[0,index],[expected[m] for m in METRICS],rtol=1e-10,atol=1e-12)
    assert np.isfinite(pilot).all()
    # Refuse silently expanding the already authorized bounded local analysis.
    assert pilot_seconds*B/protocol['pilot_replicates']<6*3600,'Projected bootstrap exceeds six hours; ask before more compute'
    write_json(STUDY/'bootstrap_pilot.json',{'passed':True,'replicates':protocol['pilot_replicates'],'seconds':pilot_seconds,
        'projected_seconds':pilot_seconds*B/protocol['pilot_replicates'],'point_estimates_match_shared_evaluator':True})
    t=time.perf_counter()
    subprocess.run([str(exe),str(DATA),str(DATA/'bootstrap_full.bin'),str(B),str(seed),str(threads)],check=True)
    elapsed=time.perf_counter()-t
    raw=np.fromfile(DATA/'bootstrap_full.bin',dtype='<f8').reshape(B+1,M,4)
    np.testing.assert_array_equal(raw[:len(pilot)],pilot)
    assert np.isfinite(raw).all(), 'Invalid replicates: do not silently discard'
    plan=json.loads((ROOT/'results/sequence_selection_v2/comparison_plan.json').read_text())
    def values(condition):
        indices=[r['index'] for r in inp['models'] if r['condition']==condition]
        assert len(indices)==(1 if condition=='zero_sequence_same_pipeline' else 3)
        return raw[:,indices,:].mean(axis=1)
    rows=[];draws={'replicate':np.arange(B,dtype=np.uint32)}
    for c in plan['comparisons']:
        delta=values(c['left'])-values(c['right'])
        for k,metric in enumerate(METRICS):
            observed=float(delta[0,k]);boot=delta[1:,k]
            lo,hi=np.quantile(boot,[.025,.975],method='linear')
            p=(1+np.sum(np.abs(boot-observed)>=abs(observed)))/(B+1)
            rows.append({**c,'metric':metric,'effect':observed,'ci_low':lo,'ci_high':hi,'bootstrap_std':float(np.std(boot,ddof=1)),
                         'p_centered_two_sided':float(p),'holm_two_primary_per_metric':None,'holm_joint_primary_auc_logloss':None,
                         'interpretation':'confirmatory_primary' if c['id']<=2 else 'mechanistic_exploratory'})
            draws[f'comparison_{c["id"]}_{metric}']=boot
    for metric in METRICS:
        ix=[i for i,r in enumerate(rows) if r['id']<=2 and r['metric']==metric]
        for i,p in zip(ix,holm([rows[i]['p_centered_two_sided'] for i in ix])):rows[i]['holm_two_primary_per_metric']=float(p)
    ix=[i for i,r in enumerate(rows) if r['id']<=2 and r['metric'] in ('auc','logloss')]
    for i,p in zip(ix,holm([rows[i]['p_centered_two_sided'] for i in ix])):rows[i]['holm_joint_primary_auc_logloss']=float(p)
    pl.DataFrame(rows).write_csv(STUDY/'bootstrap_intervals.csv');pl.DataFrame(draws).write_csv(STUDY/'bootstrap_deltas.csv')
    seed_diffs=pl.read_csv(ROOT/'results/sequence_test_v2/paired_seed_differences.csv').filter(pl.col('id').is_in([2,4]))
    seed_diffs.write_csv(STUDY/'training_seed_identity_differences.csv')
    consistency=[]
    for i in (2,4):
        sub=seed_diffs.filter(pl.col('id')==i)
        for metric in METRICS:
            v=sub[metric].to_numpy();wins=v<0 if metric=='logloss' else v>0
            consistency.append({'comparison_id':i,'metric':metric,'mean':float(v.mean()),'training_seed_std':float(v.std(ddof=1)),
                                'favorable_seeds':int(wins.sum()),'seeds':3,'all_favorable':bool(wins.all())})
    pl.DataFrame(consistency).write_csv(STUDY/'training_seed_direction_consistency.csv')
    write_json(STUDY/'bootstrap_manifest.json',{'complete':True,'valid_replicates':B,'invalid_replicates':0,'seed':seed,'threads':threads,
        'users':inp['users'],'rows':inp['rows'],'distinct_predictors':M,'zero_predictors':1,'user_draws_per_replicate':inp['users'],
        'protocol_sha256':sha256(STUDY/'protocol.json'),'input_manifest_sha256':sha256(STUDY/'input_manifest.json'),
        'elapsed_seconds':elapsed,'binary_sha256':sha256(exe),'raw_results_sha256':sha256(DATA/'bootstrap_full.bin'),
        'code_sha256':{p:sha256(ROOT/'src'/p) for p in ('cluster_bootstrap_metrics.cpp','29_bootstrap_sequence_comparisons.py','ctr_posthoc.py')},
        'compiler':subprocess.check_output(['clang++','--version'],text=True),'flags':['-O3','-std=c++17','-pthread'],
        'pilot_reproduces_full_prefix':True,'intervals':'marginal percentile 95%; not simultaneous','p_resolution':1/(B+1),
        'uncertainty':'User-sampling conditional on all fixed checkpoints and test day; no bootstrap over training seeds'})
    print(pl.DataFrame(rows).select('id','metric','effect','ci_low','ci_high','holm_joint_primary_auc_logloss'),flush=True)


if __name__=='__main__':main()
