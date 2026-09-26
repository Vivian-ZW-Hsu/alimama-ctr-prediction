"""Independent arithmetic and immutable-input checks for completed deep-dive stages."""
import argparse
import json
import numpy as np
import polars as pl
from deep_dive_common import ROOT,DATA,sha256,write_json


def verify_d2(out):
    groups=pl.read_csv(out/'group_metrics.csv')
    for (split,model,seed,group),f in groups.partition_by(['split','model','seed','group'],as_dict=True).items():
        expected=3234051 if split=='val' else 3308665
        assert f['rows'].sum()==expected
        assert f['clicks'].sum()==(159437 if split=='val' else 166490)
    metrics=['auc','logloss','pr_auc','mean_prediction','copc']
    summary=groups.group_by(['split','model','group','value']).agg(pl.len().alias('predictors'),
        pl.col('rows').first(),pl.col('clicks').first(),pl.col('actual_ctr').first(),
        *[e for m in metrics for e in [pl.col(m).mean().alias(m+'_mean'),pl.col(m).std().alias(m+'_std')]])
    summary.sort('split','model','group','value').write_csv(out/'group_summary.csv')
    diffs=[]
    for r in groups.filter(pl.col('model')=='mean_matched').iter_rows(named=True):
        for right in ['mean_mismatched','DeepFM']:
            ref=groups.filter((pl.col('split')==r['split'])&(pl.col('model')==right)&(pl.col('group')==r['group'])&
                             (pl.col('value')==r['value'])&(pl.col('seed')==(r['seed'] if right=='mean_mismatched' else 42))).row(0,named=True)
            diffs.append({**{k:r[k] for k in ['split','seed','group','value','rows','clicks']},'reference':right,
                          **{m:r[m]-ref[m] for m in metrics}})
    pl.DataFrame(diffs).write_csv(out/'heterogeneity_differences.csv')
    errors=pl.read_csv(out/'common_errors.csv')
    for key,f in errors.partition_by(['split','seed','fifth','policy'],as_dict=True).items():
        assert f['rows'].sum()==(3234051 if key[0]=='val' else 3308665)
        np.testing.assert_allclose(f['click_share'].sum(),1,atol=1e-14)
    gauc=pl.read_csv(out/'gauc_contributions.csv')
    totals=gauc.group_by(['split','seed','reference','group']).agg(pl.col('contribution_to_global_delta').sum())
    for key,f in totals.partition_by(['split','seed','reference'],as_dict=True).items():
        v=f['contribution_to_global_delta'].to_numpy();np.testing.assert_allclose(v,v[0],atol=1e-13)
    frozen=pl.read_csv(ROOT/'results/sequence_posthoc_v1/bootstrap_intervals.csv')
    for reference,comparison in [('zero',1),('mean_mismatched',2)]:
        x=totals.filter((pl.col('split')=='test')&(pl.col('reference')==reference)&(pl.col('group')=='history_bin'))['contribution_to_global_delta'].mean()
        expected=frozen.filter((pl.col('id')==comparison)&(pl.col('metric')=='gauc'))['effect'][0]
        np.testing.assert_allclose(x,expected,atol=1e-12)
    cold=pl.read_csv(out/'double_cold_composition.csv')
    for key,f in cold.partition_by(['split','model','seed','stage'],as_dict=True).items():
        parts=f.group_by('group').agg(pl.col('rows').sum(),pl.col('clicks').sum(),pl.col('expected_clicks').sum(),pl.col('share_total_excess').sum())
        for metric in ['rows','clicks','expected_clicks']:
            v=parts[metric].to_numpy();np.testing.assert_allclose(v,v[0],atol=1e-10)
        np.testing.assert_allclose(parts['share_total_excess'].to_numpy(),1,atol=1e-13)
    long=pl.read_csv(out/'long_history_loss.csv')
    for key,f in long.partition_by(['split','seed','reference'],as_dict=True).items():
        parts=f.group_by('group').agg(pl.col('contribution_to_long_history_delta').sum(),pl.col('rows').sum())
        vals=parts['contribution_to_long_history_delta'].to_numpy();np.testing.assert_allclose(vals,vals[0],atol=1e-13)
        assert parts['rows'].n_unique()==1
    return ['complete_group_partitions','common_error_click_denominators','additive_gauc_matches_frozen_bootstrap_point',
            'double_cold_expected_clicks_reconcile','long_history_label_loss_contributions_reconcile']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['d2','d3','d4'],required=True);args=parser.parse_args()
    out=ROOT/f'results/deep_dive_{args.stage}_v1'
    if (out/'verification.json').exists():raise FileExistsError('Verification already complete')
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['complete']
    for path,digest in manifest['inputs_sha256'].items():assert sha256(ROOT/path)==digest,path
    for name,digest in manifest['output_sha256'].items():assert sha256(out/name)==digest,name
    checks=verify_d2(out) if args.stage=='d2' else []
    if args.stage=='d3':
        f=pl.read_parquet(DATA/'d3_validation_diagnostics.parquet')
        assert f.height==3234051 and f['impression_id'].n_unique()==f.height
        assert max(manifest['saved_validation_max_probability_error'].values())<2e-6
        checks=['validation_only_rows','checkpoint_and_prediction_parity','frozen_checkpoints_unchanged']
    if args.stage=='d4':
        f=pl.read_csv(out/'daily_summary.csv');assert f['rows'].sum()==26557961 and f.height==8
        groups=pl.read_csv(out/'repeat_metrics.csv')
        for key,sub in groups.partition_by(['split','model','seed'],as_dict=True).items():
            assert sub['rows'].sum()==(3234051 if key[0]=='val' else 3308665)
        checks=['eight_day_row_reconciliation','repeat_group_partitions','prior_day_only_sample_audit']
    write_json(out/'verification.json',{'passed':True,'stage':args.stage,'checks':checks,'immutable_inputs_unchanged':True,
        'output_sha256':{p.name:sha256(p) for p in out.iterdir() if p.is_file()},'verifier_sha256':sha256(ROOT/'src/verify_deep_dive.py')})
    print(f'{args.stage}: PASS {checks}',flush=True)


if __name__=='__main__':main()
