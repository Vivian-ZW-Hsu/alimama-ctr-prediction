"""Evaluate saved predictions once; report the five comparisons fixed before test."""
import argparse
import json
import subprocess
import sys

import numpy as np
import polars as pl

from ctr_nn_common import ROOT
from f4_sequences import sha256, write_json

METRICS=('auc','logloss','pr_auc','gauc','copc','ece')


def summarize(rows, plan):
    frame=pl.DataFrame(rows)
    assert frame.height==15
    for condition in plan['conditions']:
        subset=frame.filter(pl.col('condition')==condition)
        assert subset.height==3 and sorted(subset['seed'].to_list())==plan['seeds']
    summary=frame.group_by('condition').agg(pl.len().alias('seeds'),*[expr for m in METRICS
        for expr in (pl.col(m).mean().alias(m+'_mean'),pl.col(m).std(ddof=1).alias(m+'_std'))]).sort('condition')
    paired=[]
    for comparison in plan['comparisons']:
        for seed in plan['seeds']:
            left=frame.filter((pl.col('condition')==comparison['left'])&(pl.col('seed')==seed)).row(0,named=True)
            right=frame.filter((pl.col('condition')==comparison['right'])&(pl.col('seed')==seed)).row(0,named=True)
            paired.append({**comparison,'seed':seed,**{m:left[m]-right[m] for m in METRICS}})
    paired=pl.DataFrame(paired)
    comparisons=paired.group_by('id','priority','left','right').agg(*[expr for m in METRICS for expr in
        (pl.col(m).mean().alias('delta_'+m+'_mean'),pl.col(m).std(ddof=1).alias('delta_'+m+'_std'))]).sort('id')
    return frame,summary,paired,comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke-test',action='store_true');args=parser.parse_args()
    study=ROOT/'results'/('sequence_test_v2_smoke' if args.smoke_test else 'sequence_test_v2')
    if (study/'report_manifest.json').exists():raise FileExistsError('Report already complete')
    manifest=json.loads((study/'manifest.json').read_text())
    freeze_path=ROOT/'results/sequence_selection_v2/freeze_manifest.json'
    assert manifest['complete'] and sha256(freeze_path)==manifest['freeze_sha256']
    freeze=json.loads(freeze_path.read_text());plan=freeze['comparison_plan'];split=manifest['split']
    assert split==('val' if args.smoke_test else 'test')
    rows=[];logs=study/'evaluation_logs';logs.mkdir(exist_ok=True)
    for record in manifest['records']:
        name=record['run_name'];pred=ROOT/'artifacts/predictions'/name/f'{split}_predictions.parquet'
        assert sha256(pred)==record['prediction_sha256'] and record['inference_passes']==1
        directory=ROOT/'results'/name/'evaluation'
        if not (directory/'evaluation_manifest.json').exists():
            with (logs/f'{name}.log').open('x') as log:
                subprocess.run([sys.executable,'-B','src/10_evaluate_predictions.py','--run-name',name,'--splits',split],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((directory/'evaluation_manifest.json').read_text())['splits']==[split]
        metric=pl.read_csv(directory/'overall_metrics.csv').row(0,named=True)
        gauc=pl.read_csv(directory/'gauc_summary.csv').row(0,named=True)
        assert metric['rows']==record['rows']
        rows.append({'run_name':name,'condition':record['condition'],'seed':record['seed'],**metric,
            **{k:gauc[k] for k in ('gauc','eligible_users','eligible_rows','eligible_row_coverage')}})
    frames=summarize(rows,plan)
    outputs=['test_metrics.csv','seed_summary.csv','paired_seed_differences.csv','comparisons.csv']
    for frame,name in zip(frames,outputs):frame.write_csv(study/name)
    # Historical reference is outside the 15-row fair comparison table.
    reference=pl.read_csv(ROOT/'results/deepfm_f1_f2_f3/evaluation/overall_metrics.csv').filter(pl.col('split')==split)
    ref_gauc=pl.read_csv(ROOT/'results/deepfm_f1_f2_f3/evaluation/gauc_summary.csv').filter(pl.col('split')==split)
    reference.with_columns(pl.lit('historical_only_seed42').alias('role'),pl.lit(ref_gauc['gauc'][0]).alias('gauc')).write_csv(study/'historical_reference.csv')
    zeros=[r for r in manifest['records'] if r['condition']=='zero_sequence_same_pipeline']
    assert len({r['prediction_sha256'] for r in zeros})==1
    old=pl.read_parquet(ROOT/'artifacts/predictions/deepfm_f1_f2_f3'/f'{split}_predictions.parquet').sort('impression_id')
    if args.smoke_test:old=old.filter(pl.col('impression_id')%1000==0)
    actual=pl.read_parquet(ROOT/'artifacts/predictions'/zeros[0]['run_name']/f'{split}_predictions.parquet').sort('impression_id')
    np.testing.assert_array_equal(old['impression_id'].to_numpy(),actual['impression_id'].to_numpy())
    error=float(np.max(np.abs(old['prediction'].to_numpy()-actual['prediction'].to_numpy())))
    assert error<=1e-6
    evidence={str(p.relative_to(ROOT)):sha256(p) for p in study.glob('*.csv')}
    for record in manifest['records']:
        for p in (ROOT/'results'/record['run_name']/'evaluation').glob('*'):
            if p.suffix in ('.csv','.json'):evidence[str(p.relative_to(ROOT))]=sha256(p)
    write_json(study/'report_manifest.json',{'complete':True,'split':split,'smoke':args.smoke_test,
        'inference_manifest_sha256':sha256(study/'manifest.json'),'freeze_sha256':sha256(freeze_path),
        'conditions':5,'seeds':plan['seeds'],'comparisons':plan['comparisons'],
        'zero_seed_predictions_identical':True,'zero_historical_reference_max_error':error,
        'primary_model':'mean_matched','std_definition':'sample std ddof=1; zero std is deterministic, not independent backbone replication',
        'predictions_reused_without_inference':True,'evidence_sha256':evidence})
    print(frames[1]);print(frames[3].select('id','priority','delta_auc_mean','delta_logloss_mean'))


if __name__=='__main__':main()
