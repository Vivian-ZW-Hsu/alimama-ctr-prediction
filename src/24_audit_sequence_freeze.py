"""Audit the validation freeze and report preregistered history-length diagnostics."""
import importlib.util
import json
from datetime import date

import numpy as np
import polars as pl

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER
from f4_sequences import sha256, source_fingerprints, write_json, midnight


def control_balance():
    """Describe remaining confounding on validation, without changing donors."""
    directory = ROOT/'data/processed/f4_click_daily_v1'
    refs = pl.read_parquet(directory/'val_2017-05-12.parquet')
    events = pl.read_parquet(directory/'events.parquet',columns=['cate_idx','time_stamp'])
    categories = events['cate_idx'].to_numpy(); timestamps = events['time_stamp'].to_numpy()
    size = int(categories.max())+1
    counts = {source:np.zeros(size,dtype=np.int64) for source in ('matched','mismatched')}
    age_sums = {source:0 for source in counts}
    cutoff = midnight(date(2017,5,12))
    for chunk in refs.iter_slices(65536):
        lengths = chunk['length'].to_numpy()
        mask = np.arange(20)[None,:] < lengths[:,None]
        for source,key in (('matched','start'),('mismatched','donor_start')):
            index = (chunk[key].to_numpy()[:,None]+np.arange(20)[None,:])[mask]
            counts[source] += np.bincount(categories[index],minlength=size)
            ages = cutoff-timestamps[index]
            assert (ages>0).all()
            age_sums[source] += int(ages.sum())
    assert counts['matched'].sum() == counts['mismatched'].sum()
    history = pl.read_parquet(directory/'history_2017-05-12.parquet')
    activity = pl.scan_parquet(ROOT/'data/processed/base/train.parquet').group_by('user_raw').agg(
        pl.len().alias('past_exposures')).collect()
    lookup = history.select('user_raw','activity_bucket').join(activity,on='user_raw',validate='1:1')
    joined = refs.filter(pl.col('length')>0).join(lookup,on='user_raw',validate='m:1').join(
        lookup.rename({'user_raw':'donor_user_raw','activity_bucket':'donor_bucket','past_exposures':'donor_past_exposures'}),
        on='donor_user_raw',validate='m:1')
    assert joined.height == refs.filter(pl.col('length')>0).height
    mismatches = joined.filter(pl.col('activity_bucket') != pl.col('donor_bucket')).height
    assert mismatches == 0
    distributions = {source:counts[source]/counts[source].sum() for source in counts}
    return {
        'split':'val','computed_after_configuration_selection':True,
        'tokens_per_condition':int(counts['matched'].sum()),
        'impression_weighted_category_total_variation':float(np.abs(distributions['matched']-distributions['mismatched']).sum()/2),
        'mean_event_age_days':{source:age_sums[source]/int(counts[source].sum())/86400 for source in counts},
        'activity_bucket_mismatch_rows':mismatches,
        'nonempty_exposure_weighted_past_activity_mean':{
            'matched':joined['past_exposures'].mean(),'mismatched':joined['donor_past_exposures'].mean()},
        'limitation':'Exact length and activity bucket are controlled; impression-weighted category popularity, age and within-bucket activity need not be identical. This is not a complete causal adjustment.'}


def main():
    study = ROOT/'results/sequence_selection_v1'
    if (study/'freeze_audit.json').exists():
        raise FileExistsError('Freeze audit already exists')
    freeze_path = study/'freeze_manifest.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['complete'] and not freeze['test_inference_performed']
    assert not freeze['test_authorized']
    assert sha256(study/'preregistration.json') == freeze['preregistration_sha256']
    assert sha256(study/'selection.json') == freeze['selection_sha256']
    assert sha256(BASE_MODEL) == freeze['base_model_sha256']
    assert sha256(BASE_SCALER) == freeze['base_scaler_sha256']
    for name,digest in freeze['code_sha256'].items():assert sha256(ROOT/'src'/name) == digest
    build = json.loads((ROOT/'results/f4_click_daily_v1/build_config.json').read_text())
    assert source_fingerprints() == build['inputs']
    spec = importlib.util.spec_from_file_location('shared_eval',ROOT/'src/10_evaluate_predictions.py')
    shared = importlib.util.module_from_spec(spec); spec.loader.exec_module(shared)
    lengths = pl.read_parquet(ROOT/'data/processed/f4_click_daily_v1/val_2017-05-12.parquet',
                              columns=['impression_id','raw_length']).with_columns(
        pl.when(pl.col('raw_length')==0).then(pl.lit('0'))
        .when(pl.col('raw_length')==1).then(pl.lit('1'))
        .when(pl.col('raw_length')<=5).then(pl.lit('2-5'))
        .when(pl.col('raw_length')<=10).then(pl.lit('6-10'))
        .when(pl.col('raw_length')<=20).then(pl.lit('11-20'))
        .otherwise(pl.lit('>20')).alias('history_bin'))
    baseline = pl.read_parquet(ROOT/'artifacts/predictions/deepfm_f1_f2_f3/val_predictions.parquet').sort('impression_id')
    empty_ids = lengths.filter(pl.col('raw_length')==0)['impression_id']
    baseline_empty = baseline.filter(pl.col('impression_id').is_in(empty_ids.implode()))['prediction'].to_numpy()
    rows = []; empty_checks = []; evidence = {}
    conditions = [{'run_name':'deepfm_f1_f2_f3','pooling':'none','source':'none','seed':42},*freeze['checkpoints']]
    for condition in conditions:
        name = condition['run_name']
        if name != 'deepfm_f1_f2_f3':
            checkpoint = ROOT/'artifacts/models'/name/'best_model.pt'
            pred_path = ROOT/'artifacts/predictions'/name/'val_predictions.parquet'
            assert sha256(checkpoint) == condition['checkpoint_sha256']
            assert sha256(pred_path) == condition['validation_prediction_sha256']
            assert not (pred_path.parent/'test_predictions.parquet').exists()
            pred = pl.read_parquet(pred_path).sort('impression_id')
            np.testing.assert_array_equal(pred['impression_id'].to_numpy(),baseline['impression_id'].to_numpy())
            np.testing.assert_array_equal(pred['clk'].to_numpy(),baseline['clk'].to_numpy())
            empty = pred.filter(pl.col('impression_id').is_in(empty_ids.implode()))['prediction'].to_numpy()
            np.testing.assert_array_equal(empty,baseline_empty)
            empty_checks.append({'run_name':name,'rows':len(empty),'max_prediction_error':0.0})
            for path in sorted((ROOT/'results'/name).rglob('*')):
                if path.is_file() and path.suffix in ('.json','.csv'):
                    evidence[str(path.relative_to(ROOT))] = sha256(path)
        frame = shared.load_and_validate('val',name,'raw').join(lengths,on='impression_id',validate='1:1')
        assert frame.height == 3_234_051 and frame['history_bin'].null_count() == 0
        for label in ('0','1','2-5','6-10','11-20','>20'):
            group = frame.filter(pl.col('history_bin')==label)
            row = shared.metric_row('val',label,group)
            gauc,_ = shared.build_gauc('val',group)
            row.update({'run_name':name,'pooling':condition['pooling'],'source':condition['source'],
                        'seed':condition['seed'],'gauc':gauc['gauc'],
                        'eligible_users':gauc['eligible_users'],'eligible_row_coverage':gauc['eligible_row_coverage']})
            rows.append(row)
        print(f'Passed validation identity/empty-history audit: {name}',flush=True)
    pl.DataFrame(rows).write_csv(study/'validation_history_length.csv')
    write_json(study/'validation_control_balance.json',control_balance())
    for path in sorted(study.glob('*')):
        if path.is_file() and path.suffix in ('.json','.csv'):
            evidence[str(path.relative_to(ROOT))] = sha256(path)
    write_json(study/'freeze_audit.json',{
        'passed':True,'freeze_manifest_sha256':sha256(freeze_path),'inputs_unchanged':True,
        'no_new_test_predictions':True,'selected_checkpoints_verified':len(freeze['checkpoints']),
        'empty_history_exact_baseline':empty_checks,'evidence_sha256':evidence,
        'audit_code_sha256':sha256(ROOT/'src/24_audit_sequence_freeze.py'),
        'history_diagnostics':'Validation only, computed after immutable selection; no further tuning',
        'scope':'Test F4 bytes are hashed for integrity only; no test model inference or scoring'})


if __name__=='__main__':main()
