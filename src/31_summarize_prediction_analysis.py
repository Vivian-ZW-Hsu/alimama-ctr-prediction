"""Recompute compact summaries from immutable per-predictor diagnostic tables."""
import json
import numpy as np
import polars as pl
from ctr_posthoc import ROOT, STUDY, sha256, write_json

METRICS = ('auc', 'logloss', 'pr_auc', 'mean_prediction', 'copc', 'ece')
ZERO = 'zero_sequence_same_pipeline'


def summarize(frame, keys, metrics):
    return frame.group_by(keys).agg(
        pl.len().alias('predictors'), pl.col('rows').first(), pl.col('clicks').first(),
        pl.col('actual_ctr').first(),
        *[expr for m in metrics for expr in
          (pl.col(m).mean().alias(f'{m}_mean'), pl.col(m).std(ddof=1).alias(f'{m}_std'))]
    ).sort(keys)


def group_differences(frame):
    rows = []
    pairs = [('mean_matched', ZERO), ('mean_matched', 'mean_mismatched'),
             ('mean_mismatched', ZERO), ('attention_matched', 'mean_matched')]
    raw = frame.filter(pl.col('stage') == 'raw')
    for left, right in pairs:
        a = raw.filter(pl.col('condition') == left)
        if left == 'attention_matched':
            a = a.filter(pl.col('group') == 'history_bin')
        for r in a.iter_rows(named=True):
            b = raw.filter((pl.col('condition') == right) & (pl.col('split') == r['split']) &
                           (pl.col('group') == r['group']) & (pl.col('value') == r['value']) &
                           (pl.col('seed') == (42 if right == ZERO else r['seed'])))
            assert b.height == 1
            ref = b.row(0, named=True)
            assert r['rows'] == ref['rows'] and r['clicks'] == ref['clicks']
            rows.append({**{k: r[k] for k in ('split', 'group', 'value', 'seed', 'rows', 'clicks', 'actual_ctr')},
                         'left': left, 'right': right,
                         **{m: r[m]-ref[m] for m in METRICS}})
    return pl.DataFrame(rows)


def calibration_differences(frame, grouped=False):
    keys = ['split', 'condition', 'seed'] + (['group', 'value'] if grouped else [])
    m = list(METRICS) + ([] if grouped else ['gauc'])
    left = frame.filter(pl.col('stage') == 'calibrated')
    right = frame.filter(pl.col('stage') == 'raw').select(keys + m)
    joined = left.join(right, on=keys, suffix='_raw', validate='1:1')
    assert joined.height == left.height
    return joined.select(keys + ['rows', 'clicks', 'actual_ctr'] +
                         [(pl.col(x)-pl.col(x+'_raw')).alias(x) for x in m])


def main():
    output = STUDY/'summary_manifest.json'
    if output.exists():
        raise FileExistsError('Summary complete; verify rather than overwrite')
    overall = pl.read_csv(STUDY/'overall_before_after.csv')
    groups = pl.read_csv(STUDY/'group_metrics.csv')
    om = list(METRICS) + ['gauc']
    products = {
        'overall_summary.csv': summarize(overall, ['split', 'condition', 'stage'], om),
        'group_summary.csv': summarize(groups, ['split', 'condition', 'stage', 'group', 'value'], METRICS),
        'group_paired_differences.csv': group_differences(groups),
        'calibration_overall_differences.csv': calibration_differences(overall),
        'calibration_group_differences.csv': calibration_differences(groups, grouped=True)}
    products['group_difference_summary.csv'] = summarize(products['group_paired_differences.csv'],
        ['split', 'left', 'right', 'group', 'value'], METRICS)
    products['calibration_overall_difference_summary.csv'] = summarize(products['calibration_overall_differences.csv'],
        ['split', 'condition'], om)
    products['calibration_group_difference_summary.csv'] = summarize(products['calibration_group_differences.csv'],
        ['split', 'condition', 'group', 'value'], METRICS)
    # Reconcile original raw metrics, without rerunning inference or fitting anything.
    formal = pl.read_csv(ROOT/'results/sequence_test_v2/test_metrics.csv')
    for r in overall.filter((pl.col('split') == 'test') & (pl.col('stage') == 'raw')).iter_rows(named=True):
        ref = formal.filter((pl.col('condition') == r['condition']) & (pl.col('seed') == r['seed'])).row(0, named=True)
        for metric in ('auc', 'logloss', 'pr_auc', 'gauc', 'copc', 'ece'):
            np.testing.assert_allclose(r[metric], ref[metric], atol=1e-12, rtol=1e-10)
    for name, frame in products.items():
        path = STUDY/name
        if path.exists():
            raise FileExistsError(path)
        frame.write_csv(path)
    write_json(output, {'complete': True, 'raw_test_metrics_match_original': True,
        'zero_distinct_predictors': 1, 'zero_sample_std': 'undefined/null, not three independent seeds',
        'seed_summary': 'arithmetic mean of metrics, sample std ddof=1; not metrics of probability ensembles',
        'input_sha256': {p: sha256(STUDY/p) for p in ('overall_before_after.csv', 'group_metrics.csv')},
        'output_sha256': {p: sha256(STUDY/p) for p in products},
        'code_sha256': sha256(ROOT/'src/31_summarize_prediction_analysis.py')})


if __name__ == '__main__':
    main()
