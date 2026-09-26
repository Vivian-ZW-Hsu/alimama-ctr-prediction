"""Read-only independent reconciliation; writes only a new verification report."""
import json
import math
import statistics
import numpy as np
import polars as pl
from ctr_posthoc import ROOT, STUDY, DATA, sha256, write_json


def main():
    output = STUDY/'verification.json'
    if output.exists():
        raise FileExistsError('Verification report exists')
    inputs = json.loads((STUDY/'input_manifest.json').read_text())
    for field in ('input_sha256', 'derived_sha256'):
        for path, digest in inputs[field].items():
            assert sha256(ROOT/path) == digest, path
    boot = json.loads((STUDY/'bootstrap_manifest.json').read_text())
    assert sha256(DATA/'bootstrap_full.bin') == boot['raw_results_sha256']
    # Independent full-row category audit: scan each retained position, no 2D gather.
    events = pl.read_parquet(ROOT/'data/processed/f4_click_daily_v1/events.parquet', columns=['cate_id_raw'])['cate_id_raw'].fill_null(-1).to_numpy()
    for split, day in [('val', '2017-05-12'), ('test', '2017-05-13')]:
        ctx = pl.read_parquet(DATA/f'{split}_groups.parquet')
        base = pl.read_parquet(ROOT/f'data/processed/base/{split}.parquet', columns=['impression_id', 'clk', 'user_raw', 'adgroup_id_raw', 'cate_id_raw']).sort('impression_id')
        assert ctx.select(base.columns).equals(base)
        refs = pl.read_parquet(ROOT/f'data/processed/f4_click_daily_v1/{split}_{day}.parquet').sort('impression_id')
        for col in ('impression_id', 'clk', 'length'):
            np.testing.assert_array_equal(ctx[col].to_numpy(), refs[col].to_numpy())
        hit = np.zeros(ctx.height, dtype=bool)
        candidate = ctx['cate_id_raw'].fill_null(-1).to_numpy()
        start = refs['start'].to_numpy(); length = refs['length'].to_numpy()
        for position in range(20):
            ix = np.flatnonzero(length > position)
            cat = events[start[ix]+position]
            hit[ix] |= (candidate[ix] == cat) & (cat >= 0) & (candidate[ix] >= 0)
        np.testing.assert_array_equal(hit, ctx['category_hit'].to_numpy())
        for col, flag in [('user_raw', 'known_user'), ('adgroup_id_raw', 'known_ad')]:
            known = pl.read_parquet(ROOT/f'artifacts/encoders/mappings/{col}_mapping.parquet', columns=[col])[col].to_numpy()
            np.testing.assert_array_equal(np.isin(ctx[col].to_numpy(), known), ctx[flag].to_numpy())
    intervals = pl.read_csv(STUDY/'bootstrap_intervals.csv')
    draws = pl.read_csv(STUDY/'bootstrap_deltas.csv')
    assert draws.height == 1000 and intervals.height == 20
    for r in intervals.iter_rows(named=True):
        values = draws[f'comparison_{r["id"]}_{r["metric"]}'].to_list()
        assert all(math.isfinite(v) for v in values)
        ordered = sorted(values)
        def quantile(q):
            x = q*(len(ordered)-1); a = int(x)
            return ordered[a] + (ordered[a+1]-ordered[a])*(x-a)
        np.testing.assert_allclose([r['ci_low'], r['ci_high']], [quantile(.025), quantile(.975)], atol=1e-15)
        p = (1+sum(abs(v-r['effect']) >= abs(r['effect']) for v in values))/1001
        assert abs(p-r['p_centered_two_sided']) < 1e-15
    # Independent Holm implementation, including stricter four-hypothesis family.
    families = [intervals.filter((pl.col('id') <= 2) & (pl.col('metric') == m)) for m in ('auc', 'logloss', 'pr_auc', 'gauc')]
    families.append(intervals.filter((pl.col('id') <= 2) & pl.col('metric').is_in(['auc', 'logloss'])))
    for i, family in enumerate(families):
        previous = 0.
        for rank, row in enumerate(family.sort('p_centered_two_sided').iter_rows(named=True)):
            previous = min(1., max(previous, (family.height-rank)*row['p_centered_two_sided']))
            key = 'holm_joint_primary_auc_logloss' if i == 4 else 'holm_two_primary_per_metric'
            assert abs(row[key]-previous) < 1e-15
    summary = pl.read_csv(STUDY/'overall_summary.csv')
    detailed = pl.read_csv(STUDY/'overall_before_after.csv')
    for r in summary.iter_rows(named=True):
        sub = detailed.filter((pl.col('split') == r['split']) & (pl.col('stage') == r['stage']) & (pl.col('condition') == r['condition']))
        expected_n = 1 if r['condition'] == 'zero_sequence_same_pipeline' else 3
        assert sub.height == r['predictors'] == expected_n
        for m in ('auc', 'logloss', 'pr_auc', 'gauc', 'copc', 'ece', 'mean_prediction'):
            values = sub[m].to_list()
            np.testing.assert_allclose(r[m+'_mean'], statistics.mean(values), atol=1e-14)
            if expected_n == 1:
                assert r[m+'_std'] is None
            else:
                np.testing.assert_allclose(r[m+'_std'], statistics.stdev(values), atol=1e-14)
    grouped = pl.read_csv(STUDY/'group_metrics.csv')
    for keys, frame in grouped.partition_by(['split', 'condition', 'seed', 'stage', 'group'], as_dict=True).items():
        split, condition, seed, stage, group = keys
        allrow = detailed.filter((pl.col('split') == split) & (pl.col('condition') == condition) & (pl.col('seed') == seed) & (pl.col('stage') == stage)).row(0, named=True)
        n = frame['rows'].to_numpy()
        assert int(n.sum()) == allrow['rows'] and frame['clicks'].sum() == allrow['clicks']
        for metric in ('logloss', 'mean_prediction', 'actual_ctr'):
            np.testing.assert_allclose(np.average(frame[metric].to_numpy(), weights=n), allrow[metric], atol=1e-12)
    fit = json.loads((STUDY/'calibration_fit_manifest.json').read_text())
    diagnostics = json.loads((STUDY/'diagnostics_manifest.json').read_text())
    assert sha256(STUDY/'calibration_fit_manifest.json') == diagnostics['fit_manifest_sha256_before_test_application']
    assert len(fit['calibrators']) == 13 and sum(r['zero_sequence'] for r in fit['calibrators']) == 1
    for r in fit['calibrators']:
        assert sha256(ROOT/r['calibrator_path']) == r['calibrator_sha256']
    for path, digest in diagnostics['prediction_sha256'].items():
        assert sha256(ROOT/path) == digest
    # Bind report outputs, including independently recalculated summaries.
    report = {'passed': True, 'original_inputs_unchanged': True,
              'full_row_identity_raw_category_and_train_vocab_checks': ['val', 'test'],
              'valid_user_cluster_replicates': 1000, 'interval_p_and_holm_recomputed': True,
              'seed_means_and_sample_stds_recomputed': True, 'group_additive_metrics_reconcile': True,
              'all_13_selected_calibrators_retained': True, 'zero_distinct_calibrators': 1,
              'new_ctr_training_or_inference': False,
              'result_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in STUDY.glob('*') if p.is_file()},
              'code_sha256': sha256(ROOT/'src/32_verify_prediction_analysis.py')}
    write_json(output, report)
    print('PASS: immutable inputs, complete groups, 1000 replicates, Holm, seed summaries and all calibrators', flush=True)


if __name__ == '__main__':
    main()
