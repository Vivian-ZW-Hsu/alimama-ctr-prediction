"""Validation-only calibration and fixed diagnostics of saved probabilities.

Run --stage fit completely before --stage apply. Neither stage loads a CTR model.
"""
import argparse
import json
import time

import joblib
import numpy as np
import polars as pl
import sklearn

from ctr_posthoc import (ROOT, STUDY, DATA, calibration, evaluator, records,
                         prediction_path, load_prediction, metrics, sha256, write_json)


def select_and_refit(ids, labels, probabilities):
    """This interface accepts validation data only; reuse the existing recipe."""
    select = calibration.deterministic_selection_mask(ids.astype(np.uint64), 5)
    fit = ~select
    platt = calibration.fit_platt(labels[fit], probabilities[fit])
    isotonic = calibration.fit_isotonic(labels[fit], probabilities[fit])
    candidates = {'raw': probabilities[select],
                  'platt': calibration.apply_platt(platt, probabilities[select]),
                  'isotonic': isotonic.predict(probabilities[select])}
    rows = [calibration.metric_row('validation_selection', name, labels[select], p, 20)
            for name, p in candidates.items()]
    method = pl.DataFrame(rows).filter(pl.col('method') != 'raw').sort('logloss').row(0, named=True)['method']
    final = (calibration.fit_platt if method == 'platt' else calibration.fit_isotonic)(labels, probabilities)
    return method, final, rows, int(fit.sum()), int(select.sum())


def fit_all():
    path = STUDY/'calibration_fit_manifest.json'
    output = ROOT/'artifacts/calibrators/sequence_posthoc_v1'
    if path.exists() or output.exists():
        raise FileExistsError('Calibration fitting already started; preserve existing products')
    output.mkdir(parents=True)
    context = pl.read_parquet(DATA/'val_groups.parquet')
    selected = []; selections = []; begin = time.perf_counter()
    for r in records():
        p = load_prediction(r, 'val', context)
        assert sha256(prediction_path(r, 'val')) == r['validation_prediction_sha256']
        method, final, rows, n_fit, n_select = select_and_refit(
            context['impression_id'].to_numpy(), context['clk'].to_numpy(), p)
        target = output/f'{r["run_name"]}.joblib'
        joblib.dump(final, target)
        selected.append({**r, 'method': method, 'calibrator_path': str(target.relative_to(ROOT)),
                         'calibrator_sha256': sha256(target), 'fit_rows': n_fit, 'selection_rows': n_select})
        selections.extend([{**row, 'condition': r['condition'], 'seed': r['seed']} for row in rows])
        print(f'Validation only: {r["condition"]} seed {r["seed"]}: {method}', flush=True)
    pl.DataFrame(selections).write_csv(STUDY/'calibration_selection.csv')
    write_json(path, {'complete': True, 'calibrators': selected, 'count': len(selected), 'zero_count': 1,
                     'test_used_for_fitting_or_selection': False, 'elapsed_seconds': time.perf_counter()-begin,
                     'numpy': np.__version__, 'sklearn': sklearn.__version__,
                     'selection_sha256': sha256(STUDY/'calibration_selection.csv'),
                     'code_sha256': {p: sha256(ROOT/'src'/p) for p in
                                    ('14_calibrate_predictions.py', '30_calibrate_diagnose_sequences.py')}})


def apply_all():
    fit_path = STUDY/'calibration_fit_manifest.json'
    fit = json.loads(fit_path.read_text())
    assert fit['complete'] and fit['count'] == 13 and fit['zero_count'] == 1
    output = ROOT/'artifacts/predictions/sequence_posthoc_v1'
    if output.exists() or (STUDY/'diagnostics_manifest.json').exists():
        raise FileExistsError('Analysis application already started; preserve existing products')
    output.mkdir(parents=True)
    contexts = {s: pl.read_parquet(DATA/f'{s}_groups.parquet') for s in ('val', 'test')}
    overall = []; groups = []; hashes = {}; begin = time.perf_counter()
    for r in fit['calibrators']:
        model_path = ROOT/r['calibrator_path']
        assert sha256(model_path) == r['calibrator_sha256']
        cal = joblib.load(model_path)
        target = output/r['run_name']; target.mkdir()
        for split, ctx in contexts.items():
            raw = load_prediction(r, split, ctx)
            calibrated = calibration.clipped(calibration.apply_platt(cal, raw) if r['method'] == 'platt' else cal.predict(raw))
            assert np.isfinite(calibrated).all() and ((calibrated > 0) & (calibrated < 1)).all()
            saved = ctx.select('impression_id', 'clk').with_columns(pl.Series('prediction', calibrated))
            predpath = target/f'{split}_predictions.parquet'
            saved.write_parquet(predpath, compression='zstd')
            np.testing.assert_array_equal(pl.read_parquet(predpath)['prediction'].to_numpy(), calibrated)
            hashes[str(predpath.relative_to(ROOT))] = sha256(predpath)
            for stage, p in (('raw', raw), ('calibrated', calibrated)):
                frame = ctx.with_columns(pl.Series('prediction', p))
                tags = {'condition': r['condition'], 'seed': r['seed'], 'stage': stage,
                        'method': r['method'], 'in_sample_refit': split == 'val' and stage == 'calibrated',
                        'independent_zero_seeds': False}
                overall.append({**metrics(frame, split, extra=True), **tags})
                # All conditions get before/after length and cold-start calibration diagnostics.
                group_columns = ['history_bin', 'cold_start_segment']
                if stage == 'raw' and r['condition'] in ('mean_matched', 'mean_mismatched', 'zero_sequence_same_pipeline'):
                    group_columns.append('category_hit')
                for column in group_columns:
                    for (value,), sub in frame.partition_by(column, as_dict=True).items():
                        row = metrics(sub, split, str(value))
                        row['ece'] = evaluator.weighted_ece(evaluator.build_reliability(split, sub, 10))
                        groups.append({**row, **tags, 'group': column, 'value': str(value)})
            print(f'Applied saved-probability calibration + shared evaluation: {r["condition"]} {r["seed"]} {split}', flush=True)
    pl.DataFrame(overall).write_csv(STUDY/'overall_before_after.csv')
    pl.DataFrame(groups).write_csv(STUDY/'group_metrics.csv')
    write_json(STUDY/'diagnostics_manifest.json', {'complete': True, 'calibrators': 13, 'zero_calibrators': 1,
        'fit_manifest_sha256_before_test_application': sha256(fit_path), 'elapsed_seconds': time.perf_counter()-begin,
        'prediction_sha256': hashes, 'probability_storage': 'float64, exact evaluated values; input raw files untouched',
        'full_val_calibrated_metrics': 'in-sample after full validation refit; not unbiased held-out estimates',
        'selection': 'No test-dependent choice or calibrator removal', 'ctr_training_or_inference': False,
        'code_sha256': {p: sha256(ROOT/'src'/p) for p in ('30_calibrate_diagnose_sequences.py', '10_evaluate_predictions.py', '14_calibrate_predictions.py')}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['fit', 'apply'], required=True)
    args = parser.parse_args()
    (fit_all if args.stage == 'fit' else apply_all)()
