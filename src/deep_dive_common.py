"""Descriptive analysis helpers; never fits or loads a CTR model."""
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
from ctr_posthoc import ROOT, DATA as POSTHOC_DATA, evaluator, sha256, write_json

DATA = ROOT/'data/processed/deep_dive_v1'
PLAN = ROOT/'docs/DEEP_DIVE_RESEARCH_PLAN.md'
TABULAR = {'LR': 'lr_f1_f2_f3', 'LightGBM': 'lightgbm_f1_f2_f3',
           'WideDeep': 'wide_deep_f1_f2_f3', 'DeepFM': 'deepfm_f1_f2_f3'}
SEEDS = (42, 2026, 3407)


def loss(y, p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def predict_path(name, split, seed=42, calibrated=False):
    if name in TABULAR:
        return ROOT/'artifacts/predictions'/TABULAR[name]/f'{split}_predictions.parquet'
    run = (f'seq_zero_same_pipeline_l20_s{seed}_v2' if name == 'zero' else
           f'seq_{name}_l20_lr500_s{seed}_v1')
    if calibrated:
        return ROOT/'artifacts/predictions/sequence_posthoc_v1'/run/f'{split}_predictions.parquet'
    return ROOT/'artifacts/predictions'/(run if split == 'val' else run+'_test_v2')/f'{split}_predictions.parquet'


def read_p(name, split, ctx, seed=42, calibrated=False):
    frame = pl.read_parquet(predict_path(name, split, seed, calibrated)).sort('impression_id')
    for c in ('impression_id', 'clk'):
        np.testing.assert_array_equal(frame[c].to_numpy(), ctx[c].to_numpy())
    p = frame['prediction'].to_numpy().astype(float)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    return p


def count_band(v, cuts, labels):
    return np.asarray(labels)[np.searchsorted(cuts, v, side='right')]


def evidence_band(v):
    return count_band(v, [1, 12, 100], ['0', '1-11', '12-99', '100+'])


def top_budget(p, ids, fraction):
    """Exact ceil quota, deterministic ascending impression ID for tied scores."""
    order = np.lexsort((ids, -p))
    out = np.zeros(len(p), dtype=bool)
    out[order[:int(np.ceil(len(p)*fraction))]] = True
    return out


def grouped_metrics(ctx, p, column, split, name, seed):
    frame = ctx.select('clk', column).with_columns(pl.Series('prediction', p))
    rows = []
    for (value,), group in frame.partition_by(column, as_dict=True).items():
        rows.append({**evaluator.metric_row(split, str(value), group),
                     'model': name, 'seed': seed, 'group': column, 'value': str(value)})
    return rows


class Stage:
    def __init__(self, name, minutes):
        self.out = ROOT/'results'/name
        if self.out.exists():
            raise FileExistsError(f'Preserve completed/partial stage: {self.out}')
        self.out.mkdir(parents=True)
        self.begin = time.perf_counter(); self.cap = minutes*60; self.inputs = {}
        self.bind(PLAN)
        report = ROOT/'docs/DIN_F4_TEST_RESULTS.md'
        assert sha256(report) == 'd508e9be7747ef1c0dc87ac4787aa7cae97e7c1f7311bdbebf3ea4b9ac29c6b3'
        self.bind(report)

    def bind(self, path):
        self.inputs[str(Path(path).relative_to(ROOT))] = sha256(path)

    def save(self, name, rows):
        assert time.perf_counter()-self.begin < self.cap, 'Stage budget exceeded; pause'
        frame = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
        frame.write_csv(self.out/name)

    def finish(self, **extra):
        for path, digest in self.inputs.items():
            assert sha256(ROOT/path) == digest, path
        write_json(self.out/'manifest.json', {'complete': True, 'elapsed_seconds': time.perf_counter()-self.begin,
            'cap_seconds': self.cap, 'frozen_evidence_commit': 'badecf4', 'inputs_sha256': self.inputs,
            'output_sha256': {p.name: sha256(p) for p in self.out.iterdir() if p.is_file()},
            'ctr_training': False, 'test_inference': False, 'bootstrap_repeated': False,
            'calibration_fitted': False, **extra})
