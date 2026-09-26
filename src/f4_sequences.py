"""Daily positive-feedback histories, deterministic donors, and independent audits."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260926
ASSUMPTION = "Exposure timestamp proxies the positive record time; feedback available next Beijing day; no independent click timestamp."


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def midnight(day):
    return int(datetime.combine(day, datetime.min.time(), ZoneInfo('Asia/Shanghai')).timestamp())


def source_fingerprints():
    paths = [ROOT / f'data/processed/{family}/{split}.parquet'
             for family in ('base', 'model_input_f3') for split in ('train', 'val', 'test')]
    paths += [ROOT / 'data/raw/raw_sample.csv', ROOT / 'data/raw/ad_feature.csv',
              ROOT / 'artifacts/encoders/mappings/cate_id_raw_mapping.parquet']
    return {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size,
            'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha256(p)} for p in paths}


def positive_events():
    columns = ['impression_id', 'user_raw', 'time_stamp', 'event_date_bj',
               'adgroup_id_raw', 'cate_id_raw']
    frame = pl.concat([pl.scan_parquet(ROOT / f'data/processed/base/{s}.parquet')
                       .filter(pl.col('clk') == 1).select(columns)
                       for s in ('train', 'val')]).collect()
    mapping = pl.read_parquet(ROOT / 'artifacts/encoders/mappings/cate_id_raw_mapping.parquet')
    train_vocab = pl.scan_parquet(ROOT / 'data/processed/base/train.parquet').select(
        pl.col('cate_id_raw').drop_nulls().unique()).collect()
    assert set(mapping['cate_id_raw']) == set(train_vocab['cate_id_raw'])
    frame = frame.join(mapping, on='cate_id_raw', how='left', validate='m:1').with_columns(
        pl.col('cate_idx').fill_null(1).cast(pl.UInt32))
    return frame.sort('user_raw', 'time_stamp', 'impression_id').with_row_index('event_index')


def donor_permutation(history, seed):
    """Coarsen activity only if necessary; never silently change a length."""
    history = history.sort('user_raw')
    donor = np.arange(history.height, dtype=np.uint32)
    fallback = np.zeros(history.height, dtype=np.uint8)
    length = history['length'].to_numpy()
    bucket = history['activity_bucket'].to_numpy()
    rng = np.random.default_rng(seed)
    for n in np.unique(length):
        all_indices = np.flatnonzero(length == n)
        if len(all_indices) < 2:
            raise ValueError(f'No different-user donor for length {n}')
        groups = [all_indices[bucket[all_indices] == b] for b in np.unique(bucket[all_indices])]
        if any(len(group) < 2 for group in groups):
            groups = [all_indices]
            fallback[all_indices] = 1
        for group in groups:
            order = rng.permutation(group)
            donor[order] = np.roll(order, int(rng.integers(1, len(order))))
    assert not np.any(donor == np.arange(history.height))
    assert len(np.unique(donor)) == history.height
    return history.with_columns(
        pl.Series('donor_start', history['start'].to_numpy()[donor], dtype=pl.UInt32),
        pl.Series('donor_user_raw', history['user_raw'].to_numpy()[donor], dtype=pl.Int64),
        pl.Series('donor_length', length[donor], dtype=pl.UInt8),
        pl.Series('fallback', fallback, dtype=pl.UInt8))


def daily_histories(events, activity, day, max_length, seed):
    past = events.filter(pl.col('event_date_bj') < day)
    schema = {'user_raw': pl.Int64, 'raw_length': pl.UInt32, 'end': pl.UInt32,
              'length': pl.UInt8, 'start': pl.UInt32, 'activity_bucket': pl.UInt8,
              'donor_start': pl.UInt32, 'donor_user_raw': pl.Int64,
              'donor_length': pl.UInt8, 'fallback': pl.UInt8}
    if past.is_empty():
        return pl.DataFrame(schema=schema)
    assert past['time_stamp'].max() < midnight(day)
    h = past.group_by('user_raw').agg(pl.len().cast(pl.UInt32).alias('raw_length'),
          (pl.col('event_index').max() + 1).cast(pl.UInt32).alias('end')).with_columns(
              pl.col('raw_length').clip(upper_bound=max_length).cast(pl.UInt8).alias('length'))
    a = activity.filter(pl.col('event_date_bj') < day).group_by('user_raw').agg(
        pl.col('impressions').sum().alias('past_impressions'))
    h = h.join(a, on='user_raw', how='left', validate='1:1').with_columns(
        (pl.col('end') - pl.col('length')).cast(pl.UInt32).alias('start'),
        (pl.col('past_impressions') >= 100).cast(pl.UInt8).alias('activity_bucket')).drop('past_impressions')
    return donor_permutation(h, seed + day.toordinal())


def targets_for_day(split, day, smoke):
    x = pl.scan_parquet(ROOT / f'data/processed/base/{split}.parquet').filter(
        pl.col('event_date_bj') == day).select('impression_id', 'user_raw', 'event_date_bj', 'clk')
    if smoke:
        x = x.filter(pl.col('impression_id') % 1000 == 0)
    return x.collect().sort('impression_id')


def attach_histories(targets, history):
    f = targets.join(history, on='user_raw', how='left', validate='m:1').with_columns(
        [pl.col(c).fill_null(0) for c in ('start', 'donor_start', 'length', 'donor_length', 'raw_length', 'fallback')]
        + [pl.col('donor_user_raw').fill_null(-1)])
    return f.select('impression_id', pl.col('clk').cast(pl.UInt8), 'user_raw', 'event_date_bj',
        pl.col('start').cast(pl.UInt32), pl.col('donor_start').cast(pl.UInt32),
        pl.col('length').cast(pl.UInt8), pl.col('donor_length').cast(pl.UInt8),
        pl.col('raw_length').cast(pl.UInt32), 'donor_user_raw', pl.col('fallback').cast(pl.UInt8)).sort('impression_id')


def assert_f123_identity(refs, split, day, smoke):
    expected = pl.scan_parquet(ROOT / f'data/processed/model_input_f3/{split}.parquet').filter(
        pl.col('event_date_bj') == day).select('impression_id', 'clk')
    if smoke:
        expected = expected.filter(pl.col('impression_id') % 1000 == 0)
    expected = expected.collect().sort('impression_id')
    assert refs.height == expected.height
    assert refs['impression_id'].n_unique() == refs.height
    np.testing.assert_array_equal(refs['impression_id'].to_numpy(), expected['impression_id'].to_numpy())
    np.testing.assert_array_equal(refs['clk'].to_numpy(), expected['clk'].to_numpy())


def audit_references(refs, events, day, max_length):
    lengths = refs['length'].to_numpy()
    starts = refs['start'].to_numpy()
    donor_starts = refs['donor_start'].to_numpy()
    positive = lengths > 0
    user = events['user_raw'].to_numpy()
    ts = events['time_stamp'].to_numpy()
    cats = events['cate_idx'].to_numpy()
    counts = {'self_match_count': int(np.sum(positive & (refs['user_raw'].to_numpy() == refs['donor_user_raw'].to_numpy()))),
              'fallback_count': int(refs['fallback'].sum()),
              'length_mismatch_count': int(np.sum(lengths != refs['donor_length'].to_numpy())),
              'future_event_count': 0, 'identity_mismatch_count': 0, 'order_violation_count': 0}
    assert np.all(lengths <= max_length)
    assert np.all(lengths == np.minimum(refs['raw_length'].to_numpy(), max_length))
    assert np.all(starts[~positive] == 0) and np.all(donor_starts[~positive] == 0)
    token_stats = {}
    # Bounded <=64K x L gathers; never allocate N x L.
    for name, start, owners in [('matched', starts, refs['user_raw'].to_numpy()),
                                 ('mismatched', donor_starts, refs['donor_user_raw'].to_numpy())]:
        unknown = tokens = 0
        for offset in range(0, refs.height, 65536):
            n = lengths[offset:offset+65536]
            mask = np.arange(max_length)[None, :] < n[:, None]
            index = start[offset:offset+65536, None] + np.arange(max_length)[None, :]
            index = np.where(mask, index, 0)
            assert not np.any(index >= events.height)
            counts['identity_mismatch_count'] += int(np.sum(mask & (user[index] != owners[offset:offset+65536, None])))
            counts['future_event_count'] += int(np.sum(mask & (ts[index] >= midnight(day))))
            counts['order_violation_count'] += int(np.sum(mask[:, 1:] & (ts[index[:, 1:]] < ts[index[:, :-1]])))
            unknown += int(np.sum(mask & (cats[index] == 1)))
            tokens += int(mask.sum())
            assert not np.any(mask & (cats[index] == 0))
        token_stats[name] = {'tokens': tokens, 'unknown_tokens': unknown,
                             'padding_slots': refs.height * max_length - tokens}
    assert all(v == 0 for v in counts.values()), counts
    assert np.array_equal(np.bincount(lengths, minlength=max_length+1),
                          np.bincount(refs['donor_length'].to_numpy(), minlength=max_length+1))
    return {'date': str(day), 'rows': refs.height, **counts, 'token_stats': token_stats,
            'empty_rows': int(np.sum(lengths == 0)),
            'truncated_rows': int(np.sum(refs['raw_length'].to_numpy() > max_length)),
            'length_histogram': np.bincount(lengths, minlength=max_length+1).tolist()}


def raw_identity_audit(events, seed=SEED, samples=256):
    rng = np.random.default_rng(seed)
    picked = events[rng.choice(events.height, min(samples, events.height), replace=False)].sort('impression_id')
    expected = {int(r['impression_id']): r for r in picked.iter_rows(named=True)}
    ads = pl.read_csv(ROOT / 'data/raw/ad_feature.csv', null_values=['', 'NULL', 'null'],
                     schema_overrides={'brand': pl.Float64}).select('adgroup_id', 'cate_id')
    categories = dict(ads.iter_rows())
    count = 0
    with (ROOT / 'data/raw/raw_sample.csv').open() as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i in expected:
                e = expected[i]
                assert int(row['clk']) == 1
                assert (int(row['user']), int(row['adgroup_id']), int(row['time_stamp'])) == (
                    e['user_raw'], e['adgroup_id_raw'], e['time_stamp'])
                assert categories[e['adgroup_id_raw']] == e['cate_id_raw']
                count += 1
            if count == len(expected):
                break
    assert count == len(expected)
    return {'seed': seed, 'sampled_events': count, 'mismatches': 0,
            'source_impression_ids': sorted(expected)}
