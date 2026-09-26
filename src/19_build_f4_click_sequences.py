"""Build immutable daily F4 references; no model execution or test metrics."""
import argparse
import json
import resource
import time
from datetime import date
from pathlib import Path

import polars as pl

from f4_sequences import (ROOT, SEED, ASSUMPTION, sha256, write_json, source_fingerprints,
    positive_events, daily_histories, targets_for_day, attach_histories,
    assert_f123_identity, audit_references, raw_identity_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--smoke-certificate', type=Path)
    parser.add_argument('--max-length', type=int, default=20)
    args = parser.parse_args()
    if not args.run_name.replace('_', '').replace('-', '').isalnum():
        parser.error('Invalid run name')
    if args.max_length != 20:
        parser.error('This authorized build is fixed to L=20')
    started = time.perf_counter()
    inputs = source_fingerprints()
    config = {'version': 1, 'max_length': args.max_length, 'seed': SEED,
              'smoke': args.smoke_test, 'activity_threshold': 100,
              'inputs': inputs, 'implementation_sha256': sha256(Path(__file__)),
              'library_sha256': sha256(ROOT / 'src/f4_sequences.py')}
    output = ROOT / 'data/processed' / args.run_name
    result = ROOT / 'results' / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    config_path = result / 'build_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, 'Resume config/input mismatch'
        if (result / 'manifest.json').exists():
            saved = json.loads((result / 'manifest.json').read_text())
            for name, digest in saved['output_sha256'].items():
                assert sha256(ROOT / name) == digest, name
            print('Verified complete immutable build; nothing changed.', flush=True)
            return
    else:
        write_json(config_path, config)
    if not args.smoke_test:
        if not args.smoke_certificate:
            parser.error('Full build requires a passed reproducible smoke certificate')
        cert = json.loads(args.smoke_certificate.read_text())
        assert cert['passed'] and cert['reproducible']
        for key in ('inputs', 'max_length', 'seed', 'activity_threshold', 'implementation_sha256', 'library_sha256'):
            assert cert['config'][key] == config[key], key
    print('Building sorted positive event store (train/val only)...', flush=True)
    events = positive_events()
    assert events.height == 1199566
    assert events['impression_id'].n_unique() == events.height
    dates = pl.from_epoch(events['time_stamp'], time_unit='s').dt.replace_time_zone('UTC').dt.convert_time_zone('Asia/Shanghai').dt.date()
    assert dates.equals(events['event_date_bj'])
    path = output / 'events.parquet'
    if not path.exists():
        events.write_parquet(path.with_suffix('.partial'), compression='zstd')
        path.with_suffix('.partial').replace(path)
    else:
        assert events.equals(pl.read_parquet(path))
    activity = pl.concat([pl.scan_parquet(ROOT / f'data/processed/base/{split}.parquet')
                         .group_by('user_raw', 'event_date_bj').agg(pl.len().alias('impressions'))
                         for split in ('train', 'val')]).collect()
    reports = []
    outputs = [path]
    for n in range(6, 14):
        day = date(2017, 5, n)
        split = 'train' if n < 12 else 'val' if n == 12 else 'test'
        refs_path = output / f'{split}_{day}.parquet'
        hist_path = output / f'history_{day}.parquet'
        audit_path = result / f'audit_{day}.json'
        if audit_path.exists():
            report = json.loads(audit_path.read_text())
            assert sha256(refs_path) == report['refs_sha256']
            assert sha256(hist_path) == report['history_sha256']
        else:
            h = daily_histories(events, activity, day, args.max_length, SEED)
            assert h['fallback'].sum() == 0, 'Formal/smoke build prohibits fallback'
            t = targets_for_day(split, day, args.smoke_test)
            refs = attach_histories(t, h)
            assert_f123_identity(refs, split, day, args.smoke_test)
            report = audit_references(refs, events, day, args.max_length)
            report.update({'split': split, 'f123_id_label_parity': True,
                           'donor_pool_users': h.height, 'donor_pool_fallback_count': int(h['fallback'].sum()),
                           'reference_schema': {k: str(v) for k, v in refs.schema.items()}})
            for frame, destination in [(h, hist_path), (refs, refs_path)]:
                tmp = destination.with_suffix('.partial')
                frame.write_parquet(tmp, compression='zstd', row_group_size=100000)
                tmp.replace(destination)
            report.update({'refs_sha256': sha256(refs_path), 'history_sha256': sha256(hist_path)})
            write_json(audit_path, report)
        reports.append(report)
        outputs.extend([refs_path, hist_path])
        print(f'{day} {split} rows={report["rows"]:,}; ID/label/cutoff/identity/length checks passed', flush=True)
    print('Checking sampled original CSV row identities and raw ad categories...', flush=True)
    raw = raw_identity_audit(events)
    write_json(result / 'raw_identity_audit.json', raw)
    if not args.smoke_test:
        assert sum(r['rows'] for r in reports) == 26557961
    for name, before in inputs.items():
        p = ROOT / name
        assert (p.stat().st_size, p.stat().st_mtime_ns) == (before['bytes'], before['mtime_ns'])
    manifest = {'run_name': args.run_name, 'complete': True, 'config': config,
        'source': 'positive ad exposures; category tokens', 'availability_assumption': ASSUMPTION,
        'cutoff': 'source Beijing date strictly less than target Beijing date; no same-day labels',
        'vocabulary': 'existing category vocabulary verified equal to training categories only',
        'reserved_indices': {'PAD': 0, 'UNKNOWN': 1}, 'smoke_selector': 'impression_id % 1000 == 0 on every date' if args.smoke_test else None,
        'test_scope': 'data construction and ID/label integrity only; no model/test metrics',
        'total_rows': sum(r['rows'] for r in reports), 'days': reports,
        'raw_identity': raw, 'output_directory': str(output.relative_to(ROOT)),
        'output_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
        'output_bytes': sum(p.stat().st_size for p in outputs),
        'elapsed_seconds': time.perf_counter()-started,
        'peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json(result / 'manifest.json', manifest)
    print(f'Complete: {result / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
