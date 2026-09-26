"""Independent persisted-artifact audit and reproducible-smoke certification."""
import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from f4_sequences import ROOT, sha256, write_json, assert_f123_identity, audit_references


def audit(name):
    path = ROOT / 'results' / name / 'manifest.json'
    manifest = json.loads(path.read_text())
    assert manifest['complete']
    output = ROOT / manifest['output_directory']
    for file, digest in manifest['output_sha256'].items():
        assert sha256(ROOT / file) == digest, file
    events = pl.read_parquet(output / 'events.parquet')
    for previous in manifest['days']:
        day = date.fromisoformat(previous['date'])
        refs = pl.read_parquet(output / f'{previous["split"]}_{day}.parquet')
        assert_f123_identity(refs, previous['split'], day, manifest['config']['smoke'])
        observed = audit_references(refs, events, day, manifest['config']['max_length'])
        for key, value in observed.items():
            assert value == previous[key], key
    assert manifest['raw_identity']['mismatches'] == 0
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--compare-smoke')
    p.add_argument('--certificate', type=Path)
    args = p.parse_args()
    first = audit(args.run_name)
    if args.compare_smoke:
        second = audit(args.compare_smoke)
        assert first['config']['smoke'] and second['config']['smoke']
        assert first['config'] == second['config']
        one = {Path(k).name: v for k, v in first['output_sha256'].items()}
        two = {Path(k).name: v for k, v in second['output_sha256'].items()}
        assert one == two, 'Smoke Parquet files must be byte-reproducible'
        assert args.certificate and not args.certificate.exists()
        write_json(args.certificate, {'passed': True, 'reproducible': True,
            'runs': [args.run_name, args.compare_smoke], 'config': first['config'],
            'artifact_sha256': one, 'checked_dates': 8,
            'raw_identity_samples_per_run': first['raw_identity']['sampled_events']})
    print('Independent F4 integrity audit passed.', flush=True)


if __name__ == '__main__':
    main()
