"""Read-only audit of saved F5 donor assignments; never rebuilds histories.

User statistics cover ALL nonempty history users, not only users exposed that day.
Exposure statistics count targets assigned a globally reused donor, including its
first use. Reuse excess counts only assignments beyond one per distinct donor.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import polars as pl
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/f5_behavior_v1'
DAYS = [f'2017-05-{n:02}' for n in range(6, 14)]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def assignment_statistics(history, refs):
    """Pure fixture-testable function; refs need only user_raw and length."""
    assert history['user_raw'].n_unique() == history.height
    assert history['length'].min() > 0
    degree = history.group_by('donor_user_raw').len(name='indegree')
    assert degree.join(history.select(pl.col('user_raw').alias('donor_user_raw')),
                       on='donor_user_raw', how='anti').height == 0
    enriched = history.join(degree, on='donor_user_raw')
    repeated = enriched.filter(pl.col('indegree') > 1).select('user_raw')
    affected = refs.filter(pl.col('length') > 0).join(repeated, on='user_raw', how='semi')
    n, distinct = history.height, degree.height
    unused = n - distinct
    summary = dict(nonempty_target_users=n, assignments=n, distinct_donors=distinct,
                   reuse_excess=n-distinct, reuse_excess_rate=(n-distinct)/n,
                   unique_donor_rate=distinct/n, reused_donors=degree.filter(pl.col('indegree')>1).height,
                   max_indegree=int(degree['indegree'].max()), eligible_donors=n,
                   unused_eligible_donors=unused, unused_eligible_rate=unused/n,
                   targets_with_reused_donor=repeated.height,
                   exposed_targets_with_reused_donor=affected['user_raw'].n_unique(),
                   affected_impressions=affected.height, day_impressions=refs.height,
                   affected_impression_share=affected.height/refs.height,
                   self_match_count=history.filter(pl.col('user_raw')==pl.col('donor_user_raw')).height,
                   length_mismatch_count=history.filter(pl.col('length')!=pl.col('donor_length')).height)
    dist = degree.group_by('indegree').len(name='donors').sort('indegree').to_dicts()
    dist.insert(0, {'indegree': 0, 'donors': unused})
    partitions = []
    for axis in ['fallback', 'length', 'activity_bucket', 'recency_bucket']:
        for (value,), group in enriched.group_by(axis):
            local = group.group_by('donor_user_raw').len(name='indegree')
            partitions.append(dict(axis=axis, value=int(value), assignments=group.height,
                distinct_donors=local.height, reuse_excess=group.height-local.height,
                reuse_excess_rate=(group.height-local.height)/group.height,
                reused_donors=local.filter(pl.col('indegree')>1).height,
                max_indegree=int(local['indegree'].max()),
                targets_with_reused_donor_within_partition=int(local.filter(pl.col('indegree')>1)['indegree'].sum()),
                targets_with_reused_donor_globally=group.filter(pl.col('indegree')>1).height))
    return summary, dist, partitions


def stream_endpoints(histories):
    """One sequential scan, bounded batches; no N-impression x L expansion.

Prove global user blocks contiguous and time nondecreasing. Combined with matching
window endpoints, this proves ownership/order/future conditions for every token.
"""
    required = np.unique(np.concatenate([
        np.concatenate([h['start'].to_numpy().astype('int64'), h['end'].to_numpy().astype('int64')-1,
                        h['donor_start'].to_numpy().astype('int64'),
                        h['donor_start'].to_numpy().astype('int64')+h['length'].to_numpy()-1])
        for h in histories]))
    owners = np.empty(len(required), dtype=np.int64)
    times = np.empty(len(required), dtype=np.int64)
    seen, previous_user, previous_time = set(), None, None
    offset = descending = unknown = pad = bad_index = 0
    for batch in pq.ParquetFile(DATA/'events.parquet').iter_batches(batch_size=1_000_000,
            columns=['event_index', 'user_raw', 'time_stamp', 'cate_idx']):
        ix, user, ts, cat = [c.to_numpy() for c in batch.columns]
        bad_index += int(np.count_nonzero(ix != np.arange(offset, offset+len(ix))))
        boundaries = np.flatnonzero(user[1:] != user[:-1])+1
        new_blocks = user[boundaries].tolist()
        if previous_user != int(user[0]): new_blocks.insert(0, int(user[0]))
        assert len(set(new_blocks)) == len(new_blocks) and not seen.intersection(new_blocks), 'noncontiguous user'
        seen.update(new_blocks)
        descending += int(np.count_nonzero((user[1:]==user[:-1]) & (ts[1:]<ts[:-1])))
        descending += int(previous_user == int(user[0]) and previous_time is not None and ts[0]<previous_time)
        lo, hi = np.searchsorted(required, [offset, offset+len(user)])
        take = required[lo:hi]-offset
        owners[lo:hi], times[lo:hi] = user[take], ts[take]
        previous_user, previous_time = int(user[-1]), int(ts[-1])
        unknown += int(np.count_nonzero(cat==1)); pad += int(np.count_nonzero(cat==0))
        offset += len(user)
        if offset % 100_000_000 == 0: print(f'event scan: {offset:,}', flush=True)
    assert required.min()>=0 and required.max()<offset
    assert descending == bad_index == pad == 0
    return required, owners, times, dict(events=offset, users=len(seen),
        unknown_tokens=unknown, unknown_share=unknown/offset,
        order_violation_count=descending, invalid_index_count=bad_index,
        pad_event_count=pad, contiguous_user_blocks=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'results/f5_donor_uniqueness_audit_v1')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Audit output must be new')
    manifest=json.loads((ROOT/'results/f5_behavior_v1/manifest.json').read_text())
    hashes={}
    for name, expected in manifest['output_sha256'].items():
        hashes[name]=digest(ROOT/name)
        assert hashes[name]==expected, name
    histories=[pl.read_parquet(DATA/f'history_{day}.parquet') for day in DAYS]
    required, owners, times, store=stream_endpoints(histories)
    summaries=[]; distributions=[]; partitions=[]; checks=[]
    for day,h in zip(DAYS,histories):
        split='train' if day<'2017-05-12' else ('val' if day=='2017-05-12' else 'test')
        refs=pl.read_parquet(DATA/f'{split}_{day}.parquet')
        # Exposure mappings must equal the independently read user-level history.
        columns=['start','donor_start','length','donor_length','raw_length','donor_user_raw','fallback']
        joined=refs.filter(pl.col('length')>0).join(h.select('user_raw',*columns),on='user_raw',suffix='_saved')
        assert joined.height==refs.filter(pl.col('length')>0).height
        assert not any(joined.filter(pl.col(c)!=pl.col(c+'_saved')).height for c in columns)
        empty=refs.filter(pl.col('length')==0)
        assert empty.filter((pl.col('start')!=0)|(pl.col('donor_start')!=0)|(pl.col('donor_length')!=0)).height==0
        s,d,p=assignment_statistics(h,refs)
        cutoff=int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Asia/Shanghai')).timestamp())
        ownership=future=0
        for start_col,owner_col in [('start','user_raw'),('donor_start','donor_user_raw')]:
            start=h[start_col].to_numpy().astype('int64'); end=start+h['length'].to_numpy()-1
            a,b=np.searchsorted(required,start),np.searchsorted(required,end)
            ownership+=int(np.count_nonzero((owners[a]!=h[owner_col].to_numpy())|(owners[b]!=h[owner_col].to_numpy())))
            future+=int(np.count_nonzero(times[b]>=cutoff))
        assert np.all(h['end'].to_numpy().astype('int64')-h['start'].to_numpy()==h['length'].to_numpy())
        integrity=dict(date=day,self_match_count=s['self_match_count'],length_mismatch_count=s['length_mismatch_count'],
            future_event_count=future,ownership_mismatch_count=ownership,order_violation_count=store['order_violation_count'])
        assert not any(v for k,v in integrity.items() if k!='date'), integrity
        summaries.append(dict(date=day,split=split,**s));checks.append(integrity)
        distributions.extend(dict(date=day,**row) for row in d)
        partitions.extend(dict(date=day,**row) for row in p)
        print(json.dumps(summaries[-1]),flush=True)
    args.output.mkdir(parents=True)
    for name,rows in [('daily_summary',summaries),('indegree_distribution',distributions),('partition_summary',partitions),('integrity',checks)]:
        pl.DataFrame(rows).sort('date').write_csv(args.output/f'{name}.csv')
    record=dict(scope='saved daily donor mapping, exposure coverage and streaming event-store structural verification; no predictions or model execution',
        user_denominator='all nonempty saved history users, including users without an exposure that day',
        reuse_excess_definition='assignments - distinct donors; differs from targets assigned a reused donor',
        exposure_definition='all day exposures of targets assigned a donor with global daily indegree > 1; denominator includes empty history',
        eligible_definition='all nonempty daily history users; unused = eligible minus distinct assigned donors',
        partition_definition='indegree/reuse calculated within each target partition; global affected-target count also retained; no claimed cross-partition eligibility pool',
        integrity_method='contiguous user blocks and monotone event time verified over full store; window endpoint ownership and cutoff checked for all saved history windows; exposure references equal history mappings. Zero counts are logical all-token proofs, not expanded per-token recounts.',
        event_store=store,validation=summaries[-2],test=summaries[-1],input_sha256=hashes,
        source_sha256=digest(__file__),passed=True)
    (args.output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__': main()
