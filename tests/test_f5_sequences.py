"""Synthetic-fixture tests for the F5 behavior-log sequence construction.

No raw table, event store, model or frozen result is read.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from f5_sequences import (
    MAX_LENGTH, PAD_INDEX, TYPE_TOKENS, TYPE_VOCABULARY_SIZE, UNKNOWN_INDEX,
    attach_histories, audit_references, daily_histories, donor_permutation, midnight,
)


def store(rows):
    """Build an event store fixture already in (user, time, source) order."""
    frame = pl.DataFrame(
        rows,
        schema={'user_raw': pl.Int64, 'time_stamp': pl.Int64,
                'cate_idx': pl.UInt16, 'type_idx': pl.UInt8},
    )
    # The real store is written in (user, time, source-position) order.
    return frame.with_row_index('source_position').sort(
        'user_raw', 'time_stamp', 'source_position'
    ).drop('source_position').with_row_index('event_index')


def arrays_of(frame):
    return {
        'user_raw': frame['user_raw'].cast(pl.Int32).to_numpy(),
        'time_stamp': frame['time_stamp'].cast(pl.Int32).to_numpy(),
        'cate_idx': frame['cate_idx'].to_numpy(),
        'type_idx': frame['type_idx'].to_numpy(),
    }


class CutoffTests(unittest.TestCase):
    def test_cutoff_is_beijing_midnight_and_excludes_same_day(self):
        day = date(2017, 5, 7)
        cutoff = midnight(day)
        # 08:00 UTC on 2017-05-07 is already 16:00 Beijing, i.e. the same day.
        self.assertEqual(cutoff, 1494086400)
        events = store([
            {'user_raw': 10, 'time_stamp': cutoff - 2, 'cate_idx': 5, 'type_idx': 2},
            {'user_raw': 10, 'time_stamp': cutoff - 1, 'cate_idx': 6, 'type_idx': 3},
            {'user_raw': 10, 'time_stamp': cutoff, 'cate_idx': 7, 'type_idx': 2},
            {'user_raw': 20, 'time_stamp': cutoff - 5, 'cate_idx': 8, 'type_idx': 2},
            {'user_raw': 20, 'time_stamp': cutoff + 100, 'cate_idx': 9, 'type_idx': 2},
            # Partners at each retained length so a different-user donor exists.
            {'user_raw': 30, 'time_stamp': cutoff - 4, 'cate_idx': 11, 'type_idx': 2},
            {'user_raw': 30, 'time_stamp': cutoff - 3, 'cate_idx': 12, 'type_idx': 2},
            {'user_raw': 40, 'time_stamp': cutoff - 6, 'cate_idx': 13, 'type_idx': 2},
        ])
        activity = pl.DataFrame({
            'user_raw': [10, 20, 30, 40],
            'event_date_bj': [date(2017, 5, 6)] * 4,
            'impressions': [150, 3, 150, 3],
        })
        history = daily_histories(events.lazy(), activity, day, max_length=20, seed=1)
        by_user = {r['user_raw']: r for r in history.iter_rows(named=True)}
        # The event exactly at the cutoff is excluded.
        self.assertEqual(by_user[10]['raw_length'], 2)
        self.assertEqual(by_user[20]['raw_length'], 1)
        # start/end address the retained window inside the store.
        self.assertEqual((by_user[10]['start'], by_user[10]['end']), (0, 2))
        self.assertEqual((by_user[20]['start'], by_user[20]['end']), (3, 4))
        # Activity bucket uses the >=100 prior-impression threshold.
        self.assertEqual(by_user[10]['activity_bucket'], 1)
        self.assertEqual(by_user[20]['activity_bucket'], 0)

    def test_truncation_keeps_the_most_recent_events(self):
        day = date(2017, 5, 7)
        cutoff = midnight(day)
        rows = [{'user_raw': 10, 'time_stamp': cutoff - 30 + i, 'cate_idx': 100 + i,
                 'type_idx': 2} for i in range(25)]
        # A second user long enough to donate exactly 20 events, plus two shorts.
        rows += [{'user_raw': 15, 'time_stamp': cutoff - 60 + i, 'cate_idx': 200 + i,
                  'type_idx': 2} for i in range(22)]
        rows += [{'user_raw': 20, 'time_stamp': cutoff - 5, 'cate_idx': 7, 'type_idx': 2}]
        rows += [{'user_raw': 25, 'time_stamp': cutoff - 4, 'cate_idx': 8, 'type_idx': 2}]
        events = store(rows)
        activity = pl.DataFrame({'user_raw': [10, 15, 20, 25],
                                 'event_date_bj': [date(2017, 5, 6)] * 4,
                                 'impressions': [200, 200, 200, 200]})
        history = daily_histories(events.lazy(), activity, day, max_length=20, seed=1)
        row = [r for r in history.iter_rows(named=True) if r['user_raw'] == 10][0]
        self.assertEqual(row['raw_length'], 25)
        self.assertEqual(row['length'], 20)
        # The retained window is the LAST 20 events, so it ends at the user's last.
        self.assertEqual(row['end'], 25)
        self.assertEqual(row['start'], 5)
        kept = events['cate_idx'].to_numpy()[row['start']:row['end']]
        self.assertEqual(kept[0], 105)
        self.assertEqual(kept[-1], 124)


class DonorTests(unittest.TestCase):
    def frame(self, lengths, activity, recency, ends=None):
        size = len(lengths)
        ends = ends if ends is not None else [l * 10 + l for l in lengths]
        return pl.DataFrame({
            'user_raw': pl.Series(range(1, size + 1), dtype=pl.Int64),
            'raw_length': pl.Series(lengths, dtype=pl.UInt32),
            'end': pl.Series(ends, dtype=pl.UInt32),
            'length': pl.Series(lengths, dtype=pl.UInt8),
            'start': pl.Series([e - l for e, l in zip(ends, lengths)], dtype=pl.UInt32),
            'activity_bucket': pl.Series(activity, dtype=pl.UInt8),
            'recency_bucket': pl.Series(recency, dtype=pl.UInt8),
        })

    def test_no_self_match_and_exact_length_preserved(self):
        history = self.frame([5, 5, 5, 5], [1, 1, 0, 0], [0, 0, 1, 1])
        result = donor_permutation(history, seed=7)
        self.assertTrue((result['donor_user_raw'] != result['user_raw']).all())
        np.testing.assert_array_equal(
            result['donor_length'].to_numpy(), result['length'].to_numpy()
        )
        # The donor window ends at the donor's own end index.
        donor_by_user = {r['donor_user_raw']: r for r in result.iter_rows(named=True)}
        ends = dict(zip(history['user_raw'], history['end']))
        for row in result.iter_rows(named=True):
            self.assertEqual(row['donor_start'] + row['donor_length'],
                             ends[row['donor_user_raw']])
        self.assertEqual(set(result['fallback'].to_list()), {0})

    def test_assignment_is_deterministic_for_a_fixed_seed(self):
        history = self.frame([3, 3, 3, 7, 7, 7], [1] * 6, [0] * 6)
        first = donor_permutation(history, seed=11)
        second = donor_permutation(history, seed=11)
        self.assertTrue(first.equals(second))
        different = donor_permutation(history, seed=12)
        self.assertEqual(first.height, different.height)

    def test_sparse_bucket_falls_back_and_records_the_level(self):
        # Length 9 exists in only one (activity, recency) cell per bucket, so the
        # exact-length rule must coarsen rather than fail or change the length.
        history = self.frame([9, 9, 4, 4, 4, 4], [1, 0, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1])
        result = donor_permutation(history, seed=5)
        self.assertTrue((result['donor_user_raw'] != result['user_raw']).all())
        np.testing.assert_array_equal(
            result['donor_length'].to_numpy(), result['length'].to_numpy()
        )
        nine = result.filter(pl.col('length') == 9)
        # Both length-9 rows needed a coarser bucket, recorded as a nonzero level.
        self.assertTrue((nine['fallback'] > 0).all())
        self.assertTrue((nine['fallback'] <= 3).all())

    def test_single_row_at_a_length_uses_a_longer_donor_at_exact_length(self):
        # One user has length 6 and nobody else does; a length-10 user can still
        # supply exactly 6 of its own most recent events.
        history = self.frame([6, 10, 10], [1, 1, 1], [0, 0, 0])
        result = donor_permutation(history, seed=3)
        row = result.filter(pl.col('length') == 6).row(0, named=True)
        self.assertNotEqual(row['donor_user_raw'], row['user_raw'])
        self.assertEqual(row['donor_length'], 6)
        self.assertEqual(row['fallback'], 3)
        ends = dict(zip(history['user_raw'], history['end']))
        self.assertEqual(row['donor_start'] + 6, ends[row['donor_user_raw']])


class ReferenceTests(unittest.TestCase):
    def test_empty_history_is_a_zero_pointer_and_audit_counts_it(self):
        day = date(2017, 5, 7)
        cutoff = midnight(day)
        events = store([
            {'user_raw': 10, 'time_stamp': cutoff - 3, 'cate_idx': 5, 'type_idx': 2},
            {'user_raw': 10, 'time_stamp': cutoff - 2, 'cate_idx': 6, 'type_idx': 5},
            {'user_raw': 20, 'time_stamp': cutoff - 9, 'cate_idx': 7, 'type_idx': 3},
            {'user_raw': 20, 'time_stamp': cutoff - 8, 'cate_idx': 8, 'type_idx': 4},
        ])
        activity = pl.DataFrame({'user_raw': [10, 20],
                                 'event_date_bj': [date(2017, 5, 6)] * 2,
                                 'impressions': [120, 130]})
        history = daily_histories(events.lazy(), activity, day, max_length=20, seed=2)
        # User 30 has no events at all, so its reference must be all zeros.
        targets = pl.DataFrame({
            'impression_id': pl.Series([1, 2, 3], dtype=pl.UInt32),
            'user_raw': pl.Series([10, 20, 30], dtype=pl.Int64),
            'event_date_bj': [day] * 3,
            'clk': pl.Series([0, 1, 0], dtype=pl.Int8),
        })
        refs = attach_histories(targets, history)
        empty = refs.filter(pl.col('user_raw') == 30).row(0, named=True)
        self.assertEqual((empty['length'], empty['start'], empty['donor_start']), (0, 0, 0))
        self.assertEqual(empty['donor_user_raw'], -1)

        report = audit_references(refs, arrays_of(events), day, max_length=20)
        self.assertEqual(report['empty_rows'], 1)
        self.assertEqual(report['future_event_count'], 0)
        self.assertEqual(report['identity_mismatch_count'], 0)
        self.assertEqual(report['self_match_count'], 0)
        self.assertEqual(report['invalid_type_count'], 0)
        # Matched and mismatched retain the same token budget.
        self.assertEqual(report['token_stats']['matched']['tokens'],
                         report['token_stats']['mismatched']['tokens'])

    def test_audit_rejects_a_reference_that_crosses_the_cutoff(self):
        day = date(2017, 5, 7)
        cutoff = midnight(day)
        events = store([
            {'user_raw': 10, 'time_stamp': cutoff - 1, 'cate_idx': 5, 'type_idx': 2},
            {'user_raw': 10, 'time_stamp': cutoff + 1, 'cate_idx': 6, 'type_idx': 2},
        ])
        # A deliberately corrupted reference: length 2 reaches past the cutoff.
        refs = pl.DataFrame({
            'impression_id': pl.Series([1], dtype=pl.UInt32),
            'clk': pl.Series([0], dtype=pl.UInt8),
            'user_raw': pl.Series([10], dtype=pl.Int64),
            'event_date_bj': [day],
            'start': pl.Series([0], dtype=pl.UInt32),
            'donor_start': pl.Series([0], dtype=pl.UInt32),
            'length': pl.Series([2], dtype=pl.UInt8),
            'donor_length': pl.Series([2], dtype=pl.UInt8),
            'raw_length': pl.Series([2], dtype=pl.UInt32),
            'donor_user_raw': pl.Series([10], dtype=pl.Int64),
            'fallback': pl.Series([0], dtype=pl.UInt8),
        })
        with self.assertRaises(ValueError) as caught:
            audit_references(refs, arrays_of(events), day, max_length=20)
        self.assertIn('reference audit failed', str(caught.exception))


class VocabularyTests(unittest.TestCase):
    def test_reserved_indices_and_fixed_type_enumeration(self):
        self.assertEqual(PAD_INDEX, 0)
        self.assertEqual(UNKNOWN_INDEX, 1)
        self.assertEqual(sorted(TYPE_TOKENS), ['buy', 'cart', 'fav', 'pv'])
        # Types occupy 2..5, leaving PAD and UNKNOWN reserved.
        self.assertEqual(min(TYPE_TOKENS.values()), 2)
        self.assertEqual(TYPE_VOCABULARY_SIZE, 6)
        self.assertEqual(MAX_LENGTH, 200)


if __name__ == '__main__':
    unittest.main()
