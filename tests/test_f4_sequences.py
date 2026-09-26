import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from f4_sequences import daily_histories, donor_permutation, midnight, attach_histories


class F4Tests(unittest.TestCase):
    def test_midnight_and_same_day_exclusion(self):
        d = date(2017, 5, 7)
        events = pl.DataFrame({'user_raw': [10, 10, 20, 20], 'event_index': [0, 1, 2, 3],
            'time_stamp': [midnight(d)-1, midnight(d), midnight(d)-1, midnight(d)+1],
            'event_date_bj': [date(2017,5,6), d, date(2017,5,6), d]})
        activity = pl.DataFrame({'user_raw':[10,20], 'event_date_bj':[date(2017,5,6)]*2, 'impressions':[4,5]})
        h = daily_histories(events, activity, d, 20, 42)
        self.assertEqual(h['length'].to_list(), [1,1])
        self.assertEqual(h['start'].to_list(), [0,2])
        self.assertEqual(h['donor_user_raw'].to_list(), [20,10])
        first = daily_histories(events, activity, date(2017,5,6), 20, 42)
        self.assertEqual(first.height, 0)

    def test_derangement_reproducibility_and_explicit_fallback(self):
        h = pl.DataFrame({'user_raw':[1,2,3,4], 'start':[0,2,4,6],
                          'length':[2]*4, 'activity_bucket':[0,1,1,1]})
        a = donor_permutation(h, 13)
        self.assertTrue(a.equals(donor_permutation(h, 13)))
        self.assertEqual(a['fallback'].sum(), 4)
        self.assertTrue(np.all(a['user_raw'].to_numpy() != a['donor_user_raw'].to_numpy()))
        self.assertEqual(sorted(a['donor_user_raw']), [1,2,3,4])
        with self.assertRaises(ValueError):
            donor_permutation(h.head(1), 13)

    def test_latest_ordered_window(self):
        day = date(2017,5,7)
        events = pl.DataFrame({'user_raw':[1]*25+[2]*25, 'event_index':range(50),
            'time_stamp':list(range(midnight(day)-25,midnight(day)))*2,
            'event_date_bj':[date(2017,5,6)]*50})
        activity=pl.DataFrame({'user_raw':[1,2], 'event_date_bj':[date(2017,5,6)]*2,'impressions':[30,30]})
        h=daily_histories(events,activity,day,20,42)
        self.assertEqual(h['start'].to_list(),[5,30])
        self.assertEqual(h['raw_length'].to_list(),[25,25])
        targets=pl.DataFrame({'impression_id':[5], 'user_raw':[3], 'clk':[0], 'event_date_bj':[day]})
        refs=attach_histories(targets,h)
        self.assertEqual(refs['length'][0],0)
        self.assertEqual(refs['donor_start'][0],0)


if __name__ == '__main__':
    unittest.main()
