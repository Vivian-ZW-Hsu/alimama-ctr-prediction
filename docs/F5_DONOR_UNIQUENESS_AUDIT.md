# F5 donor uniqueness: read-only release clarification

The frozen control is a **deterministic different-user donor assignment with a
measured approximately 0.08% user-level reuse deviation**, not a strict one-to-one
derangement. This audit reads saved mappings and events; it never regenerates
histories, predictions or model fits. The original builder remains unchanged.

## Denominators

Assignments include every nonempty user in each saved daily history, including
users without a target impression that day. Eligible donors are that same set.
Reuse excess = assignments − distinct donors; the unique-donor rate is distinct
/ assignments. A target is affected when its assigned donor has global daily
indegree > 1, including the donor's first use. Exposure coverage counts all such
targets' impressions and divides by ALL day impressions, including empty history.
These are different quantities, not estimates of metric damage.

## All dates

| Date | Assignments / eligible users | Distinct donors | Reuse excess (%) | Reused donors / max indegree | Affected targets | Affected impressions (% of day) |
|---|---:|---:|---:|---:|---:|---:|
| 2017-05-06 | 1,107,275 | 1,106,665 | 610 (0.055090%) | 610 / 2 | 1220 | 1,501 (0.045897%) |
| 2017-05-07 | 1,112,995 | 1,112,345 | 650 (0.058401%) | 650 / 2 | 1300 | 6,728 (0.196151%) |
| 2017-05-08 | 1,118,101 | 1,117,323 | 778 (0.069582%) | 775 / 3 | 1553 | 12,211 (0.364016%) |
| 2017-05-09 | 1,122,350 | 1,121,491 | 859 (0.076536%) | 848 / 3 | 1707 | 15,495 (0.477060%) |
| 2017-05-10 | 1,125,944 | 1,125,128 | 816 (0.072473%) | 809 / 3 | 1625 | 14,538 (0.436085%) |
| 2017-05-11 | 1,128,991 | 1,128,115 | 876 (0.077591%) | 863 / 4 | 1739 | 15,482 (0.458237%) |
| 2017-05-12 | 1,131,745 | 1,130,861 | 884 (0.078109%) | 869 / 4 | 1753 | 15,389 (0.475843%) |
| 2017-05-13 | 1,134,206 | 1,133,289 | 917 (0.080850%) | 911 / 3 | 1828 | 12,845 (0.388223%) |

## Validation and test

| Quantity | Validation 05-12 | Test 05-13 |
|---|---:|---:|
| Nonempty targets / assignments / eligible donors | 1,131,745 | 1,134,206 |
| Distinct donors | 1,130,861 | 1,133,289 |
| Unique-donor rate | 99.921891% | 99.919150% |
| Reuse excess | 884 (0.078109%) | 917 (0.080850%) |
| Reused donors | 869 | 911 |
| Maximum indegree | 4 | 3 |
| Unused eligible donors | 884 (0.078109%) | 917 (0.080850%) |
| All targets assigned a reused donor | 1,753 | 1,828 |
| Such targets exposed that day | 575 | 562 |
| Associated impression rows | 15,389 / 3,234,051 | 12,845 / 3,308,665 |
| Associated exposure share | 0.475843% | 0.388223% |
| Donor indegree 0 / 1 / 2 / 3 / 4 | 884 / 1,129,992 / 855 / 13 / 1 | 917 / 1,132,378 / 905 / 6 / 0 |

The independently computed 869/911 reused donors and maxima 4/3 agree with the
reported preliminary check. Reuse excess is 884/917, not 869/911. The daily CSV
also records unused-donor counts/rates; the indegree CSV includes zero-degree
eligible donors. Partition statistics cover target fallback, retained length,
activity bucket and recency bucket. Indegrees within a partition and targets
assigned globally reused donors are explicitly separate columns.

## Integrity and limits

All eight days have zero self-match, retained-length mismatch, future-event,
ownership and order violations. A bounded, sequential scan of 723,113,323 events
proves contiguous user blocks and nondecreasing time; every saved matched/donor
window has the correct endpoint owner and ends before Beijing midnight cutoff.
Together these invariants prove the conditions for every retained token without
expanding exposure × length. Every nonempty exposure reference equals its saved
user history; empty references remain empty. All build output hashes match.

The saved validation control still has impression-weighted category TV **0.04982**,
mean age **3.75 vs 4.08 days**, and exposure-weighted past activity **138.23 vs
87.72**. Retained length is exact and coarse activity/recency buckets nearly match.
Near user-level balance does not mean exposure-level balance. The activity gap is
residual imbalance under exposure weighting and coarse bins; it must not simply
be attributed to the approximately 0.08% donor reuse, an additional limitation.

Under the implemented different-user donor control, matched-minus-mismatched
accounts for about **95%** (94.5172%) of the matched-minus-zero AUC gap; mismatched
retains about **5.5%**. This is a descriptive ratio, not a causal identity
contribution or proof that popularity, activity and recency have been removed.
The control is not a marginal-distribution-preserving one-to-one permutation.
The small reuse fraction does not establish negligible metric impact. No strict
derangement version was retrained, so its counterfactual metrics are unknown.
The previously saved exclusion of 1,457 bucket-relaxed test rows is a different
sensitivity diagnostic; it does not quantify donor reuse effects.

## Evidence and read-only entry point

- [Daily summary](../results/f5_donor_uniqueness_audit_v1/daily_summary.csv)
- [Full indegree distribution](../results/f5_donor_uniqueness_audit_v1/indegree_distribution.csv)
- [Partition summary](../results/f5_donor_uniqueness_audit_v1/partition_summary.csv)
- [Integrity checks](../results/f5_donor_uniqueness_audit_v1/integrity.csv)
- [Manifest and definitions](../results/f5_donor_uniqueness_audit_v1/manifest.json)

`python -B src/57_audit_f5_donor_uniqueness.py --output results/NEW_AUDIT_DIRECTORY`
requires a new destination, reads no prediction and executes no model. Synthetic
five-user tests reproduce the frozen duplicate behavior and test audit detection.
