# D2: Saved-prediction deep dive

Plan frozen and pushed in **e494030**, before these new grouped results. Starting
DIN evidence is **badecf4**; its report, bootstrap and calibration remain unchanged.
This analysis took 419.53 seconds. No fitting, bootstrap or CTR inference occurred.
New slices are descriptive; residual-seed std is not a sampling confidence interval.

## Common errors and the archived “60%” claim

The archive README claims 60% of clicks were missed by all five models on roughly
2.85M test exposures. `archive/v1-original:scripts/analysis.py:109–124` explicitly
uses **per-model test 90th-percentile thresholds**, including ties, not p>=0.5.
The denominator is all clicks in `test_labels.npy`. Missing model arrays are
silently skipped. Arrays are aligned by position without an impression-ID check;
the later failure-analysis model is chosen by test AUC. Required old arrays and
notebook numeric outputs are unavailable, so the exact 60% cannot be reproduced.

Thus “the old 60% came from a 0.5 threshold” is **contradicted by available source**.
A budget-dependent miss rate also cannot establish intrinsic model impossibility.
The following new diagnostic uses the rebuilt cohort with ID-validated predictions.
Five models are LR, LightGBM, Wide & Deep, DeepFM and the preselected Mean matched;
Attention matched substitutes as fifth only for sensitivity.

| Policy | Common missed fraction of all clicks, mean across residual seeds |
|---|---:|
| Probability >=0.5 | 99.9676% |
| Exact top 1% per model | 94.6095% |
| Exact top 5% per model | 82.4252% |
| Exact top 10% per model | 71.3787% |
| Legacy tie-inclusive top-10% quantile | 71.3787% |

The Attention alternative gives 71.3783% at top 10%. The Mean five-model union
captures 28.62% of clicks while selecting **14.61% of exposures**, not 10%; this
is expanded coverage, not an equal-budget ensemble gain. Exact budgets break
ties by ascending impression ID. All per-seed counts and threshold sensitivity
are retained in `common_errors.csv`, `pairwise_overlap.csv` and `oracle.csv`.

Common-miss rates at top 10% are 72.27% for known/known, 86.88% for known-user/new-ad,
59.74% for new-user/known-ad, and 79.62% for double cold. Known/known contributes
about 98,266 out of 118,838 common missed clicks. Absolute count, within-group
miss rate and traffic composition answer different questions: misses are not
uniformly concentrated in new users. See `common_miss_profiles.csv`.

## Why capped-length AUC improves while LogLoss worsens

The frozen retained-length-20 Mean matched − Mean mismatched differences are
AUC +0.000474273 and LogLoss +0.000019789. Exact loss decomposition gives:

| Label | Exposures | Contribution to total capped-group ΔLogLoss |
|---|---:|---:|
| Nonclick | 239,628 | approximately +0.000615 |
| Click | 10,994 | approximately −0.000595 |

Both clicked and nonclicked groups receive higher average matched probabilities.
The many nonclicks' added loss slightly exceeds clicks' loss reduction. This is
an accounting explanation, not causal identification of truncation or training.
AUC measures pair ordering; LogLoss also responds to probability magnitudes.
Matched still improves LogLoss over **zero** in this group; the reversal is
relative to **mismatched**. Exactly-20, 21–49 and >=50 raw-history subgroups all
have positive net loss differences. The >=50 subgroup contributes roughly 61%
of net deterioration despite about 15% of capped exposures.

`long_history_loss.csv` preserves per-seed, label-separated decompositions by
raw count, candidate hit, cold quadrant, diversity, recency and zero-score band.
Each overlapping partition independently sums to the same overall loss change.
No partition is treated as an independent significance finding.

## What double-cold calibrated COPC≈0.581 contains

Evidence bits mean **F2 present / F3 user exposure history / F3 ad exposure history /
F4 positive history**. Vocabulary novelty is different from missing history:
completed May 12 feedback can exist for an ID absent from the training vocabulary.

| Evidence | Exposures | Observed clicks | Mean expected clicks | Share of excess expected clicks |
|---|---:|---:|---:|---:|
| 1010 | 14,639 | 531 | 970.62 | 59.54% |
| 1000 | 6,259 | 235 | 351.87 | 15.83% |
| 1110 | 2,716 | 89 | 174.93 | 11.64% |
| 1111 | 1,378 | 61 | 91.65 | 4.15% |
| 1100 | 1,213 | 36 | 68.39 | 4.39% |
| 0010 | 814 | 31 | 38.06 | 0.96% |
| 1101 | 696 | 22 | 40.48 | 2.50% |
| 0000 | 586 | 19 | 25.04 | 0.82% |

Three additional tiny zero-click cells are retained in the full CSV. Across
28,331 exposures, 1,024 observed clicks compare with about 1,762 expected clicks.
The largest excess comes from users with behavior and ad evidence but no user
exposure/positive history. Completely absent evidence contributes under 1% of
excess. Therefore “double cold equals no information” is false; an empty-history
fallback alone cannot explain most aggregate overprediction. This establishes
composition, not the cause of the model's probability error.

Ratios are formed from sums per seed, not averaged subgroup COPCs. Full saved
raw/calibrated decompositions also cover profile availability, user/ad evidence
strength and retained length, plus one zero reference. No calibration was refitted.

## Which users contribute the GAUC increment

Existing eligible-user AUC tables are reused. Contribution is sum(user exposure
weight × matched-minus-reference user AUC) divided by all eligible exposure
weights. Contributions add to frozen ΔGAUC; ordinary within-group AUCs do not.

| Group | Eligible exposures | Contribution to Mean matched − zero GAUC |
|---|---:|---:|
| Length 0 | 316,209 | 0 |
| Length 1 | 176,354 | approximately −0.000004 |
| Length 2–5 | 454,699 | approximately +0.000185 |
| Length 6–10 | 278,368 | approximately +0.000135 |
| Length 11–19 | 233,644 | approximately +0.000012 |
| Length 20 | 222,973 | approximately +0.000150 |
| Known user, separate partition | 1,541,782 | approximately +0.000473 |
| New user, separate partition | 140,465 | approximately +0.000005 |

About 98.88% of the +0.000478364 total comes from known users. F2 activity 100–999
contributes about 77%, >=1000 about 22%. These are effect × traffic-weight
contributions, not proof activity causes the benefit. Length 1 is slightly
negative despite positive global-ranking gains. Eligibility remains 104,884 users,
1,682,247 exposures (50.84%); no ad-based fragments redefine user eligibility.
Both zero and mismatched references are in `gauc_contributions.csv`.

## Correlation, oracle complementarity and population heterogeneity

Mean matched / DeepFM score Pearson correlation is 0.99734; Wide & Deep / DeepFM
is 0.99590. Other pairs range about 0.8955–0.9066. `correlations.csv` additionally
contains Spearman midrank and (p−y) residual correlation, separately by seed/split.
`pairwise_overlap.csv` reports FN/FP Jaccard, recall and selected/false-positive
counts at all fixed policies. Agreement is not correctness.

The label-informed per-row minimum LogLoss among five models is about 0.178235,
versus LR 0.192429: optimistic gap 0.014193. It needs the true label to choose
the predictor. It is **not** a deployable ensemble, attainable gain, or new model.
No blending or formal selection follows this oracle analysis.

`group_metrics.csv` and `group_summary.csv` report all fixed raw gender/age/value/
shopping codes, profile availability, placement, hour, F2 activity, F3 user/ad
evidence, raw history, diversity, recency and fine cold/evidence cells. All four
tabular models and three seeds each of Mean matched/Mean mismatched are included.
`heterogeneity_differences.csv` gives seed-paired matched-minus-mismatched and
matched-minus-zero differences. Fixed-reference std is undefined/null, not three
independent seeds. Sparse cells remain, including undefined single-class AUC.
Population codes are not converted into inferred identities or causal claims.

## Verification and artifacts

All tables: [`results/deep_dive_d2_v1`](../results/deep_dive_d2_v1).
`verification.json` independently reconciles full partitions, click denominators,
GAUC contributions against frozen point estimates, expected-click sums and
label-separated losses. `manifest.json` binds unchanged inputs and compute time.
Smoke tests verify quota ties, evidence boundaries, miss denominators and oracle
bounds. Every prediction load checks row ID and label parity. No new bootstrap
or significance claims were added. The original DIN report remains hash-identical.
