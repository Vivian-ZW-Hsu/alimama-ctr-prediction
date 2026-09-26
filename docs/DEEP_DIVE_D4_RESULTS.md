# D4: Descriptive temporal drift and repeated exposure

This stage preserves the original Beijing-date split and completed-day cutoff.
It scans existing base/F2/F3/F4 artifacts and reuses saved validation/test scores.
No rolling-origin model, new training, calibration or test inference is run.
Execution took 47.06 seconds. Daily source counts reconcile to 26,557,961 rows.

## Traffic and evidence change across dates

| Day | Exposures | CTR | F3 user history share | F3 ad history share | F4 nonempty share | Prior-pair repeat share |
|---|---:|---:|---:|---:|---:|---:|
| 2017-05-06 | 3,270,348 | 0.052084 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2017-05-07 | 3,430,002 | 0.051248 | 0.628587 | 0.886062 | 0.316744 | 0.028242 |
| 2017-05-08 | 3,354,523 | 0.051988 | 0.764067 | 0.918138 | 0.457525 | 0.076894 |
| 2017-05-09 | 3,248,016 | 0.052218 | 0.828629 | 0.932906 | 0.548026 | 0.121908 |
| 2017-05-10 | 3,333,752 | 0.051651 | 0.863280 | 0.937100 | 0.607246 | 0.150281 |
| 2017-05-11 | 3,378,604 | 0.052633 | 0.886756 | 0.965160 | 0.650866 | 0.170889 |
| 2017-05-12 | 3,234,051 | 0.049299 | 0.900562 | 0.932908 | 0.681318 | 0.183548 |
| 2017-05-13 | 3,308,665 | 0.050319 | 0.905965 | 0.972197 | 0.697866 | 0.196722 |

CTR moves 5.2633% → 4.9299% → 5.0319% on May 11/12/13, so drift is not a simple
monotonic decline. The observed positive-history window grows over time, making
F4 coverage and repeated exposure rise mechanically; this is partly history-window
accumulation, not purely behavioral drift. May 6 is left-censored for impression
history, even though earlier behavior-log history supports F2.

From validation to test, train-vocabulary new-user exposure share rises
9.9438% → 11.6968%, and new-ad share 6.7092% → 8.7676%. Yet past ad-feedback
coverage rises 93.2908% → 97.2197%, since completed May 12 feedback is available.
Novelty relative to train and evidence available by the current cutoff therefore
move in different directions. Training-day knownness uses the final train
vocabulary retrospectively, not an online novelty measurement.

Fixed-bin total variation distances, May 13 versus May 12, are 0.002467 for F2
activity, 0.023071 for F3 user evidence, 0.051824 for F3 ad evidence, 0.025983 for
history length, 0.034954 for cold quadrants and 0.013174 for repeat count. These
are marginal descriptive distances, not tests of concept drift or proof of why
model performance changes. Rolling three-day traffic summaries are saved; no
training-date prediction is reported as held-out model performance.

## Prior-day repeated user/ad exposure

Counts sum the same raw user/ad pair's exposures on dates strictly before D.
Current-day counts are added only after that day's cutoff. Separate within-day
repeat-exposure shares (1 − daily distinct pairs / exposures) rise from 1.9584%
on validation to 2.4374% on test; they are descriptive, not available predictors.

| Split | Prior-pair exposure count | Exposures | Clicks | CTR |
|---|---|---:|---:|---:|
| val | 0 | 2,640,447 | 134,743 | 0.051030 |
| val | 1 | 417,541 | 18,954 | 0.045394 |
| val | 2-5 | 169,177 | 5,614 | 0.033184 |
| val | 6-19 | 6,882 | 126 | 0.018309 |
| val | 20+ | 4 | 0 | 0.000000 |
| test | 0 | 2,657,778 | 139,297 | 0.052411 |
| test | 1 | 436,726 | 20,085 | 0.045990 |
| test | 2-5 | 204,335 | 6,951 | 0.034018 |
| test | 6-19 | 9,799 | 157 | 0.016022 |
| test | 20+ | 27 | 0 | 0.000000 |

Higher observed prior repeat counts are associated with lower CTR. This is not
an identified fatigue effect: exposure policy, user/ad selection, survivorship
and short left-censored histories are plausible confounders. The >=20 cells
contain only 4 validation / 27 test exposures and no clicks; AUC is undefined.
Do not generalize from this extreme sparse bucket.

The model ordering is not uniform across repeat strata. On test, Mean matched
AUC is 0.639041 versus LR 0.647198 in the no-prior-pair bucket; at 2–5 repeats it
is 0.669192 versus LR 0.666275; at 6–19 it is 0.626807 versus LR 0.600397, but the
latter has only 9,799 exposures / 157 clicks. Mean matched still has slightly
worse LogLoss than LR in those repeated strata. These are fixed-group descriptive
point estimates, not new significance claims or a rule for model routing.

## Reused held-out performance and interpretation

`heldout_daily_metrics.csv` references existing overall scores. `heldout_hour_metrics.csv`
reuses D2 hour blocks rather than recomputing them. The general validation-to-test
ranking decrease co-occurs with novelty, history and exposure composition changes;
these observational summaries do not isolate their causal contributions. Only
two held-out dates exist, insufficient to establish a stable temporal law or a
rolling deployment performance estimate.

## Audit and outputs

[`results/deep_dive_d4_v1`](../results/deep_dive_d4_v1) contains daily/rolling
summaries, semantic-bin distributions, TV distances, repeat counts and per-model,
per-residual-seed repeat metrics. `verification.json` confirms row reconciliation
and complete repeat partitions; a synthetic fixture and independent sample
cumulative-count audit verify exclusion of the whole current day. Source hashes
are unchanged. Large daily-pair/context tables remain local and Git-ignored.
