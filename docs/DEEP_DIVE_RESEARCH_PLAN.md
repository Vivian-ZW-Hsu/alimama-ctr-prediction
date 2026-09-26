# Frozen deep-dive research plan (D2–D5)

## Authority and immutable evidence

Frozen starting point: **badecf4** and `docs/DIN_F4_TEST_RESULTS.md` (SHA256
`d508e9be7747ef1c0dc87ac4787aa7cae97e7c1f7311bdbebf3ea4b9ac29c6b3`).
Keep that report and all existing bootstrap, calibration, model, and prediction
artifacts unchanged. Reuse their values, calibrators' already saved outputs and
per-user GAUC tables. Do not repeat bootstrap, calibrator fitting, CTR training,
or test inference. D3 may evaluate frozen components on validation only.

The working thesis is that high-quality past information matters more than
complexity under this protocol; matched history adds a small increment limited
by drift, cold start and sparse history. It is a hypothesis to qualify against
evidence, not a required conclusion. New slices are descriptive, not new
confirmatory tests. Existing global findings and the length-20/double-cold
anomalies were already observed; this plan is frozen before **new** slice results,
not retrospectively before the original test.

Latest user instructions supersede the charter's stale pending-DIN status and
historical requirements for benchmark-anchor retraining or further final test.
Five-model diagnostics mean LR, LightGBM, Wide & Deep, DeepFM and the preselected
Mean matched sequence model. Report Attention matched as an alternate fifth
model for common-error sensitivity, never as a newly selected primary model.
Repeat comparisons for each residual seed (42, 2026, 3407); four tabular predictors
and zero are fixed references. No seed-averaged probability ensemble is introduced.

## Fixed semantic groups, before new grouped test inspection

All boundaries below are semantic, not chosen from test outcomes. Missing values
get explicit missing groups. Sparse groups are retained with counts and undefined
AUC where appropriate; no outcome-driven merging. Report exposure counts, clicks,
CTR, AUC when defined, LogLoss, mean probability, COPC, and differences. Reuse the
previous basic length/cold/category tables instead of reproducing them.

| Variable | Fixed rule |
|---|---|
| Retained history | Existing 0, 1, 2–5, 6–10, 11–19, 20 |
| Raw positive-history count | 0, 1, 2–5, 6–10, 11–19, exactly 20, 21–49, >=50 |
| F3 user/ad prior exposures | 0; 1–11; 12–99; >=100 (smoothing evidence weights n/(n+100): 0, <0.1, <0.5, >=0.5) |
| F2 14-day event count | 0; 1–9; 10–99; 100–999; >=1000 |
| Latest positive-history age | empty; 1 day; 2–3 days; >=4 days, completed Beijing dates |
| Retained raw-category diversity | empty; unique categories / length <=0.25; (0.25,0.5]; >0.5 |
| Candidate hit | Existing raw-category membership in own last-20 matched history |
| Fine history evidence | F2 present/absent × F3 user exposure present/absent × F3 ad exposure present/absent × F4 positive history present/absent |
| Cold start | Existing train-vocabulary user/ad four quadrants; cross with fine history evidence |
| Population | Original raw profile codes: gender, age_level, pvalue_level, shopping_level, has_profile; no inferred demographic relabeling |
| Context | Placement raw code; Beijing hour blocks 00–05, 06–11, 12–17, 18–23 |
| Probability | [0,.01), [.01,.03), [.03,.05), [.05,.10), [.10,.20), [.20,.50), [.50,1]; anchor on zero prediction for loss attribution |
| Past same-user/ad exposures | 0, 1, 2–5, 6–19, >=20; strictly prior Beijing days |

For long-history diagnosis cross retained length=20 with raw count, hit, cold
quadrant, diversity, recency and zero probability band separately. Decompose
LogLoss change into clicked and nonclicked exposure contributions, weighted by
the original group size. Do not misinterpret within-slice AUC as additive global
AUC or claim observational decomposition establishes causation.

For double-cold calibrated COPC, partition by fine evidence, profile availability,
F3 exposure evidence and retained length. Sum expected clicks, observed clicks,
and excess expected clicks. Ratios are formed from sums, never averages of cell
COPCs. Report each partition's share of total excess. Reuse saved calibrated
probabilities, one zero reference, and all three Mean matched seeds.

## D2 — Saved-prediction error and heterogeneity analysis

1. Audit `archive/v1-original` code/notebook/README for the old “60% missed” claim:
   exact denominator, threshold, missing-model fallback, label/prediction alignment
   and sample size. Source inspection already shows archived `scripts/analysis.py`
   uses **top-10% quantiles**, not probability >=0.5. Reproduce old numbers only if
   original arrays exist; otherwise distinguish executable rule from unverifiable
   numeric claim. Do not execute archived training/inference code.
2. On current saved predictions report both semantic probability >=0.5 and exact
   top 1%, 5%, 10% exposure budgets (descending probability, impression-ID tie
   break, ceil(q*N)). Also reproduce archived >=90th-percentile tie-inclusive
   logic to quantify excess quota from ties. These are fixed diagnostic operating
   policies, not selected production thresholds. Denominator for common misses
   is all observed clicks; report recall, false positives, selected exposure share,
   pairwise FN/FP Jaccard and common-error profile composition at exact top 10%.
3. Report Pearson score correlation, Spearman midrank correlation, and residual
   (p-y) correlation; model agreement is not ground-truth correctness. Compare
   label-informed minimum per-row LogLoss oracle to each fixed model, and union
   recall with its expanded coverage. These are unattainable oracle bounds, not a
   deployable ensemble, and not equal-budget gains.
4. Quantify sequence heterogeneity and the fixed long-history anomaly with signed
   probability/log-loss changes by label and semantic groups. Keep seed variation
   separate; add no bootstrap or new significance claims.
5. Decompose **existing per-user GAUC** numerator differences by user-constant
   history length, known/new user, F2 activity and F3 user evidence. Use original
   eligible-user exposure weights and show each group's contribution to overall
   ΔGAUC plus coverage. Ad coldness varies within user and cannot define an
   additive whole-user GAUC decomposition; report user type, not ad-based fragments.
6. Report population/activity metrics for all five model families, plus matched
   versus mismatched sequence differences. Explain association and support sizes;
   demographic codes are dataset attributes, not causal mechanisms.

Outputs: `results/deep_dive_d2_v1/`, ignored compact context, D2 report, tests.
Reconcile subgroup counts, additive losses, expected clicks and GAUC contributions
to frozen overall results. Check source fingerprints. Commit/push D2 before D3.

## D3 — Validation-only information and component reliance

No new fitted diagnostic model is required: use frozen checkpoints and diagnostic
interventions on **May 12 only**, never call a test loader from the D3 script.
Reuse LR F1/F12/F123 validation predictions and their recorded recipes; identify
any recipe differences that confound a pure information ablation.

Fixed diagnostics for LR F123 and DeepFM F123: full input, F2 neutralized, F3
neutralized, both neutralized. Neutralization means numeric standardized value
zero (train-scaler mean), binary zero; F1 unchanged. These are reliance tests of
an already fitted model, not retrained layer ablations. They may be off-distribution
and must not be presented as optimal reduced-information models.

DeepFM exact additive decomposition: wide/first-order, FM, deep logits; report
their means/std, pairwise correlations, and validation metrics with each component
removed, wide alone, wide+FM, wide+deep, and deep alone. No bias refitting,
calibration, hyperparameter search or selection; retain all diagnostic outputs.
LR: group linear-logit contributions into F1/F2/F3 plus intercept and reconcile
their sum with frozen validation scores. Also reuse training-history curves to
describe optimization/overfitting without claiming causal identification.

Smoke uses a fixed validation prefix; full diagnostic pass follows only after
logit decomposition and saved-prediction parity checks. Hash original checkpoints
before/after. Write new validation-only artifacts and commit/push.

## D4 — Rolling descriptive temporal drift and exposure repetition

Preserve the fixed split and completed-day semantics. Describe dates May 6–13
from existing base/F2/F3/F4 tables; rolling three-day summaries are descriptive
only. Do not train rolling-origin models or evaluate frozen full-train models
on training dates as if held out. Model performance is available only for existing
validation/test predictions; report day and fixed hour-block differences without
claiming two held-out days establish a stable drift law.

Track daily CTR, fixed-vocabulary cold shares (training-day membership is
retrospective, not online novelty), history/evidence coverage, fixed semantic-bin
distribution drift (total variation distance), and past same-user/ad exposure
count. Daily pair counts accumulate only after that day ends; no current-day
feedback enters any prior-day diagnostic. Separately report within-day pair
multiplicity descriptively, not as an available feature or a fatigue causal effect.
Report repeat-bucket CTR and saved-prediction metrics on validation/test; conditional
associations may reflect selection, exposure policy and survivorship.

## D5 — Unified report and reproducible figures

Update README and produce a final report and standalone PDF/PNG plots using
standard plotting tools from versioned small summaries. Include information-layer
gains, fixed model comparison, already computed sequence CIs, common-error/threshold
audit, component reliance, GAUC contribution, cold-evidence composition, calibration
tradeoffs and temporal/repeat evidence. Inspect rendered plots.

Keep effect sizes, sampling uncertainty, seed variability, known protocol deviations
and untested causal hypotheses separate. Do not claim all old proposal ablations,
three-seed full backbones, matched external benchmarks or comparable throughput
were completed. No new inference benchmark is allowed on test. Report existing
resource manifests with measurement-scope caveats.

## Resources, gates and validation

Reuse local columnar data; never rebuild the 22 GB behavior log. Process necessary
columns/partitions rather than expanding all histories. Planning caps within the
historical finishing/diagnostic allowance: D2 60 minutes, D3 30 minutes, D4 45
minutes, D5 15 minutes of machine execution (2.5 hours total; not an extension of
the historical 60-hour total). Resource smoke precedes full jobs; record measured
time and stop if the cap/remaining historical allowance cannot be honored. No
open-ended searches. Most work is scans, ranks and saved-artifact summaries.

Only new formal test use, altered time semantics, or exceeding the existing
compute budget triggers a pause. Ordinary implementation, checks, documentation,
non-destructive commit and push proceed autonomously. At each stage: immutable
input checks, relevant unit tests, independent reconciliation, compilation, diff
review, no large artifacts staged, clear commit and push.
