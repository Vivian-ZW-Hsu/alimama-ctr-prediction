> Status: validation selection and the authorized first F5 test are complete. The
> pre-test freeze checks below describe their historical state; no new inference,
> training, resampling, calibration fitting or selection is authorized.

# F5 behavior-sequence study: validation, freeze, and the pre-test gate

**Historical pre-test state.** `test_authorized` and `test_inference_performed`
remain false in the immutable pre-test
[`freeze_manifest.json`](../results/f5_selection_v1/freeze_manifest.json). No F5
test inference had been run at that point; the subsequent authorized pass and its
metrics are now complete, as reported in F5_BEHAVIOR_TEST_RESULTS.md.

The original model-selection metrics are measured on validation, which participated in learning-rate
and epoch selection. These numbers are **descriptive**, not an unbiased estimate
of held-out performance and not a significance claim.

## Why F5 exists

The completed F4 study represents user history as the ordered last 20 categories
of a user's **past clicked ads**. Stage E1 measured what that costs: on the test
day that source retains a mean of 4.69 tokens and leaves 30.21% of impressions
with no history at all, while `behavior_log` would supply 185.19 retained tokens
at L=200 with 0.435% empty. F5 tests whether the small F4 sequence effect was
limited by its source.

F5 **adds** an information budget. F4 is not replaced, rewritten or read.

## Fixed protocol

Committed in `b223020` as
[`preregistration.json`](../results/f5_selection_v1/preregistration.json), before
any F5 model was trained.

| Element | Value |
|---|---|
| Source | `behavior_log`, completed Beijing window 2017-04-22 … 2017-05-14 |
| Tokens | event category + behavior type (`pv/cart/fav/buy`) |
| Retained length | **200**, taken from the published Alimama protocol, not from this project's data |
| Cutoff | event Beijing date strictly before the impression's; no same-day events |
| Category vocabulary | the existing **train-only ad-category mapping**, so history and candidate share one space |
| Behavior-only categories | map to UNKNOWN; 4.3029% of in-window events |
| Backbone | the existing frozen seed-42 DeepFM F123 checkpoint, never refitted |
| Residual | additive zero-initialised type embedding, width 8, head `[64,32]`, attention width 32 |
| Learning rates searched | 5e-4 and 2e-4, paired-LogLoss rule inherited from F4 |
| Seeds | 42, 2026, 3407 |
| Primary model | `behavior_mean_matched`, declared before any result |
| Compute ceiling | 6 hours; the study used **85.6 min** over 17 unique runs |

Selection outcome: paired validation LogLoss **0.189705709** at 5e-4 against
**0.189718897** at 2e-4, so **5e-4** was selected and the controls inherited it.

## Validation results, three seeds each

| Condition | AUC mean ± std | LogLoss mean ± std | PR-AUC | GAUC ± std | COPC | ECE |
|---|---|---|---:|---|---:|---:|
| zero, pipeline null | 0.6503151437 ± 0 | 0.1898125375 ± 0 | 0.087680 | 0.5762131 ± 0 | 0.956963 | 0.003808 |
| **mean matched** | 0.6517027355 ± 1.19e-4 | **0.1896931486** ± 1.35e-5 | 0.088293 | 0.5766577 ± 1.38e-4 | 0.964881 | 0.003745 |
| mean mismatched | 0.6504489543 ± 7.24e-5 | 0.1897917362 ± 7.42e-6 | 0.087747 | 0.5761838 ± 1.61e-4 | 0.965833 | 0.003616 |
| attention matched | 0.6518265825 ± 5.16e-5 | 0.1896852506 ± 2.36e-5 | 0.088305 | 0.5767523 ± 1.29e-4 | 0.962896 | 0.003804 |
| attention mismatched | 0.6504844471 ± 2.82e-6 | 0.1897909480 ± 8.24e-6 | 0.087760 | 0.5762531 ± 1.91e-5 | 0.962463 | 0.003666 |

GAUC covers **101,129 eligible users**, **50.3639%** of the 3,234,051 validation
impressions. Standard deviations use ddof=1 over three **residual** seeds on one
shared frozen backbone; they are not three independent DeepFM fits.

### Increments over the pipeline null

| Contrast | ΔAUC | ΔLogLoss | ΔGAUC |
|---|---:|---:|---:|
| mean matched − zero | **+0.0013876** | **−0.0001194** | +0.0004447 |
| attention matched − zero | +0.0015114 | −0.0001273 | +0.0005392 |
| mean mismatched − zero | +0.0001338 | −0.0000208 | **−0.0000293** |
| attention mismatched − zero | +0.0001693 | −0.0000216 | +0.0000400 |
| attention matched − mean matched | +0.0001238 | −0.0000079 | +0.0000945 |

## F5 control clarification and remaining imbalance

The control is a **deterministic different-user donor assignment with a measured
approximately 0.08% user-level reuse deviation**, not a strict one-to-one
permutation. Validation reuse excess is **884 / 1,131,745 (0.078109%)**; test is
**917 / 1,134,206 (0.080850%)**. Reused donors number **869 / 911**, with maximum
indegree **4 / 3**. Targets assigned reused donors number **1,753 / 1,828**;
**575 / 562** appear in that day's exposures, covering **15,389 (0.475843%)** /
**12,845 (0.388223%)** impressions. Self-match, length, future-event, ownership
and order violations remain zero. Small reuse does not establish negligible
metric impact; no retrained strict-derangement counterfactual exists.

The saved validation control has category TV **0.04982**, age **3.75 vs 4.08 days**,
and exposure-weighted past activity **138.23 vs 87.72**. Retained length is exact
and coarse activity/recency buckets nearly match. User-level balance does not
imply exposure-level balance. The activity gap is residual exposure-weighted,
within-bucket imbalance; it is not simply attributable to donor reuse.
Under this implemented control, the approximately **95%** AUC gap ratio is
descriptive, not causal identity attribution. Full date/indegree/partition tables:
[F5 donor audit](F5_DONOR_UNIQUENESS_AUDIT.md).

## What validation suggests, and what it does not

**The source mattered.** `mean matched − zero` is **+0.0013876** AUC against the
F4 study's **+0.000311** on the same frozen backbone, about **4.5×** larger, and
**−0.0001194** LogLoss against F4's −0.000039, about **3.1×** larger. This is the
direction F5 was built to test. It is a validation observation; the subsequent authorized test is reported
separately and is complete.

**Matched exceeds the implemented control.** Matched adds +0.0013876 AUC,
while mismatched adds +0.0001338, or **9.6%** of that gap. These validation ratios
are descriptive, not causal attribution to user identity. Exact length and coarse
buckets do not eliminate exposure-weighted activity, popularity or age imbalance.

**Attention leads mean by +0.0001238 validation AUC**, comparable to the mean
condition's 1.19e-4 residual-seed standard deviation. This observation did not
change the preregistered mean primary choice and does not identify a causal role
for sparsity. The later frozen test is reported separately.

**The gap to published work remains large.** The DSIN table's analogous
"introduce a user behavior sequence" contrast is about +0.0074 AUC; F5's
+0.00139 is roughly 5× smaller. Candidate reasons, none isolated here: the
residual sits on a frozen backbone instead of being trained jointly; the cutoff is
completed-day rather than per-event; tokens are category and type only, with no
item identifier; and 4.3029% of full in-window events map to UNKNOWN
(validation matched retained-token share is 3.939985%, a different denominator). These are limitations,
not explanations.

**Epoch behaviour differs from F4.** Every F4 run selected epoch 1. F5's 15 runs
selected epoch 1 three times, epoch 2 seven times, epoch 3 four times and epoch 4
once, so the richer source keeps improving validation LogLoss for longer.

## Freeze verification

[`freeze_audit.json`](../results/f5_selection_v1/freeze_audit.json) re-derives its
checks rather than trusting the study's own report:

- All 15 checkpoint and validation-prediction hashes match the freeze manifest.
- At that pre-test freeze, no run had a test prediction file and
  `test_authorized` was false. The subsequent authorized pass is complete.
- The build inputs are byte-identical to what the build recorded.
- All **12,744** empty-history validation rows reproduce the frozen DeepFM
  baseline **exactly** in all 15 runs.
- The zero control reproduces the frozen backbone with a maximum deviation of
  **exactly 0.0** across all 3,234,051 rows, for each of the three seeds. It is a
  genuine pipeline null, not a weak model.
- The frozen backbone and scaler hashes are unchanged, and the F5 context cache
  reproduced the saved DeepFM validation predictions with a maximum probability
  error of **exactly zero**.

### Mismatch-control balance, and what it does not control

[`validation_control_balance.json`](../results/f5_selection_v1/validation_control_balance.json),
computed after selection, on 596,629,796 tokens per condition:

| Diagnostic | Value |
|---|---|
| Impression-weighted category total variation distance | 0.04982 |
| Mean retained event age, matched / mismatched | 3.7498 / 4.0848 days |
| Activity-bucket mismatches | 196 rows |
| Recency-bucket mismatches | 2,474 rows |
| Donor coarsening by level (0/1/2/3) | 3,231,572 / 2,283 / 196 / **0** |
| Mean past exposures, matched / mismatched | 138.23 / 87.72 |

Two points need stating plainly.

The **last-resort donor level was never used**: level 3, which would have drawn a
donor whose retained length merely *exceeds* the target's, has zero rows. At full
scale the exact-retained-length rule inside the activity and recency buckets was
always satisfiable, and coarsening touched 0.077% of rows.

The **past-exposure gap is residual exposure-weighted imbalance under coarse
bucket matching**. Frequent target users contribute more impressions, whereas
user-level donor assignment does not preserve exposure weights. Activity buckets
are binary at 100 prior impressions and permit large within-bucket differences.
Near balance at user level is not exact balance in the exposure-level evaluation
population. Retained length is exact and coarse activity/recency buckets nearly
match; exposure-weighted activity, popularity and event age do not. The additional
non-bijective donor reuse is separately quantified in the donor audit; **138.23 vs
87.72 cannot simply be attributed to its approximately 0.08% reuse excess**.

## A recorded deviation

The authorization asked that donor fallback be zero in the formal build. At L=200
that is not jointly achievable with exact retained-length matching, which the same
authorization also required: about 84% of rows sit at the length cap, so
intermediate lengths are sparse and a (length, activity, recency) group can hold a
single user, which admits no different-user donor at all.

Exact retained length was judged the stronger guarantee and was preserved
unconditionally. Bucket coarsening is counted by level instead of prohibited, and
the integrity failures that *must* be zero — self match, length mismatch, future
events, ownership mismatch, order violation, invalid type — are zero on every
date. In the event the relaxation stayed small: 683 user-date donor assignments
across the whole build, and 0.077% of validation rows.

## Historical gate (completed)

The user subsequently authorized the single formal pass using the frozen
checkpoints and six comparisons. It is complete. No rerun is authorized.

## Historical execution record (completed; do not rerun)

```bash
PY=/opt/anaconda3/envs/ctr_project/bin/python   # the tested interpreter; others are refused
$PY -B src/44_build_f5_sequences.py --run-name NEW_SMOKE --smoke-test
$PY -B src/45_audit_f5_sequences.py --run-name NEW_SMOKE --compare-smoke OTHER_SMOKE --certificate NEW_CERT
$PY -B src/44_build_f5_sequences.py --run-name NEW_BUILD --smoke-certificate NEW_CERT
$PY -B src/46_cache_f5_context.py --f5-run NEW_BUILD --run-name NEW_CACHE
$PY -B src/48_select_f5_configuration.py      # refuses to run once frozen
$PY -B src/51_audit_f5_freeze.py
$PY -B -m unittest discover -s tests -v
```

Completed run names are immutable; a repeat needs fresh names. Large caches,
checkpoints and prediction Parquets are Git-ignored local artifacts, and the
committed manifests are evidence about them, not backups of them.
