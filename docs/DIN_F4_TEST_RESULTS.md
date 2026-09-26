# Frozen five-condition sequence test

## Execution and scope

The user authorized this test after a hard-zero same-pipeline validation control.
The [v2 freeze](../results/sequence_selection_v2/freeze_manifest.json), ordered
[comparison plan](../results/sequence_selection_v2/comparison_plan.json), zero
validation, 11 passing tests, reload checks and end-to-end validation smoke were
committed and pushed in **0656c23 before test**. The previous v1 freeze and all
old model results remain unchanged.

Formal inference started at **2026-09-26 22:08:40 UTC** and processed
**3,308,665 test impressions per checkpoint**, for five conditions x three seeds.
Each checkpoint made exactly one streaming inference pass. The same frozen
DeepFM backbone was evaluated once per batch and its logits shared. Inference
took 19.61 s with peak process RSS 1.50 GiB on MPS; this includes prediction
serialization and is not a standalone model throughput benchmark.

The [inference manifest](../results/sequence_test_v2/manifest.json) binds outputs
to checkpoints and the pre-test commit. An exclusive start marker prevents
automatic re-inference. The [post-inference audit](../results/sequence_test_v2/post_inference_integrity.json)
verifies all frozen code, 15 checkpoints, existing validation predictions and the
v1 freeze are unchanged. No model, hyperparameter, epoch, length, seed or
comparison priority was changed after test.

LR=5e-4, training batch=4096, L=20, seeds={42,2026,3407}, selected epoch=1 for all
15 checkpoints. Positive exposure timestamps proxy feedback time; histories use
strictly earlier Beijing dates. Test can use completed May 12 positives. There
is no independently observed click timestamp. Raw probabilities are reported;
no new calibration was fitted or selected in the original inference batch.
The later authorized saved-prediction calibration is reported below.

## Test results

Residual conditions report arithmetic mean ± sample std (ddof=1) across the
three fixed seeds. Zero's historical ±0 describes identical repeated references,
not independent training seeds; the follow-up below uses it only once.
All metrics use the shared evaluator with `--splits test`; its ECE uses 10
equal-count bins. Original DeepFM seed 42 is **not** inserted into these means.

| Condition | AUC | LogLoss | PR-AUC |
|---|---:|---:|---:|
| **Mean matched (primary model)** | **0.64626339 ± 0.00004400** | **0.193120334 ± 0.000003806** | 0.08804174 ± 0.00002511 |
| Mean mismatched | 0.64585768 ± 0.00001459 | 0.193150876 ± 0.000002009 | 0.08777754 ± 0.00000847 |
| Attention matched | 0.64625477 ± 0.00003916 | 0.193118498 ± 0.000004741 | 0.08803429 ± 0.00002822 |
| Attention mismatched | 0.64585746 ± 0.00001582 | 0.193150630 ± 0.000001218 | 0.08777694 ± 0.00000824 |
| Zero sequence, same pipeline | 0.64593304 ± 0 | 0.193176498 ± 0 | 0.08786996 ± 0 |

| Condition | GAUC | COPC | ECE |
|---|---:|---:|---:|
| Mean matched | 0.57038011 ± 0.00010783 | 0.94900500 ± 0.00132999 | 0.00460008 ± 0.00004426 |
| Mean mismatched | 0.56993575 ± 0.00005886 | 0.95244747 ± 0.00123438 | 0.00444568 ± 0.00003738 |
| Attention matched | 0.57038503 ± 0.00012487 | 0.94988648 ± 0.00142697 | 0.00457081 ± 0.00004037 |
| Attention mismatched | 0.56995242 ± 0.00002793 | 0.95264662 ± 0.00059783 | 0.00444192 ± 0.00002643 |
| Zero sequence, same pipeline | 0.56990175 ± ~0 | 0.93965590 ± 0 | 0.00479462 ± 0 |

GAUC has **104,884 eligible users**, covering **1,682,247 / 3,308,665 impressions
(50.8437%)** for every condition. It is weighted by impression count among users
with both classes. The three zero-control prediction files are bit-identical,
and probabilities match the original frozen DeepFM exactly. The recorded GAUC
std is 7.85e-17 from floating-point aggregation, not model randomness. The zero
control is structurally deterministic and must not be interpreted as three
independent DeepFM fits. All conditions share that one frozen backbone.

Historical reference, kept outside the fair-comparison mean: the original
DeepFM seed-42 AUC is 0.6459330408 and LogLoss 0.1931764982. See the separate
[historical_reference.csv](../results/sequence_test_v2/historical_reference.csv).

## Five comparisons fixed before test

Differences below are **left minus right**, paired by seed, reported as mean ±
sample std. Positive AUC and negative LogLoss favor the left condition.

| Order | Comparison | Δ AUC | Δ LogLoss |
|---|---|---:|---:|
| **Primary 1** | Mean matched − zero sequence | **+0.000330347 ± 0.000044004** | **−0.000056164 ± 0.000003806** |
| **Primary 2** | Mean matched − mean mismatched | **+0.000405708 ± 0.000040055** | **−0.000030542 ± 0.000001963** |
| Secondary 3 | Attention matched − zero sequence | +0.000321725 ± 0.000039161 | −0.000058000 ± 0.000004741 |
| Secondary 4 | Attention matched − attention mismatched | +0.000397302 ± 0.000035461 | −0.000032132 ± 0.000004883 |
| Secondary 5 | Attention matched − mean matched | −0.000008622 ± 0.000023227 | −0.000001836 ± 0.000003220 |

The primary model remains **Mean matched**, as specified before test. Both
matched variants have small AUC/LogLoss gains over zero and their mismatched
controls. Attention trades a slightly lower AUC for a slightly lower LogLoss
relative to mean; these differences do not justify a claim of clear superiority
or a post-test model switch.

Mismatched histories improve LogLoss and aggregate calibration relative to zero
despite lower AUC. Matched histories rank better than mismatched ones, while the
mismatched variants have COPC closer to one and lower ECE. Thus ranking gains and
aggregate calibration gains are different effects; the full LogLoss change must
not be attributed solely to user-specific sequence content. The negative control
preserves exact length and coarse past-activity bucket, but does not exactly
balance impression-weighted popularity, age or within-bucket activity. These
limitations were recorded before test and were not used to change the mapping.

Seed std is not a confidence interval or a user-cluster significance test.
The initial inference/reporting batch did not include user-cluster bootstrap;
the authorized saved-prediction analysis below now supplies it. The claim
remains a small conditional incremental gain on this frozen backbone and test day,
not general superiority of DIN or attention, nor evidence of solved cold-start
calibration. Each run's shared evaluation includes cold-start quadrants and
reliability bins; these did not trigger any new configuration.

## Saved evidence

- [Per-seed metrics](../results/sequence_test_v2/test_metrics.csv): all six metrics and GAUC coverage.
- [Condition mean/std](../results/sequence_test_v2/seed_summary.csv).
- [Five comparisons, all six metrics](../results/sequence_test_v2/comparisons.csv).
- [Paired per-seed differences](../results/sequence_test_v2/paired_seed_differences.csv).
- [Report manifest and hashes](../results/sequence_test_v2/report_manifest.json).
- [Validation and zero-control interpretation](DIN_F4_VALIDATION.md).

Formal predictions remain under `artifacts/predictions/*_test_v2/`, local and
Git-ignored. Small manifests, metrics, evaluation logs and documents are committed.
Reports consumed saved predictions and did not repeat model inference. Do not
rerun `26_infer_sequence_test.py`: the authorized once-only pass is complete.


## Saved-prediction statistical validation

This follow-up changes no CTR model or frozen test prediction. The analysis
protocol was committed/pushed in **15309c2**, after aggregate test results were
known but before these new intervals, fixed group diagnostics and calibration
results. The five comparison priorities were already frozen before the original
test. This is not a claim of retrospectively preregistered statistical methods.
See [analysis protocol](DIN_F4_POSTHOC_PROTOCOL.md).

**391,741 raw users**, each with all their exposures, are the resampling units.
Exactly 1,000 valid paired multinomial replicates use seed **20260927** and the
same user multiplicities for all 13 distinct predictors. Metrics are computed
per predictor, followed by arithmetic averaging of seed-paired differences; this
is not an ensemble of probabilities. Zero is loaded once. The exact rank/tie
implementation agrees with the shared evaluator and explicit duplicated-user
fixtures. Full bootstrap computation took 11.07 s after rank preparation (not
an end-to-end analysis timing). No score-binning approximation was used.

The following intervals are marginal percentile 95% intervals for **data-sampling
uncertainty conditional on the fixed checkpoints and this test day**. They do
not include training-seed or future-day uncertainty. The residual-seed sample
std is a separate quantity; all residuals share one frozen seed-42 backbone.
Zero has no independent training-seed estimate. Two-sided centered bootstrap
p-values have finite-simulation resolution 1/1001, so none is reported as zero.

| Frozen comparison | ΔAUC [95% CI] | ΔLogLoss [95% CI] | Holm p, joint primary AUC/LL family |
|---|---:|---:|---:|
| Primary 1: Mean matched − zero | +0.000330347 [+0.000215737, +0.000440027] | -0.000056164 [-0.000067417, -0.000044484] | 0.003996 for both |
| Primary 2: Mean matched − mean mismatched | +0.000405708 [+0.000304701, +0.000507655] | -0.000030542 [-0.000041474, -0.000019668] | 0.003996 for both |
| Secondary 3: Attention matched − zero | +0.000321725 [+0.000208814, +0.000431600] | -0.000058000 [-0.000069351, -0.000046682] | Exploratory; unadjusted |
| Secondary 4: Attention matched − attention mismatched | +0.000397302 [+0.000294334, +0.000499393] | -0.000032132 [-0.000042770, -0.000021458] | Exploratory; unadjusted |
| Secondary 5: Attention matched − mean matched | -0.000008622 [-0.000025994, +0.000009911] | -0.000001836 [-0.000003917, -0.000000037] | Exploratory; unadjusted |

| Frozen comparison | ΔPR-AUC [95% CI] | ΔGAUC [95% CI] |
|---|---:|---:|
| Primary 1: Mean matched − zero | +0.000171781 [+0.000113881, +0.000229218] | +0.000478364 [+0.000179023, +0.000775345] |
| Primary 2: Mean matched − mean mismatched | +0.000264203 [+0.000205212, +0.000323010] | +0.000444364 [+0.000168473, +0.000704206] |
| Secondary 3: Attention matched − zero | +0.000164329 [+0.000105517, +0.000223238] | +0.000483282 [+0.000182188, +0.000770442] |
| Secondary 4: Attention matched − attention mismatched | +0.000257348 [+0.000201324, +0.000316950] | +0.000432610 [+0.000173721, +0.000688491] |
| Secondary 5: Attention matched − mean matched | -0.000007452 [-0.000018714, +0.000005668] | +0.000004918 [-0.000072432, +0.000077555] |

Holm across the **two primary comparisons separately per metric** gives adjusted
p=0.001998 for each AUC, LogLoss and PR-AUC contrast, and p=0.007992 for each GAUC
contrast. The stricter joint family of four primary AUC/LogLoss hypotheses gives
p=0.003996 for all four. The intervals themselves are not simultaneous/Holm
intervals. Full effects, bootstrap std, raw p and both corrections are in
[bootstrap_intervals.csv](../results/sequence_posthoc_v1/bootstrap_intervals.csv).
Secondary comparisons are **mechanistic/exploratory**, with no confirmatory
multiplicity claim. This priority label does not imply that bootstrap methods
were specified before the already observed aggregate test results.

Attention − mean has AUC p=0.333666, PR-AUC p=0.232767 and GAUC p=0.897103.
Its LogLoss difference is −0.000001836 (about 0.00095% relative), with a percentile
interval barely below zero, while the predeclared centered-bootstrap p=0.064935.
Percentile intervals and centered tests are not mathematical inverses; this
borderline discrepancy is retained, not resolved by picking the favorable one.
It does not establish a robust practical attention advantage. Mean remains the
primary model; there is no post-test selection.

Mean matched reduces raw LogLoss by about **0.0291% relative to zero** and
**0.0158% relative to mean mismatched**. Its AUC gains are **0.0330 and 0.0406
percentage points**, respectively: statistically detectable, but small.

### Training seed variation, separate from bootstrap

All four metrics favor matched over mismatched in **3/3 residual seeds** for
both pooling mechanisms. These are three residual fits conditional on the
same backbone, not three independent full-backbone trainings.

| Mechanism | Seed | ΔAUC | ΔLogLoss | ΔPR-AUC | ΔGAUC |
|---|---:|---:|---:|---:|---:|
| Mean | 42 | +0.000450105 | -0.000032193 | +0.000294665 | +0.000582176 |
| Mean | 2026 | +0.000372283 | -0.000031063 | +0.000245411 | +0.000326503 |
| Mean | 3407 | +0.000394737 | -0.000028371 | +0.000252532 | +0.000424412 |
| Attention | 42 | +0.000435771 | -0.000032780 | +0.000284667 | +0.000576491 |
| Attention | 2026 | +0.000390216 | -0.000036659 | +0.000264187 | +0.000398125 |
| Attention | 3407 | +0.000365919 | -0.000026958 | +0.000223189 | +0.000323215 |

[Direction consistency and paired sample std](../results/sequence_posthoc_v1/training_seed_direction_consistency.csv)
are reported separately from the bootstrap intervals. Three seeds do not support
a reliable population-level training-randomness confidence interval.


## Fixed diagnostic groups

Membership is identical across models and uses the target user's own retained
matched history, including for the mismatched model. Length 20 means the retained
cap (raw length >=20). Raw category IDs define candidate membership; two UNKNOWN
encoded values are not treated as a match. Empty history always has no hit.
Cold-start quadrants use only the original train user/ad vocabularies. These
buckets were fixed before this follow-up's grouped results, after aggregate test
reporting. No bucket outcome changes the frozen model or configuration.

Every model/group's exposure count, CTR, AUC, LogLoss, average probability, COPC
and seed std are in [group_summary.csv](../results/sequence_posthoc_v1/group_summary.csv);
[group_metrics.csv](../results/sequence_posthoc_v1/group_metrics.csv) retains
individual seeds. Undefined single-class AUC is represented as NaN rather than
zero (all observed test groups here have both classes). Counts are exposures,
never multiplied by the number of seeds. The full raw mean/zero comparisons and
paired seed differences are in
[group_difference_summary.csv](../results/sequence_posthoc_v1/group_difference_summary.csv)
and [group_paired_differences.csv](../results/sequence_posthoc_v1/group_paired_differences.csv).
Attention-versus-mean raw comparisons are limited to history length; its other
raw group rows serve only as before-calibration references.

The table below reports Mean matched − Mean mismatched group point estimates.
These are descriptive slices, not group-level significance tests or subgroup
interaction tests; differences between slices are not established causal effects.
Global AUC is not a weighted average of within-group AUCs.

| Group | Value | Exposures | CTR | ΔAUC | ΔLogLoss |
|---|---|---:|---:|---:|---:|
| history_bin | 0 | 999,659 | 0.051085 | +0.000000000 | +0.000000000 |
| history_bin | 1 | 470,153 | 0.052155 | +0.000640956 | -0.000066292 |
| history_bin | 2-5 | 897,107 | 0.052957 | +0.000632927 | -0.000058104 |
| history_bin | 6-10 | 404,308 | 0.049418 | +0.000551368 | -0.000039772 |
| history_bin | 11-19 | 286,816 | 0.043300 | +0.000456183 | -0.000023154 |
| history_bin | 20 | 250,622 | 0.043867 | +0.000474273 | +0.000019789 |
| category_hit | False | 2,308,021 | 0.048995 | +0.000076909 | -0.000017637 |
| category_hit | True | 1,000,644 | 0.053374 | +0.000401045 | -0.000060307 |
| cold_start_segment | known_user__known_ad | 2,659,898 | 0.051117 | +0.000294507 | -0.000030497 |
| cold_start_segment | known_user__new_ad | 261,758 | 0.030058 | +0.001174370 | -0.000041765 |
| cold_start_segment | new_user__known_ad | 358,678 | 0.060308 | +0.000299697 | -0.000020017 |
| cold_start_segment | new_user__new_ad | 28,331 | 0.036144 | +0.001363201 | -0.000064335 |

Raw predictions are identical across all conditions for empty histories
(999,659 impressions, 30.21% of test): no learnable fallback signal appears.
Matched AUC exceeds mismatched in every nonempty length bucket, but at length
20 its LogLoss is **worse**, despite better AUC. Category-hit slices have larger
AUC increments descriptively; this does not prove a subgroup interaction.

| Retained length | Attention matched − Mean matched ΔAUC | ΔLogLoss |
|---|---:|---:|
| 0 | +0.000000000 | +0.000000000 |
| 1 | -0.000016586 | -0.000002025 |
| 2-5 | -0.000056135 | +0.000003922 |
| 6-10 | +0.000018837 | -0.000007538 |
| 11-19 | -0.000028487 | -0.000002911 |
| 20 | +0.000050377 | -0.000018985 |

Attention has tiny favorable point differences in the 6–10 and capped-20 groups,
but not a uniform advantage across lengths. These slices have no multiplicity-
adjusted subgroup inference and do not justify choosing attention for a group.

## Validation-only probability calibration

The existing `14_calibrate_predictions.py` helpers are reused unchanged: an
impression-ID hash splits validation 80/20; Platt and isotonic fit the 80%, method
selection uses only the 20% LogLoss, and the chosen method refits full validation.
This is the historical impression-level procedure, not a user-disjoint calibration
split. The original model selection also used validation, so full-validation
calibrated metrics are explicitly **in-sample**, not an unbiased validation claim.

All **13/13** selections choose isotonic: four residual conditions × three seeds,
plus the one deterministic zero reference. All selections/refits finish and a
hashed fit manifest is saved before any test transformation. Each is applied to
saved probabilities; no CTR inference or CTR training occurs. No calibrator is
removed on the basis of test outcomes. Calibrated probabilities are saved as
float64, the exact values evaluated, in a new ignored artifact namespace.

| Condition | Calibrated LogLoss | Calibrated COPC | Calibrated ECE |
|---|---:|---:|---:|
| mean_matched | 0.192839887 ± 0.000005261 | 0.986346 ± 0.000092 | 0.001043477 ± 0.000030821 |
| mean_mismatched | 0.192883919 ± 0.000000987 | 0.986041 ± 0.000044 | 0.001067733 ± 0.000021609 |
| attention_matched | 0.192840214 ± 0.000005049 | 0.986423 ± 0.000059 | 0.001032683 ± 0.000075592 |
| attention_mismatched | 0.192883178 ± 0.000001320 | 0.986057 ± 0.000037 | 0.001077233 ± 0.000021610 |
| zero_sequence_same_pipeline | 0.192876162 (one reference) | 0.985826 (one reference) | 0.001090287 (one reference) |

| Condition | Calibrated AUC | Calibrated PR-AUC | Calibrated GAUC |
|---|---:|---:|---:|
| mean_matched | 0.646214992 ± 0.000030841 | 0.087046822 ± 0.000082947 | 0.570069609 ± 0.000083090 |
| mean_mismatched | 0.645818106 ± 0.000019564 | 0.086778029 ± 0.000012542 | 0.569530756 ± 0.000041516 |
| attention_matched | 0.646190775 ± 0.000042076 | 0.086972759 ± 0.000061158 | 0.570009727 ± 0.000389166 |
| attention_mismatched | 0.645816155 ± 0.000017981 | 0.086827888 ± 0.000062316 | 0.569532045 ± 0.000062609 |
| zero_sequence_same_pipeline | 0.645879320 (one reference) | 0.086882239 (one reference) | 0.569559926 (one reference) |

The raw counterparts appear above. Every calibrator improves overall LogLoss and
10-equal-count-bin ECE, and all 130 condition/seed × length/cold-start cells improve
LogLoss. **All 13 reduce AUC, PR-AUC and GAUC.** For Mean matched, PR-AUC drops
from 0.088041742 to 0.087046822 (about 1.13% relative). Isotonic is nondecreasing
but can merge distinct scores into ties; ranking metrics need not be preserved.
This tradeoff is retained, not used to reject or reselect a calibrator. The five
bootstrap comparisons concern the original raw frozen predictions; no new
calibrated-model significance claim is added.

For Mean matched, LogLoss moves 0.193120334 → 0.192839887, COPC 0.949005 → 0.986346,
and ECE 0.004600077 → 0.001043477. Aggregate improvement does not remove conditional
bias. COPC <1 means average probability is too high; COPC >1 means it is too low.

| Mean matched group | Value | COPC raw → calibrated | LogLoss raw → calibrated | ECE raw → calibrated |
|---|---|---:|---:|---:|
| cold_start_segment | known_user__known_ad | 1.009267 → 1.043399 | 0.194917 → 0.194783 | 0.002998 → 0.002126 |
| cold_start_segment | known_user__new_ad | 0.700977 → 0.710881 | 0.134085 → 0.133637 | 0.012959 → 0.012225 |
| cold_start_segment | new_user__known_ad | 0.783469 → 0.843223 | 0.225337 → 0.224227 | 0.016668 → 0.011213 |
| cold_start_segment | new_user__new_ad | 0.543493 → 0.581043 | 0.162014 → 0.159995 | 0.030359 → 0.026062 |
| history_bin | 0 | 0.898434 → 0.942150 | 0.197640 → 0.197117 | 0.007630 → 0.003715 |
| history_bin | 1 | 1.012084 → 1.050476 | 0.198908 → 0.198777 | 0.002760 → 0.002506 |
| history_bin | 2-5 | 0.980422 → 1.021616 | 0.199652 → 0.199510 | 0.003009 → 0.001187 |
| history_bin | 6-10 | 0.962739 → 0.994110 | 0.188146 → 0.187883 | 0.004257 → 0.000928 |
| history_bin | 11-19 | 0.959257 → 0.973860 | 0.170003 → 0.169910 | 0.002867 → 0.001423 |
| history_bin | 20 | 0.900375 → 0.924286 | 0.175336 → 0.175003 | 0.005498 → 0.003658 |

Double-cold COPC remains about **0.581**, implying average probability about 72%
above observed CTR. Known-user/known-ad COPC moves away from one (1.009 → 1.043),
as does length 1 (1.012 → 1.050), even while their LogLoss improves. Overall
near-calibration therefore conceals both remaining overprediction in cold groups
and underprediction in warm/short-history groups. Separate global calibrators
can also make originally identical empty-history raw predictions differ after
calibration; this is a probability-map effect, not a learned sequence fallback.

Full before/after results, including average predictions, are available for every
seed and condition, both splits, all length and cold-start cells:
[overall_before_after.csv](../results/sequence_posthoc_v1/overall_before_after.csv),
[overall_summary.csv](../results/sequence_posthoc_v1/overall_summary.csv),
[group_metrics.csv](../results/sequence_posthoc_v1/group_metrics.csv),
[calibration_group_difference_summary.csv](../results/sequence_posthoc_v1/calibration_group_difference_summary.csv).
Method-selection evidence is in
[calibration_selection.csv](../results/sequence_posthoc_v1/calibration_selection.csv)
and [calibration_fit_manifest.json](../results/sequence_posthoc_v1/calibration_fit_manifest.json).

## Evidence-based interpretation

1. **Matched versus mismatched:** user-matched history adds a small measurable
   increment on this frozen backbone/test day. Both primary mean contrasts pass
   Holm; all three residual seeds agree. The negative control's previously
   documented matching limitations still prevent a fully isolated causal claim.
2. **Sequence versus zero:** mean matched improves AUC, LogLoss, PR-AUC and GAUC
   over the same frozen-backbone reference. This does not imply superiority to
   the strongest tabular global baseline. The original DeepFM remains a separate
   historical reference and is never inserted into a seed mean.
3. **Attention versus mean:** no robust practical advantage is established;
   AUC/PR-AUC/GAUC intervals cross zero, and the tiny LogLoss result is borderline
   and exploratory. Mean stays the primary model as frozen before test.
4. **Global versus within-user ranking:** historical raw LR F123 has AUC
   0.652488694, LogLoss 0.192428566, GAUC 0.569938978; LightGBM has AUC
   0.651318100, LogLoss 0.192502224, GAUC 0.569889273. Mean matched's raw AUC
   0.646263388 and LogLoss 0.193120334 are worse globally, while raw GAUC
   0.570380115 is descriptively higher. The historical comparisons are outside
   the five bootstrap contrasts: superiority to LR/LightGBM in GAUC is **not**
   statistically established. GAUC covers only 50.84% of impressions, whereas
   global AUC uses all impressions and cross-user pairs. Both population and
   ranking question differ. Calibration further lowers mean GAUC to 0.570070.
5. **Significance versus size:** the primary gains are detectable under user
   resampling but very small. Sampling intervals and residual-training seed std
   answer different questions; neither establishes stability across new days or
   newly trained backbones.
6. **Calibration versus group bias:** validation-selected isotonic improves
   overall LogLoss/ECE without solving cold-start or history-length bias, and it
   slightly harms ranking. Keep both the global and conditional results.

The supported narrative is: **user-matched histories add a small increment;
attention has no robust practical advantage over mean pooling. The sequence
model improves the same-backbone zero reference but remains below historical
strong tabular baselines in global ranking/LogLoss. Its within-user ranking is
promising descriptively, with verified gains over zero/mismatched, not verified
superiority to LR/LightGBM. Global calibration improvement leaves substantial
group bias.**

Historical reference evidence:
[LR global metrics](../results/lr_f1_f2_f3/evaluation/overall_metrics.csv),
[LR GAUC](../results/lr_f1_f2_f3/evaluation/gauc_summary.csv),
[LightGBM global metrics](../results/lightgbm_f1_f2_f3/evaluation/overall_metrics.csv),
[LightGBM GAUC](../results/lightgbm_f1_f2_f3/evaluation/gauc_summary.csv).

## Follow-up verification and execution

All **16 unit tests** pass, including explicit whole-user weighted resampling,
tie handling, thread-independent bootstrap reproducibility, validation-only
calibration split/refit, undefined group AUC and single-zero-reference handling.
The read-only [verification report](../results/sequence_posthoc_v1/verification.json)
confirms unchanged source predictions/manifests, full-row group/category/vocabulary
identity, all 1,000 finite draws, independent interval/p/Holm reconstruction,
independent seed mean/sample-std calculation, and group-to-overall additive
metric reconciliation. Raw test metrics reproduce the original shared evaluator.
All 13 chosen calibrators and all outcomes are retained.

New analysis entry points, in order (completed outputs refuse overwriting):

```text
src/28_prepare_prediction_analysis.py
src/29_bootstrap_sequence_comparisons.py
src/30_calibrate_diagnose_sequences.py --stage fit
src/30_calibrate_diagnose_sequences.py --stage apply
src/31_summarize_prediction_analysis.py
src/32_verify_prediction_analysis.py
```

Large rank inputs, group context, calibrators and derived probabilities remain
local/Git-ignored. Small protocol/manifest files, per-seed and summary CSVs,
bootstrap deltas, code and tests are versioned. The original inference marker,
freeze and results are preserved. **No additional CTR training or test inference
was performed, and no result changed the frozen comparison plan.**
