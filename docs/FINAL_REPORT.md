# Historical information, complexity and robustness in point-in-time CTR prediction

## Result and scope

Under the audited temporal protocol and observed tuning budget, past click-feedback
information provides a much larger gain than the tested increases in model
complexity. User-matched positive-category histories add a small, statistically
detectable increment to a frozen DeepFM. Target attention has no robust practical
advantage over mean pooling. Sequence benefits are uneven across history and user
groups, while probability bias persists under cold start and changing traffic.

This is a conditional empirical conclusion, not a universal claim that linear
models are better or that drift causes every generalization gap. The core statistical
evidence is frozen at **badecf4** in [DIN_F4_TEST_RESULTS.md](DIN_F4_TEST_RESULTS.md).
D2–D5 reuse it without new bootstrap, calibration fitting, CTR training or test
inference. The [deep-dive plan](DEEP_DIVE_RESEARCH_PLAN.md) was committed in
**e494030** before the new grouped findings, after aggregate test outcomes were known.

## Data and leakage controls

The study uses all 26,557,961 impressions, with no negative downsampling. Beijing
dates May 6–11 are training (20,015,245), May 12 validation (3,234,051), and May 13
frozen test reporting (3,308,665). Raw IDs join source tables before encoding;
vocabularies/scalers fit training only. Historical features use completed days
strictly before the exposure date. Test may use finished May 12 feedback, validation
may not. Positive exposure timestamps proxy feedback record time, with next-day
availability assumed; independent click timestamps are not observed.

F1 is static/context, excluding weekday in formal models. F2 summarizes past
behavior. F3 supplies smoothed entity feedback and evidence counts. F4 retains
the ordered last 20 positive-exposure categories, with train-only vocabulary,
neutral PAD/UNKNOWN and zero empty-history representations. The mismatch control
preserves cutoff, retained length and coarse activity while exchanging user
history; it does not exactly balance all popularity/age/activity effects.

## Information gain versus architecture gain

The existing LR test AUC increases from 0.595495 (F1) to 0.601790 (F12), then
0.652489 (F123). The F3 increment is about **0.05070 AUC**, versus F2's 0.00630.
These runs share optimizer settings and validation stopping, but selected epochs
differ (3, 3, 2); they are not identical fixed-epoch fits. At common F123 input,
the saved LR and LightGBM outperform the two tabular neural models globally.
F2's global AUC increase does not imply a within-user increase: the saved LR
test GAUC is 0.561913 for F1 and 0.561534 for F12, before rising to 0.569939 for
F123. Refit coefficients can change between these trained runs; this comparison
is distinct from D3's fixed-parameter F2 deletion.

| Frozen model | Raw test AUC | Raw LogLoss | GAUC |
|---|---:|---:|---:|
| LR F123 | 0.652489 | 0.192429 | 0.569939 |
| LightGBM F123 | 0.651318 | 0.192502 | 0.569889 |
| Wide & Deep F123 | 0.646219 | 0.193161 | 0.569680 |
| DeepFM F123 / zero | 0.645933 | 0.193176 | 0.569902 |
| Mean matched F123+F4 | 0.646263 ± 0.000044 | 0.193120 ± 0.000004 | 0.570380 ± 0.000108 |

Tabular rows represent one seed each; the sequence row represents three residual
seeds on one shared backbone. Its richer F4 input prevents a pure architecture
interpretation. No significance claim for LR versus LightGBM or sequence versus
LR is added. GAUC evaluates only 104,884 users with both labels, covering 50.84%
of test exposures; global AUC evaluates all exposures and cross-user pairs.

![Information and model comparison](../figures/deep_dive_v1/01_information_and_models.png)

## Controlled sequence evidence

The five frozen comparisons remain in their original order. Below are their
existing raw point effects; full four-metric intervals and p-values are in the
immutable DIN report and [bootstrap table](../results/sequence_posthoc_v1/bootstrap_intervals.csv).

| Priority | Left − right | ΔAUC | ΔLogLoss |
|---|---|---:|---:|
| Primary 1 | Mean matched − zero | +0.000330347 | −0.000056164 |
| Primary 2 | Mean matched − mean mismatched | +0.000405708 | −0.000030542 |
| Secondary 3 | Attention matched − zero | +0.000321725 | −0.000058000 |
| Secondary 4 | Attention matched − attention mismatched | +0.000397302 | −0.000032132 |
| Secondary 5 | Attention matched − Mean matched | −0.000008622 | −0.000001836 |

The two primary AUC/LogLoss contrasts pass Holm (joint four-hypothesis p=0.003996).
The 1,000 paired user-cluster draws quantify sampling uncertainty conditional on
the fixed models/day, not independent training uncertainty. All three residual
seeds favor matched over mismatched on AUC, LogLoss, PR-AUC and GAUC. Zero is one
deterministic predictor, not three independent seeds.

Attention-minus-mean AUC/PR-AUC/GAUC intervals include zero. The tiny LogLoss
percentile interval is just below zero, but the predeclared centered p=0.064935;
these statistics are not inverses. Both are retained. This exploratory, practically
tiny result does not justify replacing the preselected Mean matched model.

![Existing sequence intervals](../figures/deep_dive_v1/02_frozen_sequence_intervals.png)

## What the deeper sequence and population analysis adds

At retained length 20, matched beats mismatched in AUC by +0.000474273 but loses
in LogLoss by +0.000019789. Nonclick loss contributes about +0.000615, versus
−0.000595 from clicks: upward probability changes help clicks but slightly hurt
the much larger nonclick group more. Raw histories >=50 account for 58.64% of
the net loss worsening. This explains the arithmetic, not a causal effect of
truncation. Matched still beats zero in capped-group LogLoss.

Of Mean matched's +0.000478364 GAUC increment over zero, about **98.89% comes from
known users**. Lengths 2–5, 6–10 and capped 20 contribute most; length 1 contributes
slightly negatively and empty histories exactly zero. Contributions use existing
whole-user AUCs with the original exposure weights and sum to the frozen total.
They combine group size with group effect, rather than ranking groups by intrinsic
sequence usefulness. No new subgroup significance tests were performed.

![Additive GAUC contributions](../figures/deep_dive_v1/05_gauc_contributions.png)

Global-ranking model order remains different. In every fixed F2 activity bucket,
LR has higher AUC than Mean matched, even though high-activity users account for
most sequence-versus-zero GAUC improvement:

| F2 past-14-day events | Test exposures | LR AUC | Mean matched AUC |
|---|---:|---:|---:|
| 0 | 20,799 | 0.586027 | 0.582605 |
| 1–9 | 30,092 | 0.604783 | 0.601617 |
| 10–99 | 335,026 | 0.637135 | 0.630867 |
| 100–999 | 2,236,989 | 0.652585 | 0.646121 |
| >=1000 | 685,759 | 0.655510 | 0.651809 |

LR also leads in all age codes 1–6 and missing-profile-age rows. The small code-0
cell (1,186 exposures) reverses that point ordering; it is not evidence for
demographic routing. All original gender/age/value/shopping codes, profile presence,
placement, fixed hour blocks and evidence strata remain in the
[D2 group tables](../results/deep_dive_d2_v1/group_summary.csv), including sparse
and single-class cells. No supplied demographic code is relabeled as an inferred
identity or treated as a causal driver.

## Cold vocabulary is not absence of historical information

Mean matched's previously calibrated overall COPC is 0.986346, yet double cold
is 0.581043: 1,024 clicks versus approximately 1,762 expected clicks in 28,331
exposures. A large cell with F2 behavior and F3 ad evidence, but no F3 user
exposures/F4 positive history, accounts for **59.54% of excess expected clicks**.
The cell with none of these four evidence sources accounts for only **0.82%**.
Unknown train IDs can have newly accumulated May 12 feedback; “cold” and “empty”
therefore describe different mechanisms.

These are expected-click accounting shares, not averaged subgroup ratios or
identified causes. Profile/history/evidence-strength partitions all reconcile to
the same total. The saved global isotonic calibration improves LogLoss/ECE but
reduces AUC, PR-AUC and GAUC. It neither solves group bias nor establishes that
group-specific calibration would improve unseen future traffic.

![Loss and cold evidence](../figures/deep_dive_v1/04_loss_and_cold_evidence.png)

![Calibration by cold quadrant](../figures/deep_dive_v1/08_calibration_group_bias.png)

## Audit of common misses and model complementarity

The archived “60% missed by every model” source uses **top-10% quantiles**, not
p>=0.5. Its original arrays are absent, so the exact number is unverifiable.
The script also skips missing models and lacks ID-based array alignment. It must
not be merged numerically with this rebuilt 3.31M-exposure test cohort.

Here five-model common misses are 71.38% at exact top-10% budgets and 99.97% at
p>=0.5. The top-10% union reaches 28.62% recall at **14.61% exposure coverage**,
so it is not an equal-budget ensemble gain. Known/known traffic supplies most
absolute common misses; new-user/known-ad traffic actually has a lower within-
group common-miss rate than known/known. The original broad cold-user claim is
not transferred into the new narrative.

Neural scores are strongly correlated: Mean matched/DeepFM Pearson≈0.99734 and
Wide & Deep/DeepFM≈0.99590. Pairwise error overlap and Spearman/residual correlation
are also saved. A true-label-informed minimum-loss oracle attains approximately
0.178235 versus LR's 0.192429, but requires labels to choose each row's predictor.
It is an optimistic bound, not a demonstrated or deployable ensemble gain.

![Common errors and probability correlation](../figures/deep_dive_v1/03_common_errors.png)

## Validation-only frozen component reliance

D3 uses May 12 only, without training diagnostic models. Neutralizing F3 damages
both LR and DeepFM more than F2. LR AUC is 0.655163 full, 0.644353 without F2 and
0.584946 without F3; DeepFM is 0.650315, 0.637786 and 0.626190 respectively.
Neutralization means numerical train-mean / zero binary indicators, potentially
off-distribution; these are not retrained information ablations.

Removing FM from the fitted DeepFM barely changes AUC (0.650315→0.650292) while
LogLoss worsens (0.189813→0.190702). Removing wide/deep harms both; the branches'
learned offsets and interactions cannot be interpreted as independently trained
standalone models. LR's additive F2 values are constant within each validation-day
user, explaining why removing F2 changes global AUC but leaves GAUC essentially
unchanged. Existing DeepFM epoch curves also document worsening validation despite
falling train loss after epoch 1, without identifying a universal complexity effect.

![Validation-only diagnostics](../figures/deep_dive_v1/06_validation_reliance.png)

## Temporal and repeated-exposure limits

CTR changes nonmonotonically across May 11/12/13 (5.2633%, 4.9299%, 5.0319%).
Validation→test train-vocabulary new-user share increases 9.94%→11.70%, new-ad
share 6.71%→8.77%, while past ad evidence coverage grows 93.29%→97.22%.
History growth partly follows the expanding observed window, not purely changing
behavior. Rolling three-day summaries describe traffic; no rolling-origin model
evaluation or training-day pseudo-held-out predictions are introduced.

Test CTR falls across prior-day user/ad repeat buckets: 5.24% (0), 4.60% (1),
3.40% (2–5), 1.60% (6–19). The >=20 bucket has just 27 rows/no clicks, with
undefined AUC. This association cannot identify fatigue: exposure selection,
survivorship and left-censoring remain. Only two held-out dates are available,
so the data do not establish a stable temporal generalization law.

![Temporal and repeat descriptions](../figures/deep_dive_v1/07_temporal_repeats.png)

## Evidence limits and reproducibility

- Formal test was exposed during earlier baseline development, then frozen;
  statistical methods/deep-dive groups were fixed after aggregate results were
  known. They are not retrospectively preregistered untouched-holdout evidence.
- Existing calibration uses deterministic impression-level validation 80/20,
  not the historical proposal's user-grouped OOF design. It was reused unchanged.
- Four tabular families have one seed each. Three residual seeds share one
  frozen backbone; no independent full-backbone uncertainty is estimated.
- There are no new subgroup p-values, calibrated-model bootstrap, learned
  ensemble, routing rule, or test-dependent model choices.
- Frozen-component interventions are not DNN-only/FM-only retraining. LightGBM
  F1/F12 fits, independent tabular seeds and protocol-matched external retraining
  remain uncompleted historical proposals, outside the current authorization.
- Recorded original inference timing (19.61 s, peak RSS 1.50 GiB for shared
  backbone/batched sequence inference including serialization) is not a comparable
  standalone per-model throughput benchmark. D2/D3/D4 analysis took respectively
  419.53 / 29.90 / 47.06 seconds; these are machine execution, not total project
  development effort. No extra model-compute search was performed.

Stage reports: [D2](DEEP_DIVE_D2_RESULTS.md), [D3](DEEP_DIVE_D3_RESULTS.md),
[D4](DEEP_DIVE_D4_RESULTS.md). Each includes manifests and independent arithmetic
verification. Four new targeted tests cover quotas, oracle accounting, exact
component sums and completed-day repetition. The D3 float32 smoke discrepancy
is archived transparently; original checkpoints remain unchanged and full saved
validation probability parity is verified.

Eight figures are available as PNG and vector PDF under
[`figures/deep_dive_v1`](../figures/deep_dive_v1), with input hashes and plotting
versions in the [figure manifest](../results/deep_dive_final_v1/figure_manifest.json).
Their rendering has been visually inspected. Large predictions/models/intermediate
tables are local/Git-ignored, not backed up by the small committed manifests.
`src/37_verify_frozen_delivery.py` performs read-only hash/link checks without
repeating any statistical estimation, calibration or inference.

The defensible final statement is: **historical feedback dominates the observed
information gain under this protocol; matched sequences add a small conditional
increment, and attention adds no robust practical advantage. That increment is
uneven across history and user groups, does not overturn the strong global LR
baseline, and coexists with substantial cold-group probability bias and changing
traffic. Drift, sparse evidence and exposure repetition limit interpretation;
their causal contributions are not isolated by these diagnostics.**
