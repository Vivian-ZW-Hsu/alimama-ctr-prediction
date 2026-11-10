# Stage E3: paired user-cluster bootstrap of the frozen tabular models

## What this stage does and does not claim

The four F123 tabular families were reported as single frozen runs with no
uncertainty estimate, so the documented ordering — LR first, LightGBM close,
Wide & Deep and DeepFM behind — rested on point differences alone. This stage
attaches intervals to those differences using the **already saved** test
predictions.

No model is retrained. No test inference is run. The existing frozen predictions
are bound by SHA256 and read only.

> **These intervals are conditional on the four frozen fitted models and the
> single frozen test day. They estimate user-sampling uncertainty. They are not
> an estimate of training-seed uncertainty: each tabular family has exactly one
> seed, so re-training variability is not represented at all.**

Consequently this stage can answer "would this difference survive a different
sample of users?" and cannot answer "would LR still beat DeepFM if both were
retrained with different seeds?".

## Pre-registration

`results/frozen_model_bootstrap_v1/protocol.json` and `input_manifest.json` were
written by `src/42_prepare_frozen_model_bootstrap.py` and **committed in
`29659c3`, before `src/43_bootstrap_frozen_models.py` was allowed to run**.
`src/43` refuses to start unless both files are tracked and free of uncommitted
changes, so the hypothesis set cannot be chosen after an interval is visible.

| Role | Comparison | Metrics | Correction |
|---|---|---|---|
| Confirmatory primary | LR − DeepFM | AUC, LogLoss | Holm over these two |
| Exploratory secondary | LR − LightGBM | AUC, LogLoss, PR-AUC, GAUC | none |
| Exploratory secondary | LR − Wide & Deep | AUC, LogLoss, PR-AUC, GAUC | none |

Bootstrap settings: 1,000 replicates, exact multinomial resampling of whole
`user_raw` clusters over all 391,741 test users, seed 20260926, marginal
percentile 95% intervals (not simultaneous), centered two-sided p-values with
resolution 1/1001 ≈ 0.000999.

## Verification before any interval was read

- All bound input hashes re-verified; a changed frozen prediction aborts the run.
- The unweighted replicate (unit weights) reproduces the documented point metrics
  for all four families to within 5e-7 on AUC/LogLoss and 1e-9 on GAUC.
- Preparation independently recomputed GAUC and matched the documented values to
  rounding: LR 0.569939, LightGBM 0.569889, Wide & Deep 0.569680,
  DeepFM 0.569902, with 104,884 eligible users covering 50.8437% of 3,308,665
  impressions.
- The 5-replicate pilot is a bit-exact prefix of the full 1,001-replicate run.
- A non-finite replicate aborts the run rather than being dropped, because
  discarding single-class draws would bias the interval.

## Results

Effects are **left minus right**; metric orientation is never flipped, so a
negative LogLoss effect means the left model has the lower loss.

### Confirmatory primary: LR − DeepFM

| Metric | Effect | 95% CI | Centered p | Holm p | Excludes 0 |
|---|---:|---|---:|---:|:--:|
| AUC | **+0.00655565** | [+0.00596489, +0.00721123] | 0.000999 | **0.001998** | yes |
| LogLoss | **−0.00074793** | [−0.00083617, −0.00067152] | 0.000999 | **0.001998** | yes |

LR's advantage over DeepFM on both global ranking and probability quality is
stable under user resampling at the smallest resolvable p-value.

### Exploratory secondary

| Comparison | Metric | Effect | 95% CI | Centered p | Excludes 0 |
|---|---|---:|---|---:|:--:|
| LR − LightGBM | AUC | +0.00117059 | [+0.00059116, +0.00173101] | 0.000999 | yes |
| LR − LightGBM | LogLoss | −0.00007366 | [−0.00014903, **+0.00000767**] | 0.059940 | **no** |
| LR − LightGBM | PR-AUC | +0.00147035 | [+0.00086458, +0.00211197] | 0.000999 | yes |
| LR − LightGBM | GAUC | +0.00004970 | [−0.00124507, +0.00127498] | **0.941059** | **no** |
| LR − Wide & Deep | AUC | +0.00626926 | [+0.00568570, +0.00689256] | 0.000999 | yes |
| LR − Wide & Deep | LogLoss | −0.00073229 | [−0.00081901, −0.00065863] | 0.000999 | yes |
| LR − Wide & Deep | PR-AUC | +0.00172213 | [+0.00127618, +0.00221330] | 0.000999 | yes |
| LR − Wide & Deep | GAUC | +0.00025893 | [−0.00079951, +0.00140635] | **0.652348** | **no** |

## What the intervals change in the narrative

**1. The global ranking order is no longer only a point difference.** LR over
DeepFM excludes zero on both preregistered metrics (AUC, LogLoss); LR over
Wide & Deep excludes zero on AUC, LogLoss and PR-AUC. Under
user resampling of this frozen test day, the neural families do not reach the
linear baseline on global ranking or probability quality.

**2. LR's global AUC advantage does not transfer to within-user ranking.** Both GAUC
differences that were tested include zero (LR − DeepFM GAUC was not part of
the preregistered set), with p = 0.941 against LightGBM and
p = 0.652 against Wide & Deep. The existing qualitative statement that the four
families have "very close" GAUC is now quantified: the GAUC intervals are
consistent with no difference at all, while the AUC intervals are not. Global AUC
mixes cross-user pairs; GAUC evaluates only the 50.8437% of impressions belonging
to the 104,884 users that have both labels.

**3. LR versus LightGBM is a ranking difference, not a probability difference.**
The AUC and PR-AUC intervals exclude zero, but the LogLoss interval contains zero
(p = 0.060). Calling LightGBM worse overall would overstate the evidence; its
probability quality is indistinguishable from LR's here.

## Limits

- Single-seed families: the intervals hold the fitted parameters fixed. A
  training-seed comparison would require retraining, which is out of scope.
- One held-out day: the resampling unit is the user, not the day, so nothing here
  speaks to temporal stability.
- Intervals are marginal per metric, not simultaneous across metrics; only the
  two primary metrics carry a Holm correction.
- The primary p-values sit at the resolution floor 1/1001, so "p = 0.000999"
  means "no replicate was at least as extreme", not a precise magnitude.
- Metric values are raw, not calibrated. Calibration changes LogLoss and can
  change ranking metrics through tie creation; that is a separate analysis.

## Reproduction

```bash
python -B src/42_prepare_frozen_model_bootstrap.py   # refuses to overwrite
git add results/frozen_model_bootstrap_v1 && git commit   # protocol gate
python -B src/43_bootstrap_frozen_models.py
python -B -m unittest tests.test_frozen_model_bootstrap -v
```

Evidence: [`protocol.json`](../results/frozen_model_bootstrap_v1/protocol.json),
[`input_manifest.json`](../results/frozen_model_bootstrap_v1/input_manifest.json),
[`bootstrap_pilot.json`](../results/frozen_model_bootstrap_v1/bootstrap_pilot.json),
[`bootstrap_intervals.csv`](../results/frozen_model_bootstrap_v1/bootstrap_intervals.csv),
[`bootstrap_deltas.csv`](../results/frozen_model_bootstrap_v1/bootstrap_deltas.csv),
[`bootstrap_manifest.json`](../results/frozen_model_bootstrap_v1/bootstrap_manifest.json).
The 1,000 per-replicate deltas are committed, so every interval and p-value in
this document can be recomputed without rerunning the kernel.
