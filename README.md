# Point-in-Time CTR Prediction on E-Commerce Advertising Data

An audited CTR study on 26.56 million Taobao advertising impressions, separating
historical information, model complexity, sequence identity, calibration and
robustness under a strict point-in-time protocol.

## Summary

**The strongest formal model on global ranking is still logistic regression.** LR
F1+F2+F3 reaches test AUC **0.652489**; no model in this study beats it.

- **Historical feedback, not model complexity, is the main source of performance.**
  F3 completed-day click feedback raises LR test AUC from **0.601790** (F1+F2) to
  **0.652489** (F1+F2+F3), **+0.05070**, against **+0.00630** for the F2 behavior
  aggregates. At the common F1+F2+F3 budget LR − DeepFM AUC is **+0.00655565**,
  95% CI [+0.00596489, +0.00721123], Holm p = 0.001998.
- **F5, the `behavior_log` sequence budget, improves on the frozen null.**
  Mean matched − zero is **+0.00127246 AUC**, 95% CI
  [+0.00104340, +0.00150052], Holm p = 0.003996.
- **It exceeds the sparse F4 clicked-ad source on global ranking**, by
  **+0.00094211 AUC**, 95% CI [+0.00070402, +0.00118202] — but their **GAUC
  difference is not established** (−0.00012937, interval spans zero, p = 0.621).
- **The gain is specific to the user's own history.** Matched − mismatched is
  **+0.00120269 AUC**, 95% CI [+0.00098519, +0.00142260], Holm p = 0.003996. Since
  mismatched − zero is only +0.00007, the length-, activity- and recency-controlled
  control captures about **5.5%** of the effect.
- **Attention beats mean pooling only slightly.** Attention − mean is
  **+0.00023967 AUC**, 95% CI [+0.00017050, +0.00031038] — an interval that
  excludes zero for the first time, unlike the F4 study — but the effect is about
  one fifth of the identity effect and roughly two seed standard deviations.
  **Mean matched remains the preregistered primary model**; the test did not and
  cannot change that.
- **F5 narrows the LR gap without closing it.** The global AUC gap from LR to the
  frozen DeepFM null was 0.006556; F5 reduces it to 0.005283, closing about
  **19.4%**. LR still leads by roughly four times the entire F5 sequence effect.
  On GAUC the ordering reverses, F5 reaching 0.570251 against LR's 0.569939.
- **The current F5 implementation has no deployment advantage.** Its end-to-end
  throughput is **1/3.27** of the DeepFM backbone it sits on (379,078 against
  1,239,580 rows/s at batch 8192), so its per-row compute cost is about **3.27×**,
  and its peak memory is **7.48×** (6,082 against 813 MiB). That memory is the
  resident 723,113,323-event store, **not** the residual's 57,345 parameters.
- **Calibration fixes the overall probability level, not group-level bias.**
  Calibrated F4 mean matched reaches overall COPC **0.986**, while double cold
  start sits at **0.581**, i.e. predicted clicks run about **72% above** observed.

Two sequence sources were tested: past clicked-ad categories (F4, test mean 4.69
retained tokens, 30.21% empty) and `behavior_log` category-plus-type events (F5,
test mean 185.19 tokens, 0.4351% empty). **The result is not "DIN wins."** The
richer source produces a clearly larger but still small global-ranking gain and no
measurable within-user gain, at a substantially higher serving cost.

**Status:** complete. Four tabular families, the controlled F4 study, D2–D5
diagnostics, and the user-authorized E1–E6 extension — cross-source namespace
audit, frozen-model bootstrap, documentation scope corrections, the F5 behavior
sequence budget with its once-only formal test, the unified serving benchmark and
the experiment registry — are all frozen. See [AGENTS.md](AGENTS.md).

- [最终中文 PDF](docs/FINAL_REPORT_ZH.pdf) · [中文文本](docs/FINAL_REPORT_ZH.md)
- [Release audit、实验清单与复现入口](docs/RELEASE_AUDIT.md)
- [Final report and evidence limits](docs/FINAL_REPORT.md)
- [E1: cross-source namespace audit and F5 admissibility](docs/BEHAVIOR_NAMESPACE_AUDIT.md)
- [E3: frozen tabular user-cluster bootstrap](docs/FROZEN_MODEL_BOOTSTRAP.md)
- [E5: F5 behavior-sequence validation and freeze](docs/F5_BEHAVIOR_VALIDATION.md)
- [E5: F5 behavior-sequence formal test results](docs/F5_BEHAVIOR_TEST_RESULTS.md)
- [E6: unified serving benchmark](docs/SERVING_BENCHMARK.md)
- [Experiment index](docs/EXPERIMENT_INDEX.md)
- [Frozen DIN test/statistical/calibration report — badecf4](docs/DIN_F4_TEST_RESULTS.md)
- [Frozen deep-dive plan](docs/DEEP_DIVE_RESEARCH_PLAN.md)
- [D2: errors, sequence heterogeneity and cold evidence](docs/DEEP_DIVE_D2_RESULTS.md)
- [D3: validation-only information/component reliance](docs/DEEP_DIVE_D3_RESULTS.md)
- [D4: descriptive temporal drift and repeats](docs/DEEP_DIVE_D4_RESULTS.md)

![Information layers and frozen model comparison](figures/deep_dive_v1/01_information_and_models.png)

## Main evidence

| Model / input | Test AUC | Test LogLoss | Test GAUC |
|---|---:|---:|---:|
| LR F1 | 0.595495 | 0.196781 | 0.561913 |
| LR F1+F2 | 0.601790 | 0.196376 | 0.561534 |
| LR F1+F2+F3 | 0.652489 | 0.192429 | 0.569939 |
| LightGBM F123 | 0.651318 | 0.192502 | 0.569889 |
| Wide & Deep F123 | 0.646219 | 0.193161 | 0.569680 |
| DeepFM F123 / deterministic zero reference | 0.645933 | 0.193176 | 0.569902 |
| F4 mean matched F123+F4 | 0.646263 ± 0.000044 | 0.193120 ± 0.000004 | 0.570380 ± 0.000108 |
| **F5 mean matched F123+F5** | **0.647206 ± 0.000060** | **0.193061 ± 0.000031** | 0.570251 ± 0.000239 |
| F5 attention matched F123+F5 | 0.647445 ± 0.000077 | 0.193043 ± 0.000029 | 0.570476 ± 0.000054 |

Tabular results are one frozen seed each. Sequence mean ± sample std covers three
residual seeds on **one shared frozen backbone**; zero is not three independent
DeepFM fits. GAUC includes 104,884 eligible users / 50.8437% of test impressions.
F123 versus F123+F4 is an information change, not a pure architecture comparison.

Metric directions: AUC, PR-AUC and GAUC higher is better; LogLoss, Brier and ECE
lower is better. **COPC = observed clicks / predicted clicks = CTR / mean prediction**;
1 is ideal, below 1 means overprediction, above 1 underprediction. A COPC of
0.581 means predicted clicks are about 72% above observed clicks
(`1/0.581 − 1`), which is a different quantity from the 42% shortfall of observed
against predicted (`1 − 0.581`). ECE uses the shared evaluator's 10 equal-count
bins; historical calibration selection also records a separate 20
equal-width-bin diagnostic.

Seeds: tabular/backbone 42; sequence residuals 42, 2026, 3407; F4 deterministic
mismatch 20260926; completed F4 user bootstrap 20260927 and frozen tabular
bootstrap 20260926 (1,000 saved draws each).
The hard-zero predictor is a repeated reference to the seed-42 backbone.

### Uncertainty on the frozen tabular ordering

| Comparison | Metric | Effect | 95% CI | Excludes 0 |
|---|---|---:|---|:--:|
| LR − DeepFM (primary) | AUC | +0.00655565 | [+0.00596489, +0.00721123] | yes |
| LR − DeepFM (primary) | LogLoss | −0.00074793 | [−0.00083617, −0.00067152] | yes |
| LR − LightGBM | LogLoss | −0.00007366 | [−0.00014903, +0.00000767] | **no** |
| LR − LightGBM | GAUC | +0.00004970 | [−0.00124507, +0.00127498] | **no** |
| LR − Wide & Deep | GAUC | +0.00025893 | [−0.00079951, +0.00140635] | **no** |

Intervals are conditional on the frozen fitted models and the single frozen test
day. They estimate user-sampling uncertainty, **not** training-seed uncertainty:
each tabular family has exactly one seed. Full table and limits in
[FROZEN_MODEL_BOOTSTRAP.md](docs/FROZEN_MODEL_BOOTSTRAP.md).

### Completed F4 sequence evidence, with its scope stated

The frozen user-cluster bootstrap supports small Mean matched gains over zero and
mismatched history. Mean matched − zero AUC is +0.000330, 95% CI
[+0.000216, +0.000440]. Five ordered comparisons and Holm corrections remain
unchanged. No significance claim is added for sequence versus LR/LightGBM.

These results describe **one specific history representation**: the ordered last
20 categories of the user's past clicked ads, on completed Beijing dates strictly
before the impression date. On the test day that source retains a mean of 4.69
tokens and leaves 30.21% of impressions with no history at all. The conclusions
are not transferred to full `behavior_log` `pv/cart/fav/buy` sequences, and they
do not establish a general result about sequence modeling or about DIN.

### Published Alimama results as background, not as a reproduction

The DSIN paper ([arXiv:1905.06482](https://arxiv.org/abs/1905.06482), IJCAI 2019)
reports on the same public advertising dataset with the same calendar split
(train 2017-05-06…05-12, test 2017-05-13) and the users' most recent 200
`behavior_log` events:

| Model | Reported AUC |
|---|---:|
| YoutubeNet without user behavior | 0.6239 |
| YoutubeNet (average pooling) | 0.6313 |
| Wide & Deep | 0.6326 |
| DIN | 0.6330 |
| DIEN | 0.6343 |
| DSIN | 0.6375 |

In that table, introducing a user behavior sequence corresponds to about
**+0.0074** AUC (0.6313 − 0.6239) and DIN over the average-pooling model to about
**+0.0017** (0.6330 − 0.6313). This project observes **+0.00033** and
**−0.0000086** for the analogous contrasts on a mean-4.69-token clicked-ad
category sequence.

Because the sequence source, split, feature budget, backbone and training
procedure all differ, **this is background context and not a direct
reproduction**. The published values are single reported point estimates and are
not described here as significant. The order-of-magnitude gap — roughly 22× on
the sequence-presence contrast — is consistent with the sequence source and its
information density limiting the observed F4 effect, which is the hypothesis F5
tests rather than a conclusion already established.

## Deep-dive findings

- The old "60% of clicks missed" source used **top-10% exposure-budget thresholds**, not p>=0.5.
  Its exact number cannot be reproduced without old arrays. In the rebuilt
  cohort, five-model common misses are 71.38% at a **top-10% exposure budget per model**; p>=0.5 yields 99.97%.
- Length-20 matched clicked-ad history improves AUC over mismatched but worsens
  LogLoss: added loss on nonclicks slightly exceeds improvement on clicks.
- About 98.89% of the Mean matched versus zero GAUC increment comes from known
  users. Sparse-history benefits are not uniform; length 1 contributes negatively.
- Saved isotonic calibration improves overall LogLoss/ECE but leaves double-cold
  COPC around 0.581, about 72% predicted-over-observed. Most excess comes from
  partially observed histories, not wholly absent evidence.
- Isotonic calibration is a monotone non-decreasing step function. It introduces
  **ties** rather than new inversions, so ranking metrics **can move in either
  direction**: collapsing correctly ordered pairs into ties lowers AUC while
  collapsing incorrectly ordered pairs raises it. In the saved results AUC,
  PR-AUC and GAUC decreased. `ece_bins=20` is a reliability-binning parameter for
  ECE, not a hyperparameter of `IsotonicRegression`, whose steps are determined
  by PAVA.
- Frozen validation diagnostics show greater reliance on F3 than F2. Removing
  DeepFM's FM term barely changes AUC but changes probability level; this is a
  **sensitivity diagnostic** (frozen component deletion), not a retrained ablation.
- New-ID shares, history availability and repeats change across dates. Repeat
  count is associated with lower CTR. Both are **descriptive associations**, without identifying a fatigue effect.

## Fixed data and information protocol

| Split | Beijing dates | Impressions |
|---|---|---:|
| Train | 2017-05-06–11 | 20,015,245 |
| Validation | 2017-05-12 | 3,234,051 |
| Test | 2017-05-13 | 3,308,665 |

Raw IDs are joined before train-only encoding/scaling; no negative downsampling.
F1 contains static/context fields, excluding weekday in formal models. F2 contains
completed-day 1/3/7/14-day behavior summaries. F3 contains completed-day smoothed
user/ad/category/campaign/customer/brand feedback. F4 is the ordered last 20
categories from positive exposures on Beijing dates **strictly before D**.
Exposure timestamps proxy positive-feedback record time; feedback is assumed
available next day. There is no independent click timestamp. Validation excludes
May 12 feedback; test may use completed May 12 feedback. PAD/UNKNOWN and empty
sequence representations are neutral zeros.

The first training date, 2017-05-06, has no completed prior date inside the
impression window, so all F3 entity histories on that date fall back to the fixed
prior with `has_past_feedback = 0`. Those **3,270,348 impressions (16.3% of
training) were retained, not dropped**.

The behavior log's usable window is **2017-04-22 through 2017-05-13**: 22 dates
carrying 99.9786% of all events. Earlier timestamps are sporadic or corrupt
(127,136 events spread over 653 dates, from 1902 onward), so the effective
pre-impression history is 14 completed days before 2017-05-06, not a longer
window. Full-population evidence is in
[BEHAVIOR_NAMESPACE_AUDIT.md](docs/BEHAVIOR_NAMESPACE_AUDIT.md).

Test had been inspected in earlier baseline development; it is a frozen final
reporting set, not a never-seen holdout. New deep-dive buckets were fixed before
their new grouped results, after aggregate test findings were known.

Probability calibration was selected on a deterministic **impression-level** 80/20
split of validation, not the historically proposed user-grouped OOF design. The
calibrator receives only predicted probability and label — never `user_raw` — so
this is **not** a case of the calibrator memorising a user's labels. The actual
risk is sample dependence: a user's impressions are not independent, so splitting
by impression makes the fit and selection sets correlated and can make the
uncertainty of *method selection* look optimistic. A user-grouped split would be
the right design for judging calibration transfer to entirely new users.

### How to read the uncertainty in this report

Three distinct quantities appear and are never merged.

1. **Bootstrap intervals are conditional on the frozen models and their saved
   predictions.** They resample test users, so they answer "would this difference
   survive a different sample of users on this day?" They are **not** training
   randomness and do not answer "would it survive retraining?".
2. **Seed standard deviations describe training randomness of the residual only.**
   F4 and F5 sequence conditions use three residual seeds (42, 2026, 3407) on **one
   shared frozen DeepFM backbone**. The four tabular families have **one seed each**,
   so their ordering carries no training-randomness estimate at all.
3. **The zero control's standard deviation of zero is structural.** It is one
   deterministic predictor on the shared backbone, evaluated three times — not three
   independently fitted models — so its three references are identical by
   construction.

Because every sequence condition shares one backbone, no contrast among them is a
full-model comparison.

## Where the F5 gain lands

Effects are F5 mean matched minus the zero null, on raw probabilities,
seed-averaged. These are **descriptive subgroup differences with no intervals and
no multiplicity control**; small cells carry a direction, not evidence.

- **The gain is largest exactly where the sparse source was empty.** On the
  **985,688** test impressions with no clicked-ad history — about **29.8%** of the
  test day — F5 − zero is **+0.001607**, against +0.001163 where both sources have
  history. Both empty-history cells are exactly zero, as the architecture requires.
- **New users with known ads gain most**, **+0.002785** on 358,678 rows, where
  behavior history partly substitutes for an unseen user ID.
- **Double cold start is negative**, **−0.001741**, but on only **28,331** rows with
  no interval. The direction is reported and deliberately not interpreted.
- **Donor bucket relaxation is irrelevant to the conclusions.** It touches
  **1,457** rows (0.0440%); excluding them moves matched − mismatched by at most
  **9.0e-7** on AUC and 5.0e-6 on GAUC, with every sign preserved.
- **A high UNKNOWN token share does not degrade the gain**: +0.001113 at zero
  UNKNOWN and **+0.001526** above 0.20. The **4.30%** token mask is therefore not
  the binding constraint on F5 today.

## Engineering: report both regimes

Two measurement regimes are reported for every model, because the gap between them
changes the deployment conclusion.

| Regime | What is timed |
|---|---|
| **pure** | inputs already prepared in memory; only the model call |
| **end_to_end** | Parquet batch read, encoding, scaling, sequence gather, model call |

Logistic regression reaches **18,083,051 rows/s** pure at batch 8192 but only
**1,275,056** end to end — a **14.2×** gap caused almost entirely by building its
**2,497,016**-dimension sparse row. Any deployment conclusion drawn from the pure
number alone would be wrong: the feature-construction step, not the model, is what
would need optimising first.

The same distinction reframes F5. Its residual is only 57,345 parameters, so its
*model* cost is negligible; the 7.48× memory and 3.27× per-row compute come from
the surrounding data path.

## Limitations and future work

Carried forward deliberately, not resolved:

- **Comparison 6 has no subgroup decomposition.** Group metrics were computed for
  the five F5 conditions; the F4 reference has overall metrics only, so the
  F5-versus-F4 contrast is not broken down by cold-start quadrant, history length or
  any other partition.
- **LightGBM node count is not reported.** `num_trees` is 251 and the model file is
  363 MiB, but a full `dump_model` of that booster costs more memory than the number
  is worth.
- **The tabular families have one seed each.** Their intervals hold the fitted
  parameters fixed, so nothing here separates their ordering from training
  randomness. F4 and F5 sequence conditions use three seeds.
- **F5 is a sequence residual on a frozen DeepFM backbone, not a jointly trained
  DIN or DSIN reproduction.** No contrast in this study separates the residual's
  contribution from the backbone's, and the published Alimama analogue for
  introducing a behavior sequence is about +0.0074 AUC against F5's +0.00127.
- **The completed-day cutoff is deliberately conservative.** Behavior events carry
  usable timestamps, so a per-event cutoff is possible; it is not used, to keep F5
  comparable with F2/F3/F4 and to avoid a new feedback-availability assumption.
- **F5's serving memory is an artifact of this implementation.** Holding the
  723,113,323-event store resident costs 6,082 MiB. Streaming reads, memory mapping,
  a tiered cache or an offline feature service would each reduce it without touching
  the model, so 6,082 MiB is not a lower bound for the budget.
- **Source, length and training regime are not separately identified.** F5 changes
  the sequence source, the retained length and the token schema at once. Its results
  support the sequence source and its information density being an important
  explanation for F4's smaller effect; they do **not** isolate the causal
  contribution of source, length, token fields or joint training individually.

## Formal test inference count

**Formal test inferences: 38 in total**, each a single once-only pass, verified
against the manifests, the once-only markers and the registry:

| Source | Passes |
|---|---:|
| F4 clicked-ad sequence conditions | 15 |
| F5 behavior sequence conditions | 15 |
| LR information layers (F1, F1+F2, F1+F2+F3) | 3 |
| Tabular benchmark families (LightGBM, Wide & Deep, DeepFM at F123) | 3 |
| Superseded-name early LR runs (`logistic_regression`, `logistic_regression_no_weekday`) | 2 |
| **Total** | **38** |

Earlier text quoted "38 (F4 15 + F5 15)", which does not add up. The remaining 8 are
the tabular families and the two superseded-name LR runs, each holding exactly one
test prediction file.

## Repository and reproducibility

| Location | Contents |
|---|---|
| `src/01`–`src/12` | audited base, feature, encoding and shared evaluation pipeline |
| `src/13`–`src/18` | completed tabular selection/training/calibration code |
| `src/19`–`src/27` | F4 audit, frozen residual study and once-only test |
| `src/28`–`src/32` | completed frozen bootstrap/calibration/reporting |
| `src/33`–`src/35` | D2 saved-score, D3 validation reliance, D4 descriptive diagnostics |
| `src/36_plot_deep_dive.py` | eight PNG/PDF figures from small saved summaries |
| `src/37_verify_frozen_delivery.py` | read-only delivery/hash verification |
| `src/40`–`src/41` | E1 lossless behavior-log conversion and namespace audit |
| `src/42`–`src/43` | E3 frozen tabular bootstrap preparation and execution |
| `src/44`–`src/45` | E4 F5 behavior-sequence build and independent audit |
| `src/46`–`src/48` | E5 F5 context cache, residual training and validation freeze |
| `src/49`–`src/50` | E6 serving benchmark and experiment registry |
| `src/project_environment.py` | refuses any formal stage outside the tested interpreter |
| `results/` | versioned metrics, manifests and audits |
| `figures/deep_dive_v1/` | standalone PNG and vector PDF figures |
| `data/`, large model/prediction artifacts | local, Git-ignored; not backed up by Git |

Run commands from the repository root. **Completed analysis entry points refuse
overwriting. Do not rerun old training, bootstrap, calibration or test scripts as
setup instructions.** The once-only formal inference marker remains intact.

Read-only checks, with local ignored artifacts available:

```bash
python src/37_verify_frozen_delivery.py
python src/38_release_audit.py
python -m unittest discover -s tests -v
git diff --check
```

All tests use small synthetic fixtures, not the formal train/validation/test data.
The release audit reads saved summaries and hashes only. Local data layout and
publication-only PDF commands are documented in [RELEASE_AUDIT.md](docs/RELEASE_AUDIT.md).

Core dependencies are in `requirements.txt`; exact analyzed versions are in
`requirements-tested.txt` and manifests. Figures used a separate existing
environment with `requirements-plot.txt`, without changing the model environment.
Historical LR checkpoints were serialized with sklearn 1.8.0; the diagnostic
loader records its compatibility alias and validates saved-score parity.
Dependency pins alone do not guarantee cross-version checkpoint portability.

The [charter](docs/CTR_CODEX_PROJECT_HANDOFF.md) and historical
[proposal](docs/EXPERIMENTAL_PROTOCOL.md) preserve original intent and deviations.
Current instructions in [AGENTS.md](AGENTS.md) take precedence over stale gates.
Frozen evidence documents keep their original wording; scope corrections are made
in this README and the final reports, not by editing frozen records.
Old-project source is preserved on `archive/v1-original`; its numerical claims
are audited separately and are not mixed into rebuilt-pipeline results.

Uncompleted historical proposals remain explicit: LightGBM F1/F12 retraining,
independently retrained DNN/FM components, three independently fitted tabular
backbones, an external-protocol retraining anchor and comparable standalone
throughput benchmarks. The current deep dive does not claim to fill those gaps.
