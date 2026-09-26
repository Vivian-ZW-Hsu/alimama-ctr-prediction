# Point-in-Time CTR Prediction on E-Commerce Advertising Data

An audited CTR study on 26.56 million Taobao advertising impressions, separating
historical information, model complexity, sequence identity, calibration and
robustness. **The current result is not “DIN wins.”** Under the recorded protocol
and tuning budget, past click-feedback information provides the largest observed
gain. Matched history adds a small increment to a frozen DeepFM, while target
attention has no robust practical advantage over mean pooling.

**Completed:** four tabular families, controlled F4 study, frozen statistical/
calibration evidence, and D2–D5 diagnostics. No further training, bootstrap,
calibration fitting, test inference or new post-hoc analysis is authorized.

- [最终中文 PDF](docs/FINAL_REPORT_ZH.pdf) · [中文文本](docs/FINAL_REPORT_ZH.md)
- [Release audit、实验清单与复现入口](docs/RELEASE_AUDIT.md)
- [Final report and evidence limits](docs/FINAL_REPORT.md)
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
| Mean matched F123+F4 | 0.646263 ± 0.000044 | 0.193120 ± 0.000004 | 0.570380 ± 0.000108 |

Tabular results are one frozen seed each. Sequence mean ± sample std covers three
residual seeds on **one shared frozen backbone**; zero is not three independent
DeepFM fits. GAUC includes 104,884 eligible users / 50.84% of test impressions.
F123 versus F123+F4 is an information change, not a pure architecture comparison.

Metric directions: AUC, PR-AUC and GAUC higher is better; LogLoss, Brier and ECE
lower is better. **COPC = observed clicks / predicted clicks = CTR / mean prediction**;
1 is ideal, below 1 means overprediction, above 1 underprediction. ECE uses the
shared evaluator's 10 equal-count bins; historical calibration selection also
records a separate 20 equal-width-bin diagnostic.

Seeds: tabular/backbone 42; sequence residuals 42, 2026, 3407; F4 deterministic
mismatch 20260926; completed user bootstrap 20260927 (1,000 saved draws).
The hard-zero predictor is a repeated reference to the seed-42 backbone.

The frozen user-cluster bootstrap supports small Mean matched gains over zero
and mismatched history. Mean matched − zero AUC is +0.000330, 95% CI
[+0.000216, +0.000440]. Five ordered comparisons and Holm corrections remain
unchanged. No significance claim is added for sequence versus LR/LightGBM.

Deep-dive findings:

- The old “60% of clicks missed” source used **top-10% exposure-budget thresholds**, not p>=0.5.
  Its exact number cannot be reproduced without old arrays. In the rebuilt
  cohort, five-model common misses are 71.38% at a **top-10% exposure budget per model**; p>=0.5 yields 99.97%.
- Length-20 matched history improves AUC over mismatched but worsens LogLoss:
  added loss on nonclicks slightly exceeds improvement on clicks.
- About 98.89% of the Mean matched versus zero GAUC increment comes from known
  users. Sparse-history benefits are not uniform; length 1 contributes negatively.
- Saved isotonic calibration improves overall LogLoss/ECE but leaves double-cold
  COPC around 0.581. Most excess comes from partially observed histories, not
  wholly absent evidence. Calibration also lowers ranking metrics.
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

Test had been inspected in earlier baseline development; it is a frozen final
reporting set, not a never-seen holdout. New deep-dive buckets were fixed before
their new grouped results, after aggregate test findings were known.

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
Old-project source is preserved on `archive/v1-original`; its numerical claims
are audited separately and are not mixed into rebuilt-pipeline results.

Uncompleted historical proposals remain explicit: LightGBM F1/F12 retraining,
independently retrained DNN/FM components, three independently fitted tabular
backbones, an external-protocol retraining anchor and comparable standalone
throughput benchmarks. The current deep dive does not claim to fill those gaps.
