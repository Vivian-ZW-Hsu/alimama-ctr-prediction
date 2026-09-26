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
no new calibration was fitted or selected in this test batch.

## Test results

Each entry is arithmetic mean ± sample std (ddof=1) across the three fixed seeds.
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
User-cluster paired bootstrap has not been performed in this batch. The claim
is a small conditional incremental gain on this frozen backbone and test day,
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
