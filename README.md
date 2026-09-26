# Point-in-Time CTR Prediction on E-Commerce Advertising Data

A leakage-free CTR prediction and evaluation system built on the Alibaba Taobao Display Advertising Dataset.

The project separates information gain, model-family gain, sequence gain, calibration, cold-start robustness, temporal robustness, and engineering cost.

## Data Protocol

All date boundaries use Beijing time (`Asia/Shanghai`).

| Split | Beijing dates | Purpose |
|---|---|---|
| Train | 2017-05-06 to 2017-05-11 | model fitting |
| Validation | 2017-05-12 | model selection and diagnosis |
| Test | 2017-05-13 | frozen final reporting |

Main experiments use all 26,557,961 impressions without negative downsampling.

Point-in-time rules:

- joins use raw IDs;
- encoding occurs only after joins;
- encoders and scalers are fitted on training data only;
- behavior features use completed dates before the impression date;
- current and future labels are never used as features;
- identity checks verify that histories belong to the correct raw user.

## Current Pipeline

Historical pipeline entry points (run from the repository root). Existing
formal artifacts are immutable; do not rerun completed models as a setup step.

```bash
python src/01_validate_raw.py
python src/02_profile_raw.py
python src/03_build_base_tables.py
python src/04_build_daily_behavior.py
python src/05_build_behavior_features.py
python src/05b_check_behavior_identity.py
python src/06_profile_model_features.py
python src/07_build_model_inputs.py
python src/08_train_logistic_regression.py
python src/09_diagnose_lr_calibration.py
```

| Step | Purpose |
|---|---|
| 01 | validate raw schemas, labels, keys, and timestamps |
| 02 | profile missing values, prices, and behavior dates |
| 03 | join static tables and create Beijing-time splits |
| 04 | aggregate behavior by raw user and Beijing date |
| 05 | construct leakage-free 1/3/7/14-day behavior features |
| 05b | verify user identity in behavior features |
| 06 | define and profile the model feature schema |
| 07 | build train-only encodings and model inputs |
| 08 | train the out-of-core Logistic Regression baseline |
| 09 | diagnose ranking and probability calibration |
| 10 | shared evaluator; `--splits val` supports selection before the test gate |
| 11–12 | build completed-day click feedback F3 and common F123 inputs |
| 14 | validation-selected probability calibration |
| 15–16 | completed LightGBM, Wide & Deep and DeepFM baselines |
| 19–20 | audited daily positive-click category histories and mismatch controls |
| 21–22 | frozen DeepFM context cache and validation-only sequence residual training |
| 23–24 | preregistered paired selection, checkpoint freeze and validation audit |

## Repository Layout

```text
data/                 local data; ignored by Git
src/                  pipeline and training entry points
artifacts/            encoders, models, and predictions
results/              small metric and validation reports
docs/                 experimental protocol
requirements.txt      implemented dependencies
```

Large datasets, mapping tables, trained models, and predictions are not committed. Small manifests, metrics, and diagnostic tables are committed.

## Information Layers

- **F1 — Static/context:** user, ad, category, campaign, advertiser, brand, placement, user profile, price, hour, and validated calendar features.
- **F2 — Historical behavior:** leakage-free page-view, favorite, cart, purchase, and recency features over 1/3/7/14-day windows.
- **F3 — Past-only click feedback:** smoothed historical click statistics using only dates before the prediction date.
- **F4 — Positive-click category sequence:** ordered last 20 categories from positive exposures on completed Beijing dates strictly before the target date. Exposure timestamp proxies the positive-record time; feedback is assumed available next day. No independent click timestamp is available. Train-only category vocabulary, neutral PAD/UNKNOWN, and exact-zero empty-history residuals.

## Models

- Logistic Regression
- LightGBM
- Wide & Deep
- DeepFM
- DIN-style target attention and mean pooling on frozen DeepFM context

LR has F1/F2/F3 ablations. The four completed tabular families share F123 for
their model-family comparison; LightGBM F1/F12 ablations remain outstanding.
The sequence study adds F4 through a trainable residual on the existing frozen
DeepFM checkpoint, comparing mean/attention and matched/mismatched histories.
It tests incremental sequence information, not a reproduction of original DIN
or a new fair-input tabular architecture ranking.

## Evaluation

Core evaluation includes LogLoss, AUC, GAUC, PR-AUC, COPC, reliability curves, cold-start segments, model size, and inference throughput.

Post-training Platt scaling and isotonic regression are fitted on validation predictions and evaluated on the frozen test date. Calibration improvement is reported separately from ranking improvement.

The long-term charter is [`CTR_CODEX_PROJECT_HANDOFF.md`](docs/CTR_CODEX_PROJECT_HANDOFF.md).
The historical [`EXPERIMENTAL_PROTOCOL.md`](docs/EXPERIMENTAL_PROTOCOL.md) contains
deviations documented in the [F4 design audit](docs/DIN_F4_DESIGN.md).

## Current Status

F123 preprocessing, four tabular families, shared evaluation, calibration and
cold-start diagnostics are complete for the existing formal runs. The completed
models are not being retuned. The F4 full build passed exact ID/label, time,
identity, deterministic mismatch and original-source audits on 26,557,961 rows.
Sequence training and selection use train/validation only; **the first formal
sequence test requires separate approval**. See the [execution record](docs/DIN_F4_EXECUTION.md)
and [preregistered selection rules](results/sequence_selection_v1/preregistration.json).
Validation-only selection is complete: L=20, LR=5e-4, 12 frozen checkpoints across
four conditions and three seeds. [Results and the first-test gate](docs/DIN_F4_VALIDATION.md)
document the small matched-history gain and the lack of clear attention advantage.
Large new caches, checkpoints and predictions remain local and Git-ignored;
committed manifests bind them by SHA256.

Previous outputs are preserved on the `archive/v1-original` branch and are not used as evidence for this rebuilt pipeline.
