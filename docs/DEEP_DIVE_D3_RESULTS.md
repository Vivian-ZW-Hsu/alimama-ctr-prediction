# D3: Validation-only information and component reliance

This stage uses frozen LR/DeepFM parameters and **May 12 only**. No CTR model was
trained; no calibration was fitted; no test prediction was produced. All 13 fixed
variants are retained, with no winning variant selected. The plan in e494030
predates these diagnostic outputs. Frozen model/checkpoint hashes are unchanged.

## Existing trained LR information layers

The saved LR runs share alpha=1e-7, eta0=0.005, batch=200000, seed 42 and at most
three epochs. Best validation epochs differ: F1/F12 use 3, F123 uses 2. Thus these
are common-recipe, validation-stopped information ablations, not identical fixed-
epoch fits. The earlier validation hyperparameter-selection deviation remains
recorded in the project charter; no selection is redone here.

| Saved run | Validation AUC | Validation LogLoss |
|---|---:|---:|
| lr_f1 | 0.601207399 | 0.193556758 |
| lr_f1_f2 | 0.608416325 | 0.193055397 |
| lr_f1_f2_f3 | 0.655162945 | 0.189468972 |

The largest saved LR increment comes from F3. This is evidence in the observed
LR ablation, not proof that every learner or dataset must have the same ordering.

## Frozen-parameter reliance and components

F2/F3 neutralization replaces numerical standardized values with zero (the
training mean), and binary indicators with zero. F1 is unchanged. This can be
an inconsistent/off-distribution input, especially when a mean count coexists
with a zero availability flag. It measures reliance of the already fitted model,
not performance of a freshly trained reduced-information model.

DeepFM logits are decomposed as wide + deep + FM; removed components are not
refitted and remaining biases are not recalibrated. The deep-only diagnostic
therefore is **not a DNN-only retraining**. Its large LogLoss includes the missing
shared logit offset and cannot identify an architecture's achievable performance.

| Fixed validation diagnostic | AUC | LogLoss | GAUC | COPC |
|---|---:|---:|---:|---:|
| DeepFM_full | 0.650315144 | 0.189812537 | 0.576213071 | 0.956963 |
| DeepFM_no_F2 | 0.637786356 | 0.194177464 | 0.575353305 | 0.721876 |
| DeepFM_no_F3 | 0.626189696 | 0.215663872 | 0.567185016 | 0.446124 |
| DeepFM_no_F2_F3 | 0.620289116 | 0.246255545 | 0.567210184 | 0.325037 |
| DeepFM_no_wide | 0.627501549 | 0.289733879 | 0.550463764 | 0.246650 |
| DeepFM_no_FM | 0.650292095 | 0.190701870 | 0.575997305 | 0.828242 |
| DeepFM_no_deep | 0.588272016 | 0.246218036 | 0.553237952 | 0.330428 |
| DeepFM_wide_only | 0.581976073 | 0.264535910 | 0.552283364 | 0.288162 |
| DeepFM_deep_only | 0.619838656 | 0.318113013 | 0.540777947 | 0.216880 |
| LR_full | 0.655162944 | 0.189468971 | 0.576481774 | 1.099252 |
| LR_no_F2 | 0.644353325 | 0.190886469 | 0.576481774 | 1.189284 |
| LR_no_F3 | 0.584946398 | 0.200596457 | 0.569724824 | 1.710301 |
| LR_no_F2_F3 | 0.583330570 | 0.202999525 | 0.569724824 | 1.878146 |

F3 neutralization harms both frozen models more than F2 neutralization in AUC and
LogLoss. In LR, removing F2 leaves GAUC essentially unchanged: these additive
user-history features are constant within a user on the validation date, so they
can change cross-user ordering without changing within-user candidate ordering.
DeepFM can interact those dense features with candidate features, so its within-
user ranking can change. This gives a concrete mechanism for distinguishing
population-level ranking from within-user ranking.

Removing FM changes DeepFM validation AUC only slightly (0.650315 → 0.650292),
while LogLoss moves 0.189813 → 0.190702 and COPC 0.956963 → 0.828242. The FM term's
mean logit is −0.17729 with std 0.07037. It contributes appreciably to probability
level in this fitted decomposition without a large marginal global AUC gain.
This does not show a separately trained FM would be useless. Removing deep or
wide greatly degrades probability quality; their fitted roles are complementary,
not interchangeable standalone models.

The existing DeepFM training history also shows train loss decreasing
0.20258 → 0.17887 → 0.16223 while validation AUC falls
0.65032 → 0.62150 → 0.60036 and LogLoss worsens
0.18981 → 0.19910 → 0.21265. This documents overfitting after epoch 1 under the
frozen recipe. It does not establish that architecture complexity alone caused
it or authorize retuning after test.

## Numerical checks and reproducibility

The first 8,192-row smoke stopped because float32 sparse-logit regrouping failed
an inappropriately tight float64-level tolerance (max error 1.91e-6). Its record
is retained in `results/deep_dive_d3_preflight_failure/failure.json`. Diagnostic
LR contributions now accumulate in float64 and are checked against original
float32 reduction and saved probabilities. No model parameters were changed.
The trusted sklearn 1.8.0 checkpoint required an explicit `_loss` module alias
under installed sklearn 1.9.1; stored coefficients/scaler were used, with full
validation parity verified rather than assuming pickle-version compatibility.

Full validation rows: 3,234,051. Maximum probability difference
from saved validation: LR 2.53e-07, DeepFM 0.
Execution took 29.90 seconds on mps. No training or test calls occurred.

[Artifacts](../results/deep_dive_d3_v1) contain all validation metrics, per-component
logit summaries/correlations, original LR recipe evidence, smoke, manifest and
independent verification. Large validation diagnostic arrays remain ignored in
`data/processed/deep_dive_v1/d3_validation_diagnostics.parquet`. No diagnostic
variant is a new formal model or is applied to test.
