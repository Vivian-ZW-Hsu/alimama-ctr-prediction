# Frozen-prediction statistical and calibration analysis

This stage is authorized after overall test reporting. It does not retrain CTR
models, produce new CTR inference, or alter the frozen five comparisons. The
user's new retained-length buckets are fixed before these grouped diagnostics,
not before the already completed initial test. Historical raw-length buckets
remain preserved as earlier evidence. See the machine-readable
[protocol](../results/sequence_posthoc_v1/protocol.json).

The bootstrap draws U raw users with replacement, retaining all exposures with
their user's draw multiplicity. The same draw weights apply to every predictor.
For each draw, compute exact weighted AUC, LogLoss, average precision and GAUC
for each saved seed, then average seed-paired differences. Zero is loaded once.
No seed resampling or probability ensembling is used. This isolates data-sampling
uncertainty conditional on fixed checkpoints; training-seed variation is reported
separately. The target remains the observed test-day user population, not future-day
distribution uncertainty.

Use 1,000 valid replicates, seed 20260927, percentile 95% intervals and the
predeclared centered two-sided bootstrap p-value with finite-simulation +1
correction. Report Holm-adjusted p-values across the two primary comparisons per
metric, plus a joint four-hypothesis AUC/LogLoss primary family. Secondary tests
are exploratory. Percentile intervals are marginal, not Holm-simultaneous intervals.
P-values are approximate sampling tests, not exact randomization tests.

The optimized implementation uses exact sorted scores and tie groups, weighted
by integer user multiplicities. It must agree with explicit duplicated-user
samples and sklearn metrics on small fixtures, and with the existing full-test
point estimates before executing bootstrap. No score bins or linearized influence
approximations replace full metrics. A small resource pilot prices the full job.

Reference definitions: [sklearn average precision](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.average_precision_score.html)
uses the non-interpolated precision-recall sum and supports weights;
[SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
describes paired resampling and percentile intervals;
[statsmodels multiplicity correction](https://www.statsmodels.org/v0.12.2/generated/statsmodels.stats.multitest.multipletests.html)
defines Holm as step-down Bonferroni. User clustering and conditional seed averaging
are specified explicitly in this project's protocol.

Groups use target-user matched history for every model: retained length
0/1/2–5/6–10/11–19/20, raw candidate-category membership in the retained history,
and existing train-vocabulary user/ad cold-start quadrants. UNKNOWN index equality
is not category equality. Empty histories never have a category hit.

Calibration reuses the existing deterministic validation 80/20 impression-hash
fit/selection, Platt vs isotonic by selection LogLoss, then full-validation refit.
All 12 sequence seeds and one zero reference are calibrated independently and
retained regardless of test results. No grouped calibrator is fitted. Comparable
reports use the shared evaluator's 10 equal-count ECE; the legacy helper's
20 equal-width ECE is kept only in method-selection diagnostics.
