# F5 behavior-sequence formal test results

**One inference pass per frozen checkpoint, executed once.** Fifteen checkpoints
over 3,308,665 test impressions in 0.9 minutes, with an exclusive start marker
written before the first forward call. Empty-history rows produced an exactly zero
residual and the zero control produced predictions bit-identical to the frozen
backbone on every batch. Nothing was retuned, reselected or retrained afterwards.

The analysis rules, the six ordered comparisons and all seven group partitions were
committed in `c06db65` and `b223020` **before** this pass.

## Headline metrics, three residual seeds on one shared frozen backbone

| Condition | Test AUC ± std | Test LogLoss ± std | Test GAUC ± std |
|---|---|---|---|
| attention matched | 0.647445 ± 7.7e-5 | 0.193043 ± 2.9e-5 | 0.570476 ± 5.4e-5 |
| **mean matched** (primary) | 0.647206 ± 6.0e-5 | 0.193061 ± 3.1e-5 | 0.570251 ± 2.4e-4 |
| F4 clicked-ad mean matched | 0.646263 ± 4.4e-5 | 0.193120 ± 4.0e-6 | 0.570380 ± 1.1e-4 |
| attention mismatched | 0.646080 ± 1.1e-5 | 0.193151 ± 1.3e-5 | 0.569825 ± 5.8e-5 |
| mean mismatched | 0.646003 ± 1.3e-4 | 0.193153 ± 1.1e-5 | 0.569840 ± 6.3e-5 |
| zero pipeline null | 0.645933 ± 0 | 0.193176 ± 0 | 0.569902 ± 0 |

The zero standard deviation is structural: the zero control is one deterministic
predictor on the shared backbone, evaluated three times, not three independent
models. All five conditions share one frozen DeepFM checkpoint, so no contrast
below is a full-model comparison.

## The six frozen comparisons

Paired user-cluster bootstrap, 1,000 replicates over 391,741 test users, seed
20260927, conditional on the frozen checkpoints and the single test day. The
unweighted replicate reproduces every saved point metric. Holm covers the two
primary comparisons on AUC and LogLoss, four hypotheses.

| # | Comparison | Metric | Effect | 95% CI | Holm p | Excl. 0 |
|---|---|---|---:|---|---:|:--:|
| 1 | mean matched − zero | AUC | **+0.00127246** | [+0.00104340, +0.00150052] | 0.003996 | yes |
| 1 | mean matched − zero | LogLoss | **−0.00011503** | [−0.00013956, −0.00009046] | 0.003996 | yes |
| 2 | mean matched − mean mismatched | AUC | **+0.00120269** | [+0.00098519, +0.00142260] | 0.003996 | yes |
| 2 | mean matched − mean mismatched | LogLoss | **−0.00009160** | [−0.00011628, −0.00006841] | 0.003996 | yes |
| 3 | attention matched − zero | AUC | +0.00151214 | [+0.00127903, +0.00175247] | — | yes |
| 3 | attention matched − zero | LogLoss | −0.00013328 | [−0.00015941, −0.00010942] | — | yes |
| 4 | attention matched − attention mismatched | AUC | +0.00136519 | [+0.00114619, +0.00159020] | — | yes |
| 4 | attention matched − attention mismatched | LogLoss | −0.00010797 | [−0.00013278, −0.00008461] | — | yes |
| 5 | attention matched − mean matched | AUC | +0.00023967 | [+0.00017050, +0.00031038] | — | **yes** |
| 5 | attention matched − mean matched | LogLoss | −0.00001825 | [−0.00002580, −0.00001103] | — | yes |
| 6 | F5 mean matched − F4 mean matched | AUC | **+0.00094211** | [+0.00070402, +0.00118202] | — | yes |
| 6 | F5 mean matched − F4 mean matched | LogLoss | −0.00005886 | [−0.00008394, −0.00003399] | — | yes |

Secondary metrics tell a different story on within-user ranking:

| # | GAUC effect | 95% CI | p | Excl. 0 |
|---|---:|---|---:|:--:|
| 1 | +0.00034899 | [−0.00016254, +0.00085982] | 0.172 | **no** |
| 2 | +0.00041059 | [−0.00008260, +0.00086338] | 0.087 | **no** |
| 3 | +0.00057382 | [+0.00006347, +0.00108745] | 0.029 | yes |
| 4 | +0.00065032 | [+0.00015761, +0.00110441] | 0.007 | yes |
| 5 | +0.00022483 | [+0.00001678, +0.00043734] | 0.035 | yes |
| 6 | **−0.00012937** | [−0.00066206, +0.00037694] | 0.621 | **no** |

PR-AUC excludes zero on all six comparisons. Every one of the twelve
comparison-metric pairs on AUC and LogLoss favours the left condition in **all
three seeds**, with per-seed standard deviations of 4e-6 to 1.2e-4 — a separate
quantity from the bootstrap interval and never merged with it.

## The seven questions

**1. Does F5 improve stably over the zero pipeline null?** Yes on global ranking and
probability quality. AUC +0.00127 and LogLoss −0.000115 both exclude zero at Holm
p = 0.003996, and all three seeds agree. On **GAUC the interval includes zero**
(p = 0.172), so the improvement is not established for within-user ranking.

**2. Is F5 matched stably better than mismatched?** Yes. AUC +0.00120 excludes zero
at Holm p = 0.003996 with all seeds agreeing. Since matched − zero is +0.00127 and
mismatched − zero is only +0.00007, the mismatch control captures about **5.5%** of
the effect: roughly 95% of the gain is specific to the user's own history rather
than to retained length, coarse activity, recency or category popularity.

**3. Does F5 clearly exceed the sparse F4 clicked-ad source?** On global ranking,
yes: +0.00094211 AUC, CI excluding zero, all three seeds agreeing, which is about
2.9× F4's own gain over the same null (+0.000330). On **within-user ranking, no**:
their GAUC difference is −0.00013 with an interval spanning zero (p = 0.621). So a
40-fold richer history buys a clear global-ranking improvement and no measurable
within-user one.

**4. Is attention stably better than mean pooling?** On this source, the interval
now **excludes zero**: +0.00023967 AUC, CI [+0.00017050, +0.00031038], with all
three seeds agreeing, and the same on LogLoss and GAUC. This differs from the F4
study, where the corresponding intervals contained zero. Two qualifications are
essential. The effect is **practically tiny**, about one fifth of the
matched-versus-mismatched effect and about two seed standard deviations. And the
interval is user-sampling uncertainty conditional on these fixed checkpoints, not
training-seed uncertainty. **Mean matched remains the primary model**, fixed before
any result; nothing here reselects it.

**5. Does donor bucket relaxation affect the matched-minus-mismatched conclusion?**
No, decisively. Excluding all 1,457 relaxed rows (0.0440%) changes the effect by at
most **9.0e-7** on AUC and **5.0e-6** on GAUC, with every sign preserved:

| Pooling | Metric | Main | Relaxation excluded | Change |
|---|---|---:|---:|---:|
| mean | AUC | +0.001203 | +0.001203 | +3.8e-7 |
| mean | LogLoss | −0.000092 | −0.000092 | +5.0e-8 |
| mean | GAUC | +0.000411 | +0.000415 | +4.0e-6 |
| attention | AUC | +0.001365 | +0.001366 | +9.0e-7 |
| attention | GAUC | +0.000650 | +0.000656 | +5.0e-6 |

Level 3 relaxation, a donor merely at least as long as the target, has **zero rows**
on every date, so exact retained length held everywhere.

**6. Does F5 narrow the global AUC gap between LR and the sequence models?** Partly.
LR F123 reaches 0.652489. The gap to the frozen DeepFM null was 0.006556; F5
reduces it to **0.005283**, closing about **19.4%**. LR still leads by roughly four
times the entire F5 sequence effect. On **GAUC the ordering reverses**: F5 mean
matched reaches 0.570251 against LR's 0.569939. That is consistent with stage E3,
where every GAUC difference among the tabular families also included zero — global
AUC and within-user ranking are not measuring the same thing here.

**7. Is the F5 gain worth its serving cost?** On this implementation, no. Against
the DeepFM backbone it sits on, F5 costs about **3.3× the end-to-end throughput**
(379,078 against 1,239,580 rows/s at batch 8192) and about **7.5× the peak memory**
(6,082 against 813 MiB), for +0.00127 AUC. The memory is structural: it comes from
holding the 723-million-event store resident, not from the residual's 57,345
parameters. A production feature store that paged windows instead would change that
number; this benchmark measures the implementation that exists. See
[SERVING_BENCHMARK.md](SERVING_BENCHMARK.md).

## Calibration

The method was selected on validation only, with the existing
impression-hash 80/20 recipe. **All thirteen calibrators chose isotonic.** Applied
to the frozen test predictions, for mean matched:

| | Raw | Calibrated |
|---|---:|---:|
| AUC | 0.647206 | 0.647148 |
| LogLoss | 0.193061 | **0.192755** |
| COPC | 0.9478 | **0.9866** |
| ECE | 0.004785 | **0.000975** |
| GAUC | 0.570251 | 0.569790 |

Probability quality improves substantially and ranking metrics fall slightly. That
is the documented tie-creation mechanism: isotonic is a monotone step function, so
it introduces no new inversions but collapses distinct scores onto shared steps.
No test metric influenced the choice of method.

## Predeclared group results

All seven partitions were fixed before the test. Effects below are F5 mean matched
minus the zero null on raw probabilities, seed-averaged. These are descriptive
subgroup differences with **no subgroup intervals**; small cells are not evidence.

**F4-by-F5 emptiness cross — the clearest result.** F5's largest gain lands exactly
where F4 had nothing:

| Cell | Rows | F5 − zero AUC |
|---|---:|---:|
| F4 nonempty + F5 nonempty | 2,308,580 | +0.001163 |
| **F4 empty + F5 nonempty** | **985,688** | **+0.001607** |
| F4 empty + F5 empty | 13,971 | +0.000000 |
| F4 nonempty + F5 empty | 426 | +0.000000 |

The 985,688 impressions where the clicked-ad source is empty are 29.8% of the test
day, and F5 helps more there than where both sources have history. Both
empty-history cells are exactly zero, as the architecture requires.

**Cold start.** The gain is largest for new users with known ads, where behavior
history partly substitutes for an unseen user ID, and is **negative on double cold
start**:

| Segment | Rows | F5 − zero AUC |
|---|---:|---:|
| known user + known ad | 2,659,898 | +0.001123 |
| **new user + known ad** | 358,678 | **+0.002785** |
| known user + new ad | 261,758 | +0.000950 |
| **new user + new ad** | 28,331 | **−0.001741** |

The double-cold cell holds 0.86% of the test day and carries no interval, so the
sign is reported rather than interpreted.

**Recency.** The gain decays as the retained history gets staler: +0.001381 when the
user acted the previous day (2,759,194 rows), +0.001083 at one day (295,419),
+0.000330 at two or more (239,655).

**UNKNOWN token share.** Higher shares do **not** degrade the gain: +0.001113 at
zero UNKNOWN, +0.001470 in (0, 0.05], +0.001101 in (0.05, 0.20] and +0.001526 above
0.20. The 4.30% UNKNOWN masking is therefore not the binding constraint.

**History length.** Positive across the large bins, including +0.001280 on the
2,790,540 truncated rows, and negative on very short histories (−0.009177 at length
1 on 2,519 rows, −0.011110 at 2–5 on 7,643 rows). **User activity.** +0.001900 at
10–99 events, +0.001459 at 100–999, +0.000871 above 1,000, and negative at zero
activity (−0.001627 on 20,799 rows). **Behavior type composition.** Positive in
every cell, from +0.001088 to +0.002173.

Full tables: [`group_metrics.csv`](../results/f5_sequence_test_v1/group_metrics.csv)
and [`group_counts.csv`](../results/f5_sequence_test_v1/group_counts.csv).

## Limits

- One held-out day, one shared frozen backbone, three residual seeds. No
  independently retrained full model, so no contrast here separates the residual
  from the backbone.
- Bootstrap intervals are marginal per metric, not simultaneous, and only the two
  primary comparisons carry a Holm correction. Primary p-values sit at the
  resolution floor 1/1001.
- Subgroup results have no intervals and no multiplicity control. They are
  descriptive. No partition was added, split or re-binned after a metric was seen.
- Comparison 6 contrasts two information sources, not two architectures: retained
  length 200 against 20, mean retained tokens about 185 against 4.69, empty share
  0.44% against 30.21%.
- Group metrics were computed for the five F5 conditions; the F4 reference has
  overall metrics only, so comparison 6 has no subgroup breakdown.
- The test set had been inspected during earlier baseline development and is a
  frozen final reporting set, not a never-seen holdout.
- The published DSIN analogue for introducing a behavior sequence is about +0.0074
  AUC; F5's +0.00127 is roughly 5.8× smaller. Candidate reasons, none isolated: a
  residual on a frozen backbone rather than joint training, a completed-day rather
  than per-event cutoff, category and type tokens with no item identifier, and
  UNKNOWN masking.

## Reproduction

```bash
PY=/opt/anaconda3/envs/ctr_project/bin/python
$PY -B src/52_prepare_f5_test.py
$PY -B src/53_infer_f5_test.py --smoke-test     # validation only; issues the certificate
$PY -B src/53_infer_f5_test.py                  # refuses without a clean pushed worktree
$PY -B src/54_report_f5_test.py
$PY -B src/55_posthoc_f5_test.py --stage fit    # validation only
$PY -B src/55_posthoc_f5_test.py --stage apply
$PY -B src/56_bootstrap_f5_test.py
```

Every entry point refuses to rerun once its manifest exists. The start marker in
[`inference_started.json`](../results/f5_sequence_test_v1/inference_started.json)
forbids a second inference pass.
