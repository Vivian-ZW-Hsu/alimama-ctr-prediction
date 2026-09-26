# F4 sequence validation and first-test gate

## User-authorized v2 extension before first test

The conditional authorization has now been executed exactly once after pre-test
commit/push **0656c23**. See [frozen formal test results](DIN_F4_TEST_RESULTS.md).
The v2 freeze below remains unchanged after test; no model/configuration was
reselected. The remaining text documents the pre-test validation decision.

The latest instruction adds **zero_sequence_same_pipeline** and conditionally
authorizes one formal test pass after validation, smoke/reload and pre-test
commit/push. The current [v2 freeze](../results/sequence_selection_v2/freeze_manifest.json)
supersedes the test gate below; the original v1 manifest is retained unchanged.
The [comparison plan](../results/sequence_selection_v2/comparison_plan.json) fixes:

| Priority | Ordered comparison (left minus right) |
|---|---|
| Primary 1 | Mean matched vs zero sequence |
| Primary 2 | Mean matched vs mean mismatched |
| Secondary 3 | Attention matched vs zero sequence |
| Secondary 4 | Attention matched vs attention mismatched |
| Secondary 5 | Attention matched vs mean matched |

**Mean matched is the primary model** because validation differences are
negligible and mean is simpler. Test cannot change this choice. Primary metrics
are raw AUC and LogLoss, with PR-AUC, GAUC, COPC and ECE also reported; each
condition uses sample mean ± std over seeds 42/2026/3407. Calibration fitting is
not part of this raw-probability comparison batch; no calibration method is
chosen from test results. This scope supersedes the earlier prospective v1
calibration plan for the current batch.

The zero control uses the same `22_train_sequence_ctr.py`, frozen DeepFM/F123
cache, LR=5e-4, batch=4096, L=20, seed schedule, head dimensions, optimizer and
stopping rule. Its entire residual interface is hard-zero, including query and
sequence representation. The same 32-dimensional head executes, but its output
is unconditionally masked to zero; biases and gradients cannot create signal.
The original sequence module and all 12 existing checkpoints are unchanged.

All three zero-control runs selected epoch 1 (tie rule), completed three epochs,
and produced bit-identical validation predictions, exactly matching the frozen
DeepFM: **AUC 0.650315143693 ± 0; LogLoss 0.189812537469 ± 0**.
The [zero validation audit](../results/sequence_selection_v2/zero_validation_audit.json)
includes smoke checkpoint reload, full zero-control reloads and unchanged
predictions from reloading all 12 original checkpoints under the updated entry
point. Unit tests deliberately inject nonzero head weights/biases and verify
zero representations, zero outputs and zero gradients through optimizer updates.

**Interpretation correction:** the original backbone is frozen for all sequence
conditions; the residual seeds and batch size never retrain that backbone.
This is a pipeline null control for batch/seed/interface effects, not three
independent DeepFM fits or a trainable-capacity-matched no-history model. Its zero
standard deviation is structural and must not be presented as independent
replication. The original seed-42 DeepFM is listed separately as historical
reference and never inserted into the three-seed comparison mean.

The formal inference entry streams one batch at a time through the frozen
backbone and each of the 15 frozen checkpoints, writes immutable predictions,
and creates an exclusive start marker. Its smoke mode uses validation only.
Formal execution requires the passing smoke certificate and a clean commit
matching `origin/main`. Reports only read these predictions; no second inference
or post-test selection is allowed. Existing results are preserved under v1;
new test results use distinct `*_test_v2` names.

## Scope and evidence

This is the user-authorized Stage B/C study, conditional on the existing frozen
seed-42 DeepFM F123 checkpoint. No existing tabular family is retrained or
retuned. Sequence test inference, calibration application to test, and test
evaluation remain unauthorized. Test F4 construction and source-file integrity
hashing are the only test-related operations in this stage.

The [preregistration](../results/sequence_selection_v1/preregistration.json) was
committed in `524eb92` before any full-validation sequence search score. The
[selection evidence](../results/sequence_selection_v1/selection.json) records
all four matched-source candidates, including the rejected learning rate.
The [execution record](DIN_F4_EXECUTION.md) documents the Stage B construction
audits and train-only resource pilot. The [tested environment](../results/sequence_selection_v1/environment.json)
records the actual installed versions; historical `requirements.txt` still
omits torch/lightgbm and excludes the installed PyArrow 25. No packages were
reinstalled during this study.

## Frozen shared recipe

| Component | Specification |
|---|---|
| F4 source | Ordered categories of past positive ad exposures |
| Cutoff | Source Beijing date strictly less than target Beijing date |
| Timing assumption | Exposure timestamp proxies positive-record time; feedback available next day; independent click time unavailable |
| Validation history | Completed May 6–11 only; no May 12 clicks |
| Future authorized test history | Completed May 6–12; construction already audited |
| Length | Last 20 events; no other length searched |
| Vocabulary | Existing train-only category mapping; PAD=0, UNKNOWN=1 |
| Empty/unknown | Exactly zero representation and residual; reserved vectors remain zero after optimization |
| Context | Frozen original DeepFM checkpoint and numeric scaler; identical F123 predictions |
| Trainable branch | Shared query/history category embedding width 8, head `[64,32]`, zero-initialized output |
| Pooling | Masked mean or candidate-conditioned softmax attention with hidden width 32 |
| Optimizer | AdamW, LR **0.0005**, weight decay 1e-5, dropout 0.3, batch 4096 |
| Stopping | Up to 4 epochs; 2 consecutive non-improvements; select minimum validation LogLoss |
| Training randomness | Global row permutation with seed+epoch; model seeds 42, 2026, 3407 |
| Mismatch | Fixed seed 20260926; deterministic user derangement within cutoff / capped length / past-activity bucket |
| Activity buckets | 1–99 and >=100 prior exposures; fixed before model scores |

The event order is retained in storage and minibatches. Neither pooling operator
contains positional encoding, so neither is claimed to learn event order. This
is a DIN-style attention residual, not a reproduction of original DIN.

One common learning rate was selected by the average of the two matched
poolings' best validation LogLoss at seed 42:

| Learning rate | Mean LogLoss | Attention LogLoss | Paired selection score |
|---|---:|---:|---:|
| **0.0005** | 0.1897715111 | 0.1897717963 | **0.1897716537** |
| 0.0002 | 0.1897882038 | 0.1897892425 | 0.1897887231 |

Exact ties favor 0.0005; none occurred. All four candidates selected epoch 1.
Mismatch variants inherit the common recipe and stopping rule without an
independent search. The bounded first study retains L=20; it does not claim that
20 is an optimal sequence length.

## Completed validation results

All 14 unique runs completed; every run selected epoch 1 and stopped after epoch
3. The 12 selected checkpoints cover four conditions x three seeds. Each was
reloaded and regenerated its saved validation predictions **exactly**, with zero
probability error. The [freeze manifest](../results/sequence_selection_v1/freeze_manifest.json)
binds the recipe, checkpoint/prediction hashes, source manifests and code hashes.
All candidates were evaluated with `10_evaluate_predictions.py --splits val`.

Means ± sample standard deviations across seeds 42/2026/3407:

| Condition | Validation AUC | Validation LogLoss | Validation GAUC |
|---|---:|---:|---:|
| Frozen DeepFM (one existing seed) | 0.65031514 | 0.189812537 | 0.57621307 |
| Mean, matched | 0.65062618 ± 0.00003240 | 0.189773717 ± 0.000001998 | 0.57627641 ± 0.00006451 |
| Attention, matched | 0.65062308 ± 0.00002381 | 0.189773066 ± 0.000002500 | 0.57634333 ± 0.00002740 |
| Mean, mismatched | 0.65028718 ± 0.00001767 | 0.189795682 ± 0.000001103 | 0.57614163 ± 0.00004271 |
| Attention, mismatched | 0.65028943 ± 0.00001573 | 0.189795427 ± 0.000001341 | 0.57613073 ± 0.00003170 |

GAUC uses **101,129 eligible users**, covering **1,628,795 / 3,234,051 impressions
(50.3639%)**. Exact per-run metrics, PR-AUC, COPC/ECE and coverage are in
[validation_metrics.csv](../results/sequence_selection_v1/validation_metrics.csv);
[validation_seed_summary.csv](../results/sequence_selection_v1/validation_seed_summary.csv)
contains the aggregation. Every run's `evaluation/` also contains reliability
bins and cold-start quadrants. The original DeepFM reports were only read.

The [independent freeze audit](../results/sequence_selection_v1/freeze_audit.json)
verifies unchanged original input hashes/model/scaler and all selected artifacts.
All **1,030,633 empty-history validation rows** reproduce the baseline exactly in
every condition/seed. The [history-length report](../results/sequence_selection_v1/validation_history_length.csv)
uses the preregistered raw-length bins; the zero-history bin is unchanged.
Cold-start reports retain substantial raw probability error on new-ad/new-user
segments; this stage does not claim to solve cold-start calibration.

Matched histories improve validation AUC over the frozen baseline by approximately
0.00031 and LogLoss by 0.000039. They also outperform the mismatched controls on
AUC and LogLoss in every paired seed. Mismatched histories slightly improve
LogLoss despite slightly lower AUC than the baseline, so the entire probability
gain cannot be attributed to user-specific sequence content. Attention vs mean
is essentially tied on AUC/LogLoss at this scale; its GAUC mean is somewhat higher,
but no significance claim is made.

The complete selection/evaluation/checkpoint-reload orchestration took **859.46 s
(14.32 min)**. Summed training-process time was 766.55 s; peak observed training
RSS was **2.81 GiB**. F4 occupies 428.14 MiB and the train/validation context cache
365.34 MiB. This is below the preregistered six-hour budget. The full context
cache took 21.03 s; these are local observations, not a formal throughput benchmark.

## Interpretation boundaries and next gate

Validation participates in LR/epoch selection. Its improvements are descriptive,
not an unbiased final performance estimate or a significance claim. Three
residual seeds are conditional on one shared frozen DeepFM checkpoint, not three
independent full-model seeds. Paired user bootstrap and formal test remain pending.

The mismatch control exactly preserves retained length and activity bucket;
user-level permutation does not guarantee equal impression-weighted category
popularity, event age, or activity within each bucket. Report the recorded balance
diagnostic rather than treating it as complete causal adjustment. No rematching
or tuning follows these diagnostics.

The validation control-balance check has 13,852,168 non-PAD tokens per condition
and zero activity-bucket mismatches. Impression-weighted category total variation
distance is 0.04928. Mean retained-event age is 2.5315 days (matched) versus 2.6890
days (mismatched); mean past exposure activity on nonempty target rows is 194.65
versus 146.12. Thus matching the coarse bucket does not balance activity exactly.
The original preregistered mapping is retained, and this limitation accompanies
any interpretation of the negative control.

The cache manifest's `test_read: false` refers to model-input/inference access:
the cache script verifies F4 Parquet hashes, including test construction files,
but never scores test rows. The event store contains completed May 12 positives
for later test construction; train/validation range references exclude events
on or after their cutoff. No aggregate over those inaccessible events is fed to
the model.

The user explicitly requires a pause before the first formal sequence test.
After separate authorization, use the frozen checkpoints as-is: no retraining on
train+validation, no changed history rules, and no feedback from test into model
selection. Calibration is not yet fitted; the preregistered plan retains the
historical validation impression-hash 80/20 Platt/isotonic selection and full-val
refit, performed before viewing any sequence test metric. Formal reporting should
then use the shared evaluator, cold-start and preregistered raw-length buckets,
followed by paired user-cluster uncertainty analysis.

## Reproduction and local artifacts

Use the existing `ctr_project` Python environment, from the repository root.
`src/19_build_f4_click_sequences.py` and `src/20_audit_f4_sequences.py` are described
in the execution record. Cache and training commands require fresh run names;
completed predictions and formal runs are immutable. The selection orchestrator
refuses further tuning after its freeze manifest exists.

```bash
python -B -m unittest discover -s tests -v
python -B src/21_cache_sequence_context.py --f4-run f4_click_daily_v1 --run-name NEW_CACHE_NAME
python -B src/22_train_sequence_ctr.py --cache-run NEW_CACHE_NAME --run-name NEW_RUN_NAME --pooling attention --sequence-source matched --learning-rate 0.0005 --seed 42
python -B src/10_evaluate_predictions.py --run-name NEW_RUN_NAME --splits val
```

`src/23_select_sequence_configuration.py` records the complete preregistered
search and freeze; `src/24_audit_sequence_freeze.py` checks saved hashes, unchanged
inputs, exact empty-history predictions, and validation length/balance diagnostics.
Their existing completed study names must not be reused for a new experiment.
Large caches, weights and prediction Parquets are local Git-ignored artifacts.
Only code, documents and small evidence files are committed; Git push does not
back up those large local artifacts.
