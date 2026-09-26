# DIN / F4 execution record

## Authorization and test gate

On 2026-09-26 the user approved Stages B/C, including full F4 construction,
correctness/smoke, frozen-context sequence models, pilot and validation-only
selection. Stop before the first formal sequence test and present committed
configuration/selection evidence. No model test has been authorized.

The approved time assumption is: exposure timestamp is a proxy for the positive
record time; feedback is available the next Beijing day. There is no independent
click occurrence timestamp. Every history uses source Beijing date strictly less
than the target date. Validation excludes May 12 clicks; test construction can
use completed May 12 feedback. Test ID/label parity is a construction check only.

## Stage B — passed

Implementation: `src/f4_sequences.py`, `src/19_build_f4_click_sequences.py`, and
`src/20_audit_f4_sequences.py`. Category vocabulary is checked against train-only
categories and reuses the existing mapping, with PAD=0 and UNKNOWN=1. The event
store is sorted by raw user, exposure timestamp and original impression ID.
Per-exposure rows hold compact range references into the event store.

Before any model score, activity strata were fixed to past exposure counts 1–99
and >=100. This resolves a singleton found in the Stage-A proposed finer bins.
Within cutoff/length/activity strata, seed 20260926 determines a user-level
derangement. Same-length activity coarsening is explicitly counted as fallback;
all successful real smoke/full builds use **zero** fallback. An unmatchable length
stratum fails instead of silently self-matching or changing length.

Validation evidence:

- Three boundary/derangement tests pass: strict midnight exclusion, first-day
  emptiness, ordered latest-L truncation, deterministic derangement, counted
  coarsening fallback, and singleton failure.
- Two date-preserving smoke runs (`impression_id % 1000 == 0`) produce identical
  SHA256 for all event/history/reference Parquet files. Each passes independent
  persisted-artifact auditing and 256 original CSV/ad-category spot checks.
- [Smoke certificate](../results/f4_click_daily_v1_smoke_certificate.json) binds
  the tests to source hashes and builder/library hashes; the full builder requires it.
- [Full manifest](../results/f4_click_daily_v1/manifest.json): **26,557,961 rows**,
  eight dates, exact F123 IDs/labels, all original inputs unchanged.
- Zero self matches, fallback, length mismatches, future events, ownership
  mismatches and ordering violations, for both matched/mismatched references.
- Length histograms agree exactly; empty/truncated/UNKNOWN/PAD counts are recorded
  for every date. Test distribution summaries are integrity artifacts, not selection inputs.
- [Raw identity audit](../results/f4_click_daily_v1/raw_identity_audit.json):
  256 source rows recovered by original CSV row index; user/ad/time/positive label
  and raw ad-table category agree, with zero mismatches.
- Independent full persisted-artifact audit and Python compilation passed.

Full construction took 31.07 seconds, measured peak RSS 2.28 GiB, and wrote
448,941,614 bytes (~428 MiB). Timings are local observations, not service guarantees.
Large Parquet files remain Git-ignored under `data/processed/f4_click_daily_v1/`;
small manifests/audits and source code are committed. Completed builds are immutable
and verified on re-entry; date checkpoints support resuming interrupted builds.

Reproduce (from repository root, using the existing ctr_project interpreter):

```bash
python -B -m unittest discover -s tests -p 'test_f4*.py' -v
python -B src/19_build_f4_click_sequences.py --run-name f4_click_daily_v1_a_smoke --smoke-test
python -B src/19_build_f4_click_sequences.py --run-name f4_click_daily_v1_b_smoke --smoke-test
# Certificate creation is exclusive: use a new path if independently repeating.
python -B src/20_audit_f4_sequences.py --run-name f4_click_daily_v1_a_smoke --compare-smoke f4_click_daily_v1_b_smoke --certificate results/f4_click_daily_v1_smoke_certificate.json
python -B src/19_build_f4_click_sequences.py --run-name f4_click_daily_v1 --smoke-certificate results/f4_click_daily_v1_smoke_certificate.json
python -B src/20_audit_f4_sequences.py --run-name f4_click_daily_v1
```

## Stage C — passed; stopped before first formal test

Reuse the tabular implementation via a read-only adapter, without editing its
training script. Cache frozen DeepFM logits for train/validation only, asserting
saved validation predictions agree to absolute probability tolerance 1e-6. Train
only a small category-sequence residual. The context remains frozen; empty or
all-UNKNOWN sequences must contribute exactly zero even after training.

Initial max length is fixed at 20. Predeclare the mean/attention paired search
and its budget after resource pilot, before the first searched validation score.
No test branch is exposed by the sequence cache/training entry points.

Implementation checks: seven unit tests pass, including padding/UNKNOWN/empty
gradients and an evaluator integration test with no test files present. Mean
matched and attention mismatched smoke runs pass on MPS. Full train/validation
context caching covers 20,015,245 / 3,234,051 rows; frozen validation probabilities
match the original DeepFM predictions exactly. Cache construction took 21.03 s,
peak RSS 3.05 GiB. The adapter never updates the original model/scaler.

The shared evaluator now accepts explicit `--splits val`; its default remains
val+test for historical callers. It rejects overwriting existing evaluations.
New sequence training runs have no test option, save validation predictions only,
use global seeded train-row permutations and expand at most one minibatch of
length-20 history at a time. A train-only attention pilot precedes grid selection.

Pilot passed: one complete attention train epoch took 17.09 seconds, peak RSS
1.61 GiB, no validation score or test inference. The bounded first study retains
L=20 (no L=50 experiment) and preregisters the paired LR grid `{5e-4,2e-4}` in
`results/sequence_selection_v1/preregistration.json` before full validation scores.
Select one common LR by mean best validation LogLoss across the two matched
poolings, then run inherited matched/mismatched conditions for seeds 42/2026/3407.
At most 14 unique runs x 4 epochs, estimated under 30 minutes including shared
evaluation, with the existing hard 6 machine-hour cap. All final checkpoints
are chosen using validation and remain behind the formal test gate.

Completed outcome: 14 unique validation-only runs, shared evaluator reports for
all candidates, and 12 frozen checkpoints (four conditions x three seeds). All
best epochs are 1. Selected common LR is 5e-4; mean/attention matched paired
selection LogLoss is 0.1897716537 versus 0.1897887231 at 2e-4. Three-seed matched
validation AUC is approximately 0.65062 and LogLoss 0.189774; attention is not
clearly superior to mean. See [full analysis and recipe](DIN_F4_VALIDATION.md).

Nine unit tests and repeated bit-identical mean smoke predictions pass. Every
selected checkpoint reproduces its saved validation predictions exactly. The
[independent freeze audit](../results/sequence_selection_v1/freeze_audit.json)
confirms unchanged original input fingerprints/model/scaler, exact baseline
predictions on all 1,030,633 empty-history validation rows for all 12 models,
and absence of new test predictions. Validation-only history-length and control-
balance diagnostics are saved after configuration selection.

Full search/evaluation/reload took 859.46 seconds (14.32 min), peak training RSS
2.81 GiB. [Freeze manifest](../results/sequence_selection_v1/freeze_manifest.json)
binds the checkpoints, recipe and code. No calibration was fitted, no sequence
test inference was run, and no test metric selected any setting. The next action
requires the user's explicit first-test authorization; do not continue tuning.
