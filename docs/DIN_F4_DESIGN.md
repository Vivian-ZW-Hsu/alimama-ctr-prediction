# DIN / F4 design — Stage A

Date: 2026-09-26. Audited baseline: `f0ba2a1f314e7b6d6dfc9a64c5e22b21ba4319ca`.

**Status: Stage-A read-only source audit complete; design is proposed, not yet
approved for construction or training.** Handoff §15 requires direction
confirmation before Stage B. This document does not authorize changes to the
existing four tabular models or their formal outputs.

## 1. Charter, scope, and evidence

The long-term charter is [CTR_CODEX_PROJECT_HANDOFF.md](CTR_CODEX_PROJECT_HANDOFF.md).
Current user instructions and repository facts take precedence. `AGENTS.md`
points future work to the charter and this gate.

At entry, `main` was clean except for the user-provided, untracked handoff file.
The remote `origin/main` was checked with `git ls-remote` and matched the audited
HEAD. The five latest commits were `f0ba2a1`, `802158e`, `a14cefe`, `0285426`, and
`746e3e0`. No old models were rerun, no test predictions were generated, and no
existing input or formal result was written by this audit.

Evidence:

- [Source audit v2](../results/din_f4_audit_20260926/source_audit_v2.json): full
  train/validation narrow-column scans, source schemas, metadata, coverage,
  timestamp ties, and click-to-ad-table alignment.
- [Initial source audit](../results/din_f4_audit_20260926/source_audit.json): retained
  preliminary pass, superseded by v2, which adds the raw ad-table alignment check.
- [Handoff verification](../results/din_f4_audit_20260926/handoff_verification.json):
  assertions against the existing metric tables and recorded environment versions.
- [Read-only audit script](audits/audit_din_f4_sources.py): prints JSON to stdout;
  does not construct features, train models, import project entry points, or write inputs.

Reproduce the final source audit from the repository root with the existing environment:

```bash
/opt/anaconda3/envs/ctr_project/bin/python -B docs/audits/audit_din_f4_sources.py
```

The absolute interpreter path is specific to this machine; it is not a project
data path. Current default Homebrew `python3` is Python 3.14 without Polars or
PyArrow. The existing `ctr_project` environment has Python 3.11.16, Polars 1.44.2,
and PyArrow 25.0.0. No dependencies were installed or changed.

Test access was restricted to existing published metric CSVs for handoff
verification, and Parquet footers/schema/row counts for input inventory. There
was no new test-label scan, sequence coverage analysis on test, or test-driven
choice. Raw behavior was read only as a 200,000-row prefix plus bounded fingerprint
reads; the large behavior log was not rebuilt or fully scanned.

Large raw-file fingerprints are size/mtime plus SHA256 of the first and last
1 MiB, explicitly **not** full-file checksums. Parquet inventory uses stat and
footer metadata. Strong output checksums belong to the future build manifest.

## 2. Handoff verification and conflict register

The reported split sizes, LR information-ablation metrics, four-model main
metrics, GAUC coverage, calibration metrics, and neural calibrated cold-start
rows/biases agree with existing files at the handoff's displayed precision.
Formal model configurations and epoch histories agree with the recorded manifests.
The latest neural formal runs contain three epochs and select epoch 1.

| ID | Claim / discrepancy | Repository evidence and treatment |
|---|---|---|
| A01 | Handoff §4 says `user_profile` is about 1.1 GB | Raw file is **24,056,588 bytes** (~24.1 MB); `raw_sample.csv` is 1,088,060,964 bytes. Use actual stat values, preserve the supplied handoff as historical text. |
| A02 | Handoff §7 broadly says UNKNOWN correction improved neural performance and stability | W&D test AUC changes 0.645958464 → 0.646219439, but raw LogLoss changes **0.193144820 → 0.193160860** (slightly worse). DeepFM raw LogLoss improves 0.193195200 → 0.193176498. Use metric-specific statements. Stability across seeds and causal attribution were not measured. |
| A03 | The code comment says UNKNOWN index cannot occur in training | `results/model_input/model_input_summary.csv` records train `brand_unknown_rate=0.30996398`; index 1 also represents missing values. The correction freezes **missing and unseen** embeddings, not only unseen user/ad IDs. No historical code changes in Stage A. |
| A04 | “F3 is the largest information gain” | Supported by the saved LR ablation. LightGBM F1/F12 ablations are absent; do not generalize it to every model or claim a causal mechanism. |
| A05 | Old protocol vs actual LR selection | Actual `results/lr_tuning/selection.json` chooses highest AUC within 0.0002 of minimum validation LogLoss; old protocol describes a different LogLoss/Pareto rule. Keep this deviation explicit. |
| A06 | Old protocol vs actual calibration | `src/14_calibrate_predictions.py` uses one deterministic impression-ID holdout (~80/20), not five-fold user-grouped OOF. Existing results stay historical; no silent recalibration. |
| A07 | “Global calibration improves overall bias” needs qualification | Neural COPC improves, but LightGBM calibrated test LogLoss and COPC are slightly worse. Keep family-specific conclusions. ECE also uses 10 equal-count bins in evaluator vs 20 equal-width bins in calibrator; do not compare them as one definition. |
| A08 | README/protocol status and dependencies are stale | README stops at LR; requirements omit torch/lightgbm and constrain PyArrow <25 although neural manifests record 25.0.0. Early LR manifest contains `*_v2` paths. Record now, reconcile in the appropriate documentation/environment phase. |
| A09 | Old protocol asks for a DIN context-only run and extra tabular work before sequence work | The new charter prioritizes F4 and names existing DeepFM as the no-sequence reference; the user's current instruction forbids retuning the four tabular models. Reuse the frozen DeepFM reference. Additional architecture claims require separate authorization/design. |

Sources for A02 are `results/nn_unknown_embedding_diagnostic/*/before_fix_overall.csv`
and current neural `evaluation/overall_metrics.csv`. The “not the main reason”
statement about UNKNOWN in the handoff remains an interpretation, not a controlled
mechanism result. New direction decisions will be recorded here rather than
editing old metrics or claiming historical preregistration.

## 3. Candidate sources and measured semantics

| Property | Raw `behavior_log.csv` | Positive labeled ad impressions |
|---|---|---|
| Observed fields | `user,time_stamp,btag,cate,brand` | `impression_id,user_raw,time_stamp,adgroup_id_raw,clk` plus ad-table category/brand |
| Event meaning supported by fields | `pv/cart/fav/buy` behavior; **not** an advertising click flag | A labeled ad exposure with `clk=1`; timestamp is the exposure record's timestamp, not a separately observed click arrival |
| Candidate alignment | Category/brand field overlap; no item/adgroup ID, and repository has no authoritative cross-source namespace contract | Same adgroup key and same `ad_feature.csv` mapping as the current candidate |
| Pre-training history | Existing aggregation spans 2017-04-22 onward | Labeled exposures start 2017-05-06; first training date has no prior-day click history |
| Availability constraint | Event timestamp available, but ingestion delay not recorded | Separate click time / feedback arrival time absent; do not equate earlier exposure with earlier available feedback |
| Reusable ordered data | None: daily table retains counts and last timestamps, but discards individual category/brand events | Narrow columns already exist in base train/val Parquet |
| Size / measured count | Raw CSV 23,728,773,580 bytes; existing daily aggregate reports 653,299,059 events through May 11 | 1,040,129 train + 159,437 validation positives = **1,199,566** potential historical records through May 12 |

The raw ad table has 846,811 distinct adgroup keys. A full check of all 1,199,566
train/validation positive rows found **zero category/brand mismatches** against
that table and zero missing categories. This establishes the clicked-ad mapping
used by this project; it does not establish point-in-time changes of static ad
attributes, which have no version timestamps and inherit the F1 assumption.

Behavior coverage from already materialized F2 is 99.4436% of train impressions
and 99.4466% of validation impressions, with 14-day event medians 438 and 445.
This is **any-behavior coverage**, not proof that usable aligned F4 tokens have
the same coverage. The prior-to-validation daily aggregate consists of
622,184,472 pv, 14,414,787 cart, 8,417,318 fav, and 8,282,482 buy events.

The first 200,000 raw behavior rows are a non-random prefix. Of 199,983 rows in
the permitted pre-validation date window, 96.4472% have category IDs present in
the train ad-category vocabulary and 83.7276% have brand IDs present in its brand
vocabulary. These are **sample-only overlaps**, neither population estimates nor
proof of shared ID semantics. Full behavior-source alignment remains unverified.

Prior-day positive-click histories, before truncation:

| Measurement | Train | Validation |
|---|---:|---:|
| Target impressions | 20,015,245 | 3,234,051 |
| Impressions with at least one prior-day click | 8,624,634 | 2,203,418 |
| Nonempty history rate | **43.0903%** | **68.1318%** |
| Maximum history length over target impressions | 131 | 152 |
| Impressions truncated at L=20 | 1.6403% | 5.7450% |
| Impressions truncated at L=50 | 0.1252% | 0.6620% |

All 3,270,348 May 6 targets have empty history. On validation, an untruncated
past click shares the candidate category for 948,733 impressions (**29.3357%**),
and the exact adgroup for only 84,326 (**2.6074%**). These overlap rates use no
target outcomes in their definition and are not model performance measures.

There are 153,581 user/second groups containing multiple positive exposures
(335,404 records; maximum 15 per group). Serialization must resolve ties
deterministically; row order must never be used to infer click availability.

## 4. Recommended direction and limits

**Primary recommendation: prior completed-day positive ad-exposure feedback,
category-only tokens, initially the latest L=20 records per user.** Keep original
adgroup/source impression IDs for provenance, but do not add adgroup or brand as
sequence model inputs in the first experiment.

Reasons: candidate/category alignment is fully checked in the same ad namespace;
event count is small; ordered records can be recovered without scanning/rebuilding
the large behavior log; category reuse is much more common than exact-ad reuse.
Brand is missing in 33.2185% of train positive records and 34.1941% of validation
positive records. This is a semantic/resource recommendation, not a claim that
click histories predict better than behavior histories.

The resulting question is: **does the composition of a user's previously clicked
advertising categories add information beyond F1/F2/F3?** It does not test all
shopping behavior, identify a true item sequence, or measure a pure temporal-order
effect. Empty-history and first-day limitations must remain in the report.

Behavior-log F4 remains a distinct alternative, not an interchangeable fallback.
Switching sources changes the information budget and requires direction
confirmation. A raw-event streaming extraction and additional namespace audit
would be required; daily aggregates cannot recover ordered events.

## 5. Cutoff, availability assumption, and accumulation

For a target exposure on Beijing date `d`:

```text
cutoff = Beijing midnight at the start of d
history = source rows with same raw user, clk == 1,
          source event_date_bj < d AND source time_stamp < cutoff
sequence = latest L records sorted by (source time_stamp, source impression_id)
```

This is the conservative **completed-date** option explicitly allowed by handoff
§11, aligned with F2/F3 daily freshness. Do not include same-day click labels,
even if their exposure timestamps are earlier than the target. Do not mix this
with a same-day `event_time < impression_time` policy between experiments.

**Availability assumption requiring direction confirmation:** previous-day click
labels are available by the next Beijing midnight, matching existing F3's daily
feedback assumption. The dataset cannot verify actual click/ingestion latency;
daily exclusion reduces leakage risk but does not prove real-world arrival time.
If that assumption is rejected, a separately approved lag or a different source
is necessary; do not invent click timestamps.

| Target | Permitted positive-feedback source dates |
|---|---|
| Train May 6 | Empty |
| Train May 7–11 | May 6 through the preceding completed date |
| Validation May 12 | May 6–11 |
| Test May 13, only after configuration freeze | May 6–12, including completed validation feedback |

No May 13 label is needed to construct any history in this design. Retain repeated
positive exposures with different impression IDs; do not deduplicate category
tokens or treat repeated impressions as one click. Tie-breaker ID orders storage
only. Category vocabularies and scalers reuse the existing train-fitted artifacts;
no validation/test vocabulary expansion, popularity fitting, or normalization fit.
This inherits the existing full-training-split preprocessing convention; sequence
events, activity matching, and length statistics obey the per-date cutoff.

## 6. Proposed event and output contract

Build under a new versioned directory, e.g. `data/processed/f4_click_daily_v1/`,
with a unique build manifest. Never add columns to existing F123 Parquet files.

| Field | Meaning / type |
|---|---|
| `source_impression_id` | UInt32; unique original positive exposure identity |
| `source_user_raw` | Int64; original user, never encoded before the ownership join |
| `source_exposure_timestamp` | Int64 seconds; explicitly not named `click_time` |
| `source_date_bj` | Date; derived consistently from timestamp |
| `source_adgroup_raw`, `source_category_raw` | Int64; provenance and source-table verification |
| `category_idx` | UInt32, mapped with existing `cate_id_raw_mapping.parquet` |
| `adgroup_idx` | Optional audit-only mapping; not a first-experiment token |
| `cutoff_timestamp`, `source_start`, `source_end` | Per-user/day index into a user/time-sorted event store |
| `raw_history_length`, `retained_length` | Counts before/after truncation; audit/stratification metadata |
| `impression_id`, `history_ref`, `mismatched_history_ref` | Per-target references; preserve row identity without duplicating F123 |
| `donor_user_raw`, `matching_stratum`, `mismatch_seed` | Provenance for negative-control auditing, never neural inputs |

Use a normalized event store and range/reference tables, then gather tokens per
minibatch. Alignment with F123 is by validated impression ID, never positional
row alignment. Builder writes train/validation first; test references are deferred
until model and cutoff freeze. Including validation positives in the event store
must not make them visible to earlier targets.

PAD=0; UNKNOWN=1; known tokens start at 2. Both reserved embeddings must remain
exactly zero. Existing tabular code freezes index 1 only, so sequence code must
also mask/neutralize PAD=0 explicitly. Real unknown events retain their positions
for audit length, but have neutral token vectors; all-invalid or empty histories
produce a strict zero sequence contribution. Mean and attention use the same
valid-token mask. An all-masked attention row must return zero without softmax
NaNs or a learnable bias contribution.

Start L=20. The only predeclared length alternative is L=50 if included in the
bounded validation experiment below. Keep raw length and truncation indicators
for reporting. No explicit length feature, event-age embedding, brand token,
positional encoder, or extra action feature is introduced in the first experiment.
Mean and target attention are permutation-invariant aggregators; ordering selects
the most recent window but does not make them temporal-order models.

## 7. Reproducible mismatched-user control

Use a **daily donor mapping**, constructed solely from positive histories and
exposure activity strictly before that day's cutoff. The donor pool is users with
available prior histories, not users discovered from later target-day exposures.
No label, target category, target prediction, or future activity enters matching.

1. Freeze seed 20260926. Sort raw user IDs and use a documented deterministic RNG;
   avoid Python's process-randomized `hash()`.
2. Compute capped history length `min(raw_length,L)` and prior exposure-activity
   buckets `[0], [1,9], [10,99], [100,+inf)` from completed dates only.
3. Form `(date, capped_length, activity_bucket)` strata. For singleton activity
   strata, deterministically coarsen/merge activity buckets within the **same exact
   capped length**. Finish all merging before assigning any donors, so the final
   groups are disjoint.
4. Within each resulting group of at least two users, apply a seeded random
   permutation followed by a nonzero cyclic rotation. This is a derangement:
   every recipient gets a different user's whole history. Do not reuse a donor
   twice in the daily permutation or change length silently.
5. If an entire positive-length stratum has fewer than two users, fail the build
   with evidence rather than return the recipient's history or improvise a fallback.
   The audit found no such length singletons among observed train/validation
   target user-days at L=20; this does not replace the full donor-pool assertion.
6. Empty-history targets stay empty in every variant. Preserve F123, target ID,
   target label, and candidate. Only history references change.
7. Assert donor user differs from recipient, retained lengths agree, donor dates
   precede the recipient cutoff, and selected events belong to the donor. Record
   bucket coarsening and length distributions by date.

The mapping preserves exact capped length per target and preserves the multiset
of donor histories within each user-level permutation. **It does not guarantee
identical impression-weighted category popularity**, because users have different
numbers of target exposures. Report token-frequency/age/activity balance and
coarsening rates; no test-based rematching. Do not claim that length matching
alone removes every popularity confound.

Apply the same mapping in training and evaluation for a separately trained
negative-control variant. This avoids confusing a test-only distribution shift
with an identity experiment. Use both mean/matched vs mean/mismatched and
attention/matched vs attention/mismatched where the budget permits, with identical
architecture/configuration within each pair. This controls parameter count within
the identity comparison; mean vs attention still has different attention parameters.

## 8. Proposed model comparison and validation selection

Do not retune/retrain existing LR, LightGBM, W&D, or DeepFM runs. Reuse the saved
DeepFM F123 checkpoint and its saved numeric scaler as the immutable no-sequence
reference. Its historical results remain untouched.

Recommended first implementation is a **frozen DeepFM context plus a trainable
sequence residual**:

```text
logit = frozen_DeepFM(F123) + sequence_residual(candidate_category, history)
sequence_residual(empty/all-unknown history) = 0 exactly
```

Mean and target attention share the same category token/query embedding table
within each model, embedding width 8, residual MLP, context checkpoint, scaler,
optimizer grid, dropout, seeds, and data. The new trainable category table can be
initialized from a copy of the frozen checkpoint's category table; it is shared
between history and query, never written back into the base model. Freeze the
context in eval mode, including dropout, and assert it receives no gradients.
Initialize the residual output to zero and mask the entire residual for empty
histories. Exact no-sequence predictions must match the saved DeepFM baseline
within a predeclared floating-point tolerance before any sequence training.

Attention scores candidate/category-history compatibility; it is a DIN-style
target-attention extension, **not a claim of reproducing original DIN**. Frozen
context narrows the claim to incremental category information on this baseline;
it may restrict achievable gains. Joint context retraining is not silently added.
This choice is part of the direction confirmation, since the old protocol's
context-only DIN design is different.

Predeclare after pilot feasibility, before looking at new validation scores:

- Four trainable conditions: mean/matched, attention/matched, mean/mismatched,
  attention/mismatched, alongside the already frozen no-sequence reference.
- Initial pair: L=20, LR=5e-4, AdamW, weight decay 1e-5, dropout 0.3, max 4 epochs,
  at most two consecutive non-improving validation epochs, batch initially 4096.
  Batch may be reduced for memory before the experiment grid is frozen.
- Bounded matched-source grid: L in `{20,50}` and LR in `{5e-4,2e-4}`, same grid
  for mean and attention, at most 4 configurations each. Choose a **common** L/LR
  by average validation LogLoss across the two matched aggregators, with fixed
  ties favoring L=20 then LR=5e-4. Each run selects its epoch by validation LogLoss.
  Report that this is a paired mechanism study, not separately optimized families.
- Do not independently tune the mismatch models; inherit the common config and
  train under the same stopping rule. Select all rules without test metrics.
- Final sequence seeds `{42,2026,3407}` fixed before final test; mismatch seed
  stays fixed across model seeds. These are new sequence runs, not permission
  to rerun the four completed tabular baselines.
- Save a freeze manifest with source/cutoff, vocabulary/checkpoint/scaler hashes,
  architecture, selected common config, seeds, donor policy, and planned comparisons.
  No test sequence diagnostics or predictions before this gate.

Formal raw predictions use the existing schema and evaluator. Add a backwards-
compatible validation-only `--splits val` option before sequence search, because
the present evaluator's raw mode always expects both val and test. Its historical
defaults/results must remain unchanged. Keep history-length diagnostics in a
separate report joined by impression ID.

After freeze, use the current calibration protocol as a separately recorded
historical-compatible method, clearly labeled impression holdout rather than OOF;
do not select calibration changes based on test. Report raw and calibrated results,
GAUC eligibility/coverage, existing cold-start quadrants, empty/nonempty history,
and fixed length bins `0,1,2–5,6–10,11–20,>20` using **raw** history length.
Predeclare paired user-cluster bootstrap for matched vs mismatched and attention
vs mean before final reporting. Neural three-seed variation is conditional on
the one existing frozen DeepFM checkpoint, not a three-seed tabular comparison.

## 9. Resources: measured facts and provisional estimates

Measured hardware: Apple M4 Max, 16 logical CPUs, 48 GiB unified RAM; approximately
477 GiB free at audit time. Source audit v2 took **2.48 seconds** and peaked at
**5.04 GiB RSS**. This is a narrow-column read/audit measurement, not a model or
sequence-construction benchmark. Existing neural training manifests report about
1.54/1.60 minutes, but their timing excludes scaler fitting and cannot establish
future DIN throughput or full wall-clock cost.

| Work / storage | Estimate or limit, pending pilot |
|---|---|
| Historical positive event store | 1.20M rows; eight fixed-width fields totaling 48 bytes give ~55 MiB before Parquet overhead/compression |
| History/index representation | 2,719,458 observed train/val target user-days; normalized event ranges + target/donor references expected below 1 GiB total, verify donor-pool expansion |
| Avoided dense expansion | Two uint32 arrays at N=26,557,961 and L=20 alone cost **3.96 GiB**, excluding masks/other fields; do not materialize them per impression |
| Builder working RAM | Plan for 6–12 GiB; audit's 5.04 GiB is a lower-level observation, not a guaranteed builder peak |
| Training unified RAM | Initial target 8–16 GiB total; measure MPS allocations and process RSS during smoke, stop/escalate if >24 GiB or sustained swapping |
| New files incl. model/prediction runs | Reserve 5 GiB initially; store references and reuse frozen checkpoint, do not replicate F123 inputs or base weights in every residual checkpoint |
| Build + audits | Planning allowance 5–30 minutes; stop and reassess if 30 minutes is exceeded |
| Model smoke / pilot | <=5 minutes for smoke; one full-train epoch timing pilot <=10 minutes before committing to a search budget |
| First sequence experiment block | Proposed **6 machine-hour cap**, within the old 20-hour sequence cap; reduce optional grid before sacrificing controls; ask before exceeding |

The time ranges are conservative planning allowances, not measured predictions.
Use pilot epoch time to price the actual number of runs (at most 8 matched-search
runs plus the necessary fixed-config controls/final seeds, reusing eligible
seed-42 runs). Include scaler/data loading, evaluation, serialization, calibration,
and three seeds in that estimate. If the plan does not fit, retain L=20 and the
single shared LR and document the reduced grid **before** score-based selection.
No search or large compute was started in Stage A.

## 10. Phase plan, correctness gates, and expected files

| Phase | Work | Exit evidence |
|---|---|---|
| A, current | Audit semantics and charter; design and conflict register | This document, read-only audit code/JSON, numeric handoff assertions, reviewed commit |
| B, after direction confirmation | Versioned positive-event store, daily indices, matched/mismatched references | Smoke then full train/val construction; provenance, cutoff, alignment and checksum assertions |
| C | Shared loading integration and residual mean/attention modules | Synthetic mask/gradient/identity tests; smoke; measured time/memory; no test |
| D | Bounded validation study; freeze; final predictions/evaluator/calibration | Unique runs, freeze manifest, test-only-after-freeze logs, segment metrics |
| E | Seed summary, paired user bootstrap, resources and reporting | Reproducible summary tables, uncertainty and protocol deviations; no retuning old models |

Stage-A validation completed: six named source assertions, twelve handoff metric
comparisons, Python compilation of all 19 existing scripts plus the new audit
script, local document-link/fence checks, and staged `git diff --check` passed.
The final audit report matches the audit script's SHA256. The four raw files and
15 inventoried Parquet files retained their sizes/mtimes, and the user-supplied
handoff retained its original SHA256. No model smoke test is claimed in Stage A;
there is no new model or feature builder yet. Only small docs/audit files are staged.

Stage B must test: every source record has `clk=1`; source IDs are unique; all
source users match their own matched history or the explicit mismatch donor;
timestamps/dates precede each cutoff; same-day and same-second boundary fixtures
are excluded correctly; May 6 history is empty; original target IDs and labels
remain unchanged; source category joins agree; unknown/PAD handling is neutral;
truncation keeps the latest L with stable ties; split references never expose
later events; interrupted date builds cannot be mistaken for completed ones.

Include random raw-row provenance checks: recover selected `source_impression_id`
records from `raw_sample.csv` in one bounded-memory sequential pass, since
`03_build_base_tables.py` originally assigns that ID before joins/splitting.
Recheck the raw ad mapping for those records. Do not use encoded IDs for ownership.
Stage A's full ad alignment is evidence for design; it does not substitute for
Stage B's sequence-level identity/cutoff audit.

Stage C must test shapes, finite logits, gradients only in the residual branch,
zero gradients for reserved token rows, padding invariance, empty-history exact
baseline recovery, all-UNKNOWN neutrality, stable serialization, and permutation
invariance for these non-positional aggregators. Matched/mismatched target F123
inputs must be byte-identical. No smoke metric is a formal model result.

Expected subsequent additions (names provisional until direction approval):

- `src/19_build_f4_click_sequences.py`: independent, resumable/date-keyed builder;
  refuse existing completed run directories unless exact manifest reuse is proven.
- `src/20_audit_f4_sequences.py`: source identity, cutoff and donor audits.
- `src/ctr_nn_common.py` and `src/21_train_sequence_ctr.py`: narrowly extract reusable
  loader/scaler/model helpers without changing tabular behavior; separate sequence module.
- `tests/test_f4_sequences.py`, `tests/test_sequence_ctr.py`: boundary and mechanism tests.
- `data/processed/f4_click_daily_v1/`, `artifacts/models/<unique_sequence_run>/`,
  `artifacts/predictions/<unique_sequence_run>/`: local large artifacts.
- `results/f4_click_daily_v1/`, `results/<unique_sequence_run>/`: committed small
  manifests, audit summaries, config, metrics, resource measurements.
- Minimal evaluator split selection and separate history-length analysis; document
  its compatibility checks before any formal evaluation.

Never overwrite a completed formal output. A resumed build must validate source
fingerprints/configuration before reusing completed partitions. New direction,
larger resource budget, time semantics, or test usage needs confirmation. Each
validated stage gets a reviewed commit and push without repeated routine approvals.

## 11. Direction confirmation requested before Stage B

Confirm or amend the following concrete proposed direction:

1. Prior completed-day **positive advertising feedback**, category-only, L=20
   first, with the explicitly stated next-day label-availability assumption.
2. Fixed-cutoff, exact-length different-user derangement with past-activity
   matching; empty histories remain empty and unmatchable strata fail explicitly.
3. Frozen existing DeepFM context + new mean/target-attention residuals, including
   corresponding mismatch controls; no retuning of the four existing models.
4. Pilot-governed validation budget capped at 6 machine-hours; test remains closed
   to new sequence analysis until the configuration-freeze gate.

Until this direction is confirmed, Stage B/C code, sequence materialization,
training, and new test evaluation remain unstarted. This stop is the explicit
handoff §15 gate, not an added approval requirement for ordinary implementation.
