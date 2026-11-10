> **Archived working note.** Internal agent instructions / planning from the
> September 2026 rebuild, kept for transparency. It may describe states that are
> no longer current; see the [README](../../README.md) and
> [final report](../FINAL_REPORT.md) for the results.

# CTR project execution

Read `docs/process/CTR_CODEX_PROJECT_HANDOFF.md` completely before starting a new project
phase. It is the long-term execution charter. Current user instructions take
precedence; repository code, manifests, Git history, and saved results take
precedence over factual claims in documents. Record conflicts explicitly.

## Current state: closed; audit clarification only

E1–E6 and the authorized first F5 test are COMPLETE. All authorization sections
below are historical and superseded by this closure. Do not train, run formal
test inference, bootstrap/resampling, fit calibration, select models or rerun
serving benchmarks. Only read-only consistency audits, synthetic-fixture tests,
document corrections and PDF rebuilds are allowed. Frozen code/data/checkpoints,
predictions, once-only markers and original formal CSV/JSON remain immutable.

New commits must not include any AI `Co-Authored-By`, `Generated-By` or similar
trailer. Preserve all historical commit messages and attribution. Never rewrite
history. The current revision is on `fix/f5-audit-clarifications`; push that branch
only and stop before merging into main pending the user's confirmation.

## Historical authorization: E1-E6 extension (completed)

The user authorized a post-release extension on branch `feature/f5-behavior-sequence`,
from release baseline `cce6966`. Authorized work, in order:

- **E1** full-population cross-source namespace audit, scripted and committed.
- **E2** documentation revision: top summaries, narrowed F4/attention scope,
  qualified DSIN background comparison, corrected isotonic/COPC/calibration text.
- **E3** paired user-cluster bootstrap of the **saved** LR vs DeepFM F123 test
  predictions. No retraining and no new test inference.
- **E4/E5** a new **F5** information budget: `behavior_log` category + behavior-type
  sequences, L=200, completed-day cutoff, five conditions on the frozen DeepFM
  backbone, seeds 42/2026/3407. F5 **adds** a budget; it does not replace or
  rewrite F4.
- **E6** unified serving benchmark, experiment registry, final release update.

All existing frozen evidence is immutable: the four tabular families and their
predictions, the 15 F4 checkpoints, the F4 formal test predictions, `badecf4`
bootstrap/calibration evidence, the once-only test markers, and every original
CSV/JSON under existing formal result directories. New work writes only to new
`*_v1` directories listed in the user instruction.

Proceed autonomously through every non-test stage. The **single** mandatory pause
is after F5 validation selection and full config/checkpoint freeze, before the
first formal F5 test inference. Ordinary engineering choices are made
conservatively, recorded, and not escalated.

## Release closure (superseded by the authorization above; retained as history)

Experiments and deep dives are complete. Only final release consistency checks,
Chinese PDF delivery, synthetic-fixture tests, documentation, commit and push
are authorized. No new models, retraining, test inference, bootstrap, calibration
fitting or new post-hoc analysis. After delivery stop project expansion.
Read `docs/RELEASE_AUDIT.md` for final files and read-only verification commands.

## Historical F4 authorization (completed; superseded by closure)

The authorized formal test is now COMPLETE: 15 checkpoints, one pass each.
Read `docs/DIN_F4_TEST_RESULTS.md` and `results/sequence_test_v2/manifest.json`.
Do not run test inference again. Existing predictions are the sole input to
further reporting. No post-test model/configuration changes are authorized.

The completed phase authorized user-cluster paired bootstrap, fixed group
diagnostics, and validation-only probability calibration of saved predictions.
Read `docs/DIN_F4_POSTHOC_PROTOCOL.md` and the immutable protocol in
`results/sequence_posthoc_v1/protocol.json`. No new CTR training or test inference.
Calibrator fitting is allowed only on validation, with the existing selection
recipe; retain every selected calibrator regardless of its test result. Zero is
one distinct predictor and gets one calibrator, not three independent seeds.
Keep sampling uncertainty separate from conditional residual-training seed std.

As of the next user authorization, `badecf4` and `DIN_F4_TEST_RESULTS.md` are
immutable evidence. No further bootstrap, calibration fitting, CTR training or
test inference. Follow `docs/DEEP_DIVE_RESEARCH_PLAN.md` through D2–D5: saved-
prediction diagnostics, validation-only frozen component reliance, descriptive
temporal/repeat analysis, then final reporting. Commit/push each verified stage.
Do not turn diagnostic interventions into model selection. The current user
instruction supersedes the completed calibration authorization above.

D2–D5 are now completed. Read `docs/FINAL_REPORT.md` and the D2/D3/D4 reports.
The frozen research plan and `badecf4` DIN evidence remain unchanged. Further
work must not automatically rerun completed diagnostic stages or reinterpret
validation component deletions as newly trained formal models. Read-only delivery
checks use `src/37_verify_frozen_delivery.py`; plots use saved summaries only.

The previous completed phase authorized the hard-zero same-pipeline control at
seeds 42/2026/3407, then exactly one formal test pass for five conditions after
validation, smoke/reload checks and a pre-test commit/push pass. Do not ask again
once these prerequisites pass. The current freeze is
`results/sequence_selection_v2/freeze_manifest.json`; v1 remains immutable history.
Mean matched is the primary model; attention remains a mechanism experiment.
Follow the five ordered comparisons in `results/sequence_selection_v2/comparison_plan.json`.
No model, hyperparameter, epoch, sequence length, priority or seed changes after
test. Existing inference start markers forbid automatic re-inference: reports
must reuse saved predictions. All conditions share one frozen DeepFM backbone;
the hard-zero control is deterministic, not three independently fitted DeepFMs.

## Historical Stage B/C authorization

Read `docs/DIN_F4_DESIGN.md` for the Stage-A audit, proposed F4 design, and current
approval gate. At its initial publication, the design is a recommendation;
The user confirmed Stage B/C on 2026-09-26, including full F4 construction,
smoke/audits, model pilot and validation-only tuning. Do not ask again for those
actions. Stop after configuration selection and commit/push the freeze evidence
BEFORE the first formal sequence test. No DIN test is currently authorized.

Stages B/C are now validated and frozen. Read `docs/DIN_F4_VALIDATION.md` and
`results/sequence_selection_v1/freeze_manifest.json` for the selected L=20,
LR=5e-4 recipe and all 12 checkpoint hashes. No further sequence tuning is
authorized after this freeze. Wait for explicit user authorization for the first
formal test; retain the existing frozen checkpoints and completed artifacts.

- Do not retune the four completed tabular model families.
- Do not select features, cutoffs, hyperparameters, or fallbacks using test results.
- Do not overwrite unarchived formal outputs or delete important data.
- Ask only for changes to research questions, time semantics, test usage,
  substantial extra compute, destructive actions, or an explicit charter gate.
- Advance one verifiable phase at a time. Ordinary implementation is authorized.
- Run relevant correctness checks and smoke tests before full experiments.
- Use the shared evaluator; retain unique run names, manifests, and evidence.
- Before each stage commit, run relevant checks, Python compilation,
  `git diff --check`, and inspect the diff/status. Commit and push validated stages
  to `origin/main`, checking that large data/models/predictions are not staged.
- Preserve historical protocol deviations instead of silently rewriting history.
