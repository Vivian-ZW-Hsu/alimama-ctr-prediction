# CTR project execution

Read `docs/CTR_CODEX_PROJECT_HANDOFF.md` completely before starting a new project
phase. It is the long-term execution charter. Current user instructions take
precedence; repository code, manifests, Git history, and saved results take
precedence over factual claims in documents. Record conflicts explicitly.

## Current authorization (supersedes the historical gate below)

The authorized formal test is now COMPLETE: 15 checkpoints, one pass each.
Read `docs/DIN_F4_TEST_RESULTS.md` and `results/sequence_test_v2/manifest.json`.
Do not run test inference again. Existing predictions are the sole input to
further reporting. No post-test model/configuration changes are authorized.

The current phase authorizes user-cluster paired bootstrap, fixed group
diagnostics, and validation-only probability calibration of saved predictions.
Read `docs/DIN_F4_POSTHOC_PROTOCOL.md` and the immutable protocol in
`results/sequence_posthoc_v1/protocol.json`. No new CTR training or test inference.
Calibrator fitting is allowed only on validation, with the existing selection
recipe; retain every selected calibrator regardless of its test result. Zero is
one distinct predictor and gets one calibrator, not three independent seeds.
Keep sampling uncertainty separate from conditional residual-training seed std.

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
