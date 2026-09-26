# CTR project execution

Read `docs/CTR_CODEX_PROJECT_HANDOFF.md` completely before starting a new project
phase. It is the long-term execution charter. Current user instructions take
precedence; repository code, manifests, Git history, and saved results take
precedence over factual claims in documents. Record conflicts explicitly.

Read `docs/DIN_F4_DESIGN.md` for the Stage-A audit, proposed F4 design, and current
approval gate. At its initial publication, the design is a recommendation;
Stage B must wait for the user's direction confirmation required by handoff §15.
Record any subsequent confirmation before advancing; do not ask again once given.

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
