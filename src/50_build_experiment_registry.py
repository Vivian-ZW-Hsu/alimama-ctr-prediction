"""Build a machine-readable registry of every experiment directory under results/.

The repository holds well over a hundred result directories of several different
shapes: tabular model runs carry `run_manifest.json`, sequence and data stages
carry `manifest.json`, and tuning and smoke directories are local-only. Reading
them by hand is the navigation problem this registry removes.

Classification is driven by the declared prefix rules in STAGES, so a reader can
see exactly why a directory was placed in a stage. Nothing is inferred silently:
a directory that matches no rule is recorded as `unclassified` rather than
dropped. This script only reads; it never writes into an existing run directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import polars as pl

from project_environment import require_tested_environment

from f5_sequences import ROOT, sha256, write_json

RESULTS = ROOT / "results"
OUTPUT_CSV = RESULTS / "experiment_registry.csv"
OUTPUT_INDEX = ROOT / "docs" / "EXPERIMENT_INDEX.md"

TRAIN_DATES = "2017-05-06..05-11"
VALIDATION_DATE = "2017-05-12"
TEST_DATE = "2017-05-13"

# (stage, matcher, feature_budget, notes). Order matters: the first match wins.
STAGES: list[tuple[str, str, str, str]] = [
    ("data", "model_input", "F1+F2", "shared encoded model input"),
    ("data", "model_input_f3", "F1+F2+F3", "shared encoded model input with past feedback"),
    ("data", "past_ctr", "F3", "completed-day entity feedback"),
    ("data", "feature_profile", "F1+F2", "feature role and coverage profile"),
    ("data", "f4_click_daily_v1", "F4", "clicked-ad category references, L=20"),
    ("data", "f5_behavior_v1", "F5", "behavior_log category+type references, L=200"),
    ("data", "behavior_namespace_audit_v1", "-", "E1 cross-source namespace audit"),
    ("context_cache", "sequence_context_v1", "F1+F2+F3", "frozen DeepFM logits for F4"),
    ("context_cache", "f5_context_v1", "F1+F2+F3", "frozen DeepFM logits for F5"),
    ("tabular_information_ablation", "lr_f1", "F1", "LR information layer"),
    ("tabular_information_ablation", "lr_f1_f2", "F1+F2", "LR information layer"),
    ("tabular_information_ablation", "lr_f1_f2_f3", "F1+F2+F3", "LR information layer"),
    ("tabular_benchmark", "lightgbm_f1_f2_f3", "F1+F2+F3", "frozen single seed"),
    ("tabular_benchmark", "wide_deep_f1_f2_f3", "F1+F2+F3", "frozen single seed"),
    ("tabular_benchmark", "deepfm_f1_f2_f3", "F1+F2+F3", "frozen single seed"),
    ("f4_sequence_validation", "sequence_selection_v", "F1+F2+F3+F4", "F4 validation freeze"),
    ("f4_sequence_validation", "seq_", "F1+F2+F3+F4", "F4 validation run"),
    ("f4_sequence_test", "sequence_test_v", "F1+F2+F3+F4", "F4 once-only formal test"),
    ("f4_posthoc", "sequence_posthoc_v", "F1+F2+F3+F4", "frozen bootstrap and calibration"),
    ("f5_sequence_validation", "f5_selection_v", "F1+F2+F3+F5", "F5 validation freeze"),
    ("f5_sequence_validation", "f5b_", "F1+F2+F3+F5", "F5 validation run"),
    ("f5_sequence_validation", "f5_pilot", "F1+F2+F3+F5", "F5 resource pilot, no validation scoring"),
    ("frozen_model_bootstrap", "frozen_model_bootstrap_v", "F1+F2+F3", "E3 user-cluster intervals"),
    ("serving_benchmark", "serving_benchmark_v", "-", "E6 unified inference benchmark"),
    ("deep_dive", "deep_dive_", "-", "saved-prediction diagnostics"),
    ("release", "release_audit_v", "-", "release consistency audit"),
    ("diagnostic", "lr_weekday_ablation", "F1", "weekday exclusion diagnostic"),
    ("diagnostic", "lr_tuning", "F1+F2+F3", "LR tuning summary"),
    ("diagnostic", "nn_unknown_embedding_diagnostic", "F1+F2+F3", "pre-fix UNKNOWN embedding"),
    ("tuning", "lr_tune_", "F1+F2+F3", "local-only tuning run"),
    ("tuning", "lr_boundary_", "F1+F2+F3", "local-only tuning run"),
    ("tuning", "lgbm_f123_", "F1+F2+F3", "local-only tuning run"),
    ("tuning", "wide_deep_f123_", "F1+F2+F3", "local-only tuning run"),
    ("tuning", "deepfm_f123_", "F1+F2+F3", "local-only tuning run"),
    ("f4_design_audit", "din_f4_audit_", "F4", "Stage A read-only F4 source audit"),
    ("f4_sequence_validation", "sequence_attention_pilot", "F1+F2+F3+F4", "F4 attention resource pilot"),
    ("legacy", "logistic_regression", "F1+F2", "superseded LR run name"),
    ("legacy", "f4_click_daily_v1_", "F4", "superseded F4 build attempt"),
]


# (stage, suffix, feature_budget, notes). Checked BEFORE the prefix rules, because
# a per-run formal test directory is named `seq_<condition>_test_v2` and the `seq_`
# prefix rule would otherwise classify it as a validation run.
SUFFIX_STAGES: list[tuple[str, str, str, str]] = [
    ("f4_sequence_test", "_test_v2", "F1+F2+F3+F4", "F4 once-only formal test run"),
]


def classify(name: str) -> tuple[str, str, str]:
    for stage, suffix, budget, note in SUFFIX_STAGES:
        if name.endswith(suffix):
            return stage, budget, note
    best = None
    for stage, prefix, budget, note in STAGES:
        if name == prefix or name.startswith(prefix):
            if best is None or len(prefix) > len(best[1]):
                best = (stage, prefix, budget, note)
    if best is None:
        return "unclassified", "-", "no declared prefix rule matched"
    return best[0], best[2], best[3]


def sequence_attributes(name: str, config: dict | None) -> dict[str, object]:
    if config is None:
        return {"pooling": None, "sequence_source": None, "seed": None, "zero_sequence": None}
    return {
        "pooling": config.get("pooling"),
        "sequence_source": config.get("sequence_source"),
        "seed": config.get("seed"),
        "zero_sequence": config.get("zero_sequence"),
    }


def tracked_paths() -> set[str]:
    listing = subprocess.run(
        ["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return {line.split("/")[1] for line in listing.stdout.splitlines() if "/" in line}


def main() -> None:
    environment = require_tested_environment("E6 experiment registry")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true",
                        help="also list directories that Git does not track")
    args = parser.parse_args()

    committed = tracked_paths()
    rows = []
    for directory in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        name = directory.name
        is_committed = name in committed
        if not is_committed and not args.include_untracked:
            continue
        stage, budget, note = classify(name)

        manifest_path = None
        for candidate in ("manifest.json", "run_manifest.json"):
            if (directory / candidate).exists():
                manifest_path = directory / candidate
                break
        manifest = json.loads(manifest_path.read_text()) if manifest_path else None
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else None

        checkpoint = ROOT / "artifacts/models" / name
        predictions = ROOT / "artifacts/predictions" / name
        calibration = ROOT / "artifacts/calibrators" / name
        test_prediction = predictions / "test_predictions.parquet"
        validation_prediction = predictions / "val_predictions.parquet"

        status = "complete" if (manifest or {}).get("complete") else (
            "complete" if manifest_path and "run_name" in (manifest or {}) else "unknown"
        )
        frozen = bool((manifest or {}).get("complete")) and stage not in ("tuning", "diagnostic")

        rows.append(
            {
                "experiment_id": name,
                "stage": stage,
                "model_family": (manifest or {}).get("model_type")
                or (config or {}).get("pooling") and f"sequence_residual_{config['pooling']}"
                or (manifest or {}).get("budget")
                or "-",
                "feature_budget": budget,
                **sequence_attributes(name, config),
                "train_dates": TRAIN_DATES,
                "validation_date": VALIDATION_DATE,
                "test_date": TEST_DATE if test_prediction.exists() else None,
                "status": status,
                "frozen": frozen,
                "git_tracked": is_committed,
                "smoke": bool((manifest or {}).get("smoke")) or name.endswith("_smoke"),
                "checkpoint_path": str(checkpoint.relative_to(ROOT)) if checkpoint.exists() else None,
                "prediction_path": str(predictions.relative_to(ROOT)) if predictions.exists() else None,
                "validation_prediction_exists": validation_prediction.exists(),
                "metrics_path": (
                    str((directory / "evaluation").relative_to(ROOT))
                    if (directory / "evaluation").exists() else None
                ),
                "calibration_path": str(calibration.relative_to(ROOT)) if calibration.exists() else None,
                "manifest_path": str(manifest_path.relative_to(ROOT)) if manifest_path else None,
                "test_inference_count": 1 if test_prediction.exists() else 0,
                "notes": note,
            }
        )

    table = pl.DataFrame(rows)
    table.write_csv(OUTPUT_CSV)

    by_stage = table.group_by("stage").agg(
        pl.len().alias("directories"),
        pl.col("test_inference_count").sum().alias("test_inferences"),
    ).sort("stage")

    lines = [
        "# Experiment index",
        "",
        "Generated by `src/50_build_experiment_registry.py` from the manifests that",
        "each run wrote. The machine-readable form is",
        "[`results/experiment_registry.csv`](../results/experiment_registry.csv).",
        "",
        "Directories are classified by the declared prefix rules in that script, so a",
        "reader can see why each one landed in its stage. A directory matching no rule",
        "is recorded as `unclassified` rather than dropped. Local-only tuning and smoke",
        "directories are Git-ignored and appear only when the script is run with",
        "`--include-untracked`.",
        "",
        "## Stages",
        "",
        "| Stage | Committed directories | Formal test inferences |",
        "|---|---:|---:|",
    ]
    for row in by_stage.iter_rows(named=True):
        lines.append(f'| `{row["stage"]}` | {row["directories"]} | {row["test_inferences"]} |')
    lines += [
        "",
        f'Total committed directories: **{table.height}**. Formal test inferences: '
        f'**{int(table["test_inference_count"].sum())}**, each a single once-only pass.',
        "",
        "## Committed experiments",
        "",
        "| Experiment | Stage | Budget | Pooling | Source | Seed | Test | Notes |",
        "|---|---|---|---|---|---:|:--:|---|",
    ]
    for row in table.sort("stage", "experiment_id").iter_rows(named=True):
        lines.append(
            f'| `{row["experiment_id"]}` | {row["stage"]} | {row["feature_budget"]} | '
            f'{row["pooling"] or "-"} | {row["sequence_source"] or "-"} | '
            f'{row["seed"] if row["seed"] is not None else "-"} | '
            f'{"yes" if row["test_inference_count"] else "-"} | {row["notes"]} |'
        )
    lines.append("")
    OUTPUT_INDEX.write_text("\n".join(lines) + "\n")

    write_json(
        RESULTS / "experiment_registry_manifest.json",
        {
            "complete": True,
            "directories": table.height,
            "include_untracked": args.include_untracked,
            "stages": {row["stage"]: row["directories"] for row in by_stage.iter_rows(named=True)},
            "total_test_inferences": int(table["test_inference_count"].sum()),
            "registry_sha256": sha256(OUTPUT_CSV),
            "index_sha256": sha256(OUTPUT_INDEX),
            "classification": "declared prefix rules in src/50_build_experiment_registry.py",
        },
    )
    print(by_stage, flush=True)
    print(f"Registry: {OUTPUT_CSV.relative_to(ROOT)}  Index: {OUTPUT_INDEX.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
