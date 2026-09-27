"""Report the frozen F5 test metrics, three-seed summaries and the six comparisons.

Reads only saved predictions. No inference, no retraining, no reselection.
"""

from __future__ import annotations

import json
import statistics as stats
import subprocess
import sys

import numpy as np
import polars as pl

from f5_sequences import ROOT, sha256, write_json
from f5_test_common import STUDY, TEST_ROWS, TEST_STUDY
from project_environment import require_tested_environment

TEST_FREEZE = STUDY / "test_freeze_manifest.json"
F4_TEST = ROOT / "results/sequence_test_v2/manifest.json"
METRIC_KEYS = ("auc", "logloss", "pr_auc", "gauc", "copc", "ece")

# Conditions that participate in the six frozen comparisons.
F5_CONDITIONS = {
    "behavior_mean_matched", "behavior_mean_mismatched",
    "behavior_attention_matched", "behavior_attention_mismatched",
    "behavior_zero_same_pipeline",
}
F4_REFERENCE_CONDITION = "mean_matched"


def evaluate(run: str, logs) -> dict:
    """Run the shared evaluator on one saved test prediction set."""
    evaluation = ROOT / "results" / run / "evaluation"
    if not (evaluation / "evaluation_manifest.json").exists():
        with (logs / f"{run}_evaluate.log").open("x") as log:
            subprocess.run(
                [sys.executable, "-B", "src/10_evaluate_predictions.py",
                 "--run-name", run, "--splits", "test"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
    manifest = json.loads((evaluation / "evaluation_manifest.json").read_text())
    if manifest["splits"] != ["test"]:
        raise ValueError(f"{run} evaluation covered {manifest['splits']}")
    overall = pl.read_csv(evaluation / "overall_metrics.csv").row(0, named=True)
    gauc = pl.read_csv(evaluation / "gauc_summary.csv").row(0, named=True)
    if overall["rows"] != TEST_ROWS:
        raise ValueError(f"{run} evaluated {overall['rows']} rows")
    row = {key: overall[key] for key in ("auc", "logloss", "pr_auc", "copc", "ece", "rows", "clicks")}
    row.update({k: gauc[k] for k in ("gauc", "eligible_users", "eligible_row_coverage")})
    return row


def summarise(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    summary = []
    for condition, members in sorted(grouped.items()):
        record = {"condition": condition, "seeds": len(members)}
        for key in METRIC_KEYS:
            values = [m[key] for m in members]
            record[f"{key}_mean"] = stats.mean(values)
            record[f"{key}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
        summary.append(record)
    return summary


def condition_mean(summary: list[dict], condition: str, key: str) -> float:
    for record in summary:
        if record["condition"] == condition:
            return record[f"{key}_mean"]
    raise KeyError(condition)


def main() -> None:
    environment = require_tested_environment("F5 test reporting")
    freeze = json.loads(TEST_FREEZE.read_text())
    inference = json.loads((TEST_STUDY / "manifest.json").read_text())
    if not inference["complete"] or inference["smoke"] or inference["split"] != "test":
        raise ValueError("the formal test inference manifest is not in the expected state")
    if inference["freeze_sha256"] != sha256(TEST_FREEZE):
        raise ValueError("the test freeze changed after inference")
    logs = TEST_STUDY / "logs"
    logs.mkdir(exist_ok=True)

    rows = []
    for record in inference["records"]:
        if sha256(ROOT / "artifacts/predictions" / record["run_name"] / "test_predictions.parquet") != record["prediction_sha256"]:
            raise ValueError(f"{record['run_name']} prediction changed after inference")
        metrics = evaluate(record["run_name"], logs)
        rows.append(
            {
                "run_name": record["run_name"], "training_run": record["training_run"],
                "condition": record["condition"], "seed": record["seed"],
                "pooling": record["pooling"], "source": record["source"],
                "zero_sequence": record["zero_sequence"], "budget": "F5", **metrics,
            }
        )
    if {r["condition"] for r in rows} != F5_CONDITIONS:
        raise ValueError("unexpected F5 condition set")

    # The F4 reference for comparison 6 reuses its frozen test predictions.
    f4_manifest = json.loads(F4_TEST.read_text())
    f4_rows = []
    for record in f4_manifest["records"]:
        if record["condition"] != F4_REFERENCE_CONDITION:
            continue
        metrics = evaluate(record["run_name"], logs)
        f4_rows.append(
            {
                "run_name": record["run_name"], "condition": "f4_click_mean_matched",
                "seed": record["seed"], "pooling": "mean", "source": "matched",
                "zero_sequence": False, "budget": "F4", **metrics,
            }
        )
    if len(f4_rows) != 3:
        raise ValueError(f"expected 3 F4 mean matched seeds, found {len(f4_rows)}")

    all_rows = rows + f4_rows
    summary = summarise(all_rows)
    pl.DataFrame(all_rows).write_csv(TEST_STUDY / "test_metrics.csv")
    pl.DataFrame(summary).write_csv(TEST_STUDY / "test_seed_summary.csv")

    # Paired per-seed differences, which describe training randomness rather than
    # user-sampling uncertainty.
    by_condition_seed = {(r["condition"], r["seed"]): r for r in all_rows}
    comparisons = freeze["comparison_plan"]
    effect_rows = []
    paired_rows = []
    for comparison in comparisons:
        left, right = comparison["left"], comparison["right"]
        for key in METRIC_KEYS:
            effect_rows.append(
                {
                    "id": comparison["id"], "priority": comparison["priority"],
                    "left": left, "right": right, "metric": key,
                    "effect": condition_mean(summary, left, key) - condition_mean(summary, right, key),
                    "left_mean": condition_mean(summary, left, key),
                    "right_mean": condition_mean(summary, right, key),
                }
            )
        # Zero is one deterministic predictor, so a per-seed pairing against it
        # describes only the left side's training randomness.
        for seed in (42, 2026, 3407):
            left_row = by_condition_seed.get((left, seed))
            right_row = by_condition_seed.get((right, seed))
            if left_row is None or right_row is None:
                continue
            paired_rows.append(
                {
                    "id": comparison["id"], "left": left, "right": right, "seed": seed,
                    **{key: left_row[key] - right_row[key] for key in METRIC_KEYS},
                }
            )
    pl.DataFrame(effect_rows).write_csv(TEST_STUDY / "comparison_effects.csv")
    paired = pl.DataFrame(paired_rows)
    paired.write_csv(TEST_STUDY / "paired_seed_differences.csv")

    consistency = []
    for comparison in comparisons:
        subset = paired.filter(pl.col("id") == comparison["id"])
        if subset.height == 0:
            continue
        for key in METRIC_KEYS:
            values = subset[key].to_numpy()
            favourable = values < 0 if key in ("logloss", "ece") else values > 0
            consistency.append(
                {
                    "id": comparison["id"], "left": comparison["left"],
                    "right": comparison["right"], "metric": key,
                    "mean": float(values.mean()),
                    "training_seed_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "favourable_seeds": int(favourable.sum()), "seeds": int(len(values)),
                    "all_favourable": bool(favourable.all()),
                }
            )
    pl.DataFrame(consistency).write_csv(TEST_STUDY / "paired_seed_direction_consistency.csv")

    write_json(
        TEST_STUDY / "report_manifest.json",
        {
            "complete": True, "split": "test", "budget": "F5",
            "rows_per_condition": TEST_ROWS,
            "f5_conditions": sorted(F5_CONDITIONS),
            "f4_reference_condition": F4_REFERENCE_CONDITION,
            "comparisons": len(comparisons),
            "metrics": list(METRIC_KEYS),
            "test_freeze_sha256": sha256(TEST_FREEZE),
            "inference_manifest_sha256": sha256(TEST_STUDY / "manifest.json"),
            "f4_test_manifest_sha256": sha256(F4_TEST),
            "uncertainty_note": (
                "Seed standard deviations describe three residual seeds on one "
                "shared frozen backbone. The zero control is one deterministic "
                "predictor, so its standard deviation is structural. User-sampling "
                "uncertainty is a separate bootstrap and is never merged with these."
            ),
            "comparison_6_note": (
                "F5 mean matched against F4 mean matched contrasts two information "
                "sources on one shared backbone. It is not an architecture "
                "comparison: retained length is 200 against 20, mean retained "
                "tokens about 185 against 4.69, and empty-history share about "
                "0.44% against 30.21%."
            ),
            "environment": environment,
            "no_reselection": "No model, length, learning rate, donor or primary model changed.",
        },
    )
    print(pl.DataFrame(summary).select("condition", "seeds", "auc_mean", "auc_std",
                                      "logloss_mean", "logloss_std"), flush=True)
    print(pl.DataFrame(effect_rows).filter(pl.col("metric").is_in(["auc", "logloss"])).select(
        "id", "priority", "left", "right", "metric", "effect"), flush=True)


if __name__ == "__main__":
    main()
