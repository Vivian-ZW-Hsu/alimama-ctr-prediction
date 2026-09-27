"""Paired user-cluster bootstrap over the frozen F5 test predictions.

Reuses the exact bootstrap kernel. Reads saved predictions only: no inference, no
retraining, no reselection. The intervals describe user-sampling uncertainty
conditional on the frozen checkpoints and the single test day; three-seed training
randomness is reported separately by src/54 and never merged with these.
"""

from __future__ import annotations

import json
import subprocess
import time

import numpy as np
import polars as pl

from ctr_posthoc import evaluator, holm, write_common, write_rank_model
from f5_sequences import ROOT, sha256, write_json
from f5_test_common import STUDY, TEST_ROWS, TEST_STUDY
from project_environment import require_tested_environment

DATA = ROOT / "data/processed/f5_sequence_test_v1_bootstrap"
TEST_FREEZE = STUDY / "test_freeze_manifest.json"
F4_TEST = ROOT / "results/sequence_test_v2/manifest.json"
METRICS = ("auc", "logloss", "pr_auc", "gauc")
REPLICATES = 1000
PILOT = 5
SEED = 20260927
THREADS = 8
F4_REFERENCE = "f4_click_mean_matched"


def predictors() -> list[dict]:
    """Thirteen distinct F5 predictors plus the three F4 mean-matched references."""
    freeze = json.loads(TEST_FREEZE.read_text())
    inference = json.loads((TEST_STUDY / "manifest.json").read_text())
    by_training = {r["training_run"]: r for r in inference["records"]}
    records = []
    for checkpoint in freeze["checkpoints"]:
        if checkpoint["zero_sequence"] and checkpoint["seed"] != 42:
            continue
        record = by_training[checkpoint["run_name"]]
        records.append(
            {"run_name": record["run_name"], "condition": record["condition"],
             "seed": record["seed"], "budget": "F5"}
        )
    f4 = json.loads(F4_TEST.read_text())
    for record in f4["records"]:
        if record["condition"] == "mean_matched":
            records.append(
                {"run_name": record["run_name"], "condition": F4_REFERENCE,
                 "seed": record["seed"], "budget": "F4"}
            )
    if len(records) != 16:
        raise ValueError(f"expected 16 distinct predictors, found {len(records)}")
    return records


def main() -> None:
    environment = require_tested_environment("F5 test bootstrap")
    manifest_path = TEST_STUDY / "bootstrap_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Bootstrap already complete")
    if DATA.exists():
        raise FileExistsError("Bootstrap inputs are immutable; inspect before resuming")
    DATA.mkdir(parents=True)
    started = time.perf_counter()

    context = pl.read_parquet(
        ROOT / "data/processed/base/test.parquet",
        columns=["impression_id", "clk", "user_raw"],
    ).sort("impression_id")
    if context.height != TEST_ROWS:
        raise ValueError("unexpected test row count")
    user_values, user = np.unique(context["user_raw"].to_numpy(), return_inverse=True)
    user = user.astype(np.uint32)
    n_users = len(user_values)
    labels = context["clk"].to_numpy().astype(np.uint8)
    counts = np.bincount(user, minlength=n_users)
    clicks = np.bincount(user, weights=labels, minlength=n_users)
    eligible = (clicks > 0) & (clicks < counts)
    write_common(DATA / "common.bin", context.height, n_users, 16, counts, counts * eligible)

    records = []
    for index, record in enumerate(predictors()):
        # The evaluator returns rows in file order; impression_id is a global
        # raw_sample row index and is scattered across days, so an explicit sort is
        # required before aligning against the sorted context.
        frame = evaluator.load_and_validate("test", record["run_name"], "raw").sort("impression_id")
        np.testing.assert_array_equal(
            frame["impression_id"].to_numpy(), context["impression_id"].to_numpy()
        )
        np.testing.assert_array_equal(frame["clk"].to_numpy(), labels)
        probabilities = frame["prediction"].to_numpy().astype(np.float64)
        gauc, users = evaluator.build_gauc("test", frame.with_columns(context["user_raw"]))
        positions = np.searchsorted(user_values, users["user_raw"].to_numpy())
        numerator = np.zeros(n_users, dtype=float)
        numerator[positions] = users["rows"].to_numpy() * users["user_auc"].to_numpy()
        write_rank_model(DATA / f"model_{index}.bin", user, labels, probabilities, n_users, numerator)
        records.append({**record, "index": index, "saved_gauc": gauc["gauc"],
                        "prediction_sha256": sha256(
                            ROOT / "artifacts/predictions" / record["run_name"] / "test_predictions.parquet")})
        print(f'Prepared {index + 1}/16: {record["condition"]} seed {record["seed"]}', flush=True)

    executable = DATA / "cluster_bootstrap_metrics"
    subprocess.run(
        ["clang++", "-O3", "-std=c++17", "-pthread",
         str(ROOT / "src/cluster_bootstrap_metrics.cpp"), "-o", str(executable)],
        check=True,
    )

    def run(output, count):
        begin = time.perf_counter()
        subprocess.run([str(executable), str(DATA), str(output), str(count), str(SEED), str(THREADS)],
                       check=True)
        raw = np.fromfile(output, dtype="<f8").reshape(-1, 16, len(METRICS))
        return raw, time.perf_counter() - begin

    pilot, pilot_seconds = run(DATA / "pilot.bin", PILOT)
    projected = pilot_seconds * REPLICATES / PILOT
    if projected >= 6 * 3600:
        raise RuntimeError("Projected bootstrap exceeds six hours")
    if not np.isfinite(pilot).all():
        raise ValueError("non-finite pilot replicate")
    # Replicate 0 has unit weights, so it must reproduce the saved point metrics.
    saved = pl.read_csv(TEST_STUDY / "test_metrics.csv")
    for record in records:
        row = saved.filter(
            (pl.col("condition") == record["condition"]) & (pl.col("seed") == record["seed"])
        ).row(0, named=True)
        np.testing.assert_allclose(
            pilot[0, record["index"]], [row[m] for m in METRICS], rtol=1e-9, atol=1e-11,
            err_msg=f'{record["condition"]} seed {record["seed"]}',
        )

    full, elapsed = run(DATA / "full.bin", REPLICATES)
    np.testing.assert_array_equal(full[: len(pilot)], pilot)
    if not np.isfinite(full).all():
        raise ValueError("non-finite replicate; do not silently discard")

    index_by_condition: dict[str, list[int]] = {}
    for record in records:
        index_by_condition.setdefault(record["condition"], []).append(record["index"])

    def values(condition: str) -> np.ndarray:
        indices = index_by_condition[condition]
        expected = 1 if condition == "behavior_zero_same_pipeline" else 3
        if len(indices) != expected:
            raise ValueError(f"{condition} has {len(indices)} predictors, expected {expected}")
        return full[:, indices, :].mean(axis=1)

    plan = json.loads(TEST_FREEZE.read_text())["comparison_plan"]
    rows, draws = [], {"replicate": np.arange(REPLICATES, dtype=np.uint32)}
    for comparison in plan:
        delta = values(comparison["left"]) - values(comparison["right"])
        for position, metric in enumerate(METRICS):
            observed = float(delta[0, position])
            boot = delta[1:, position]
            low, high = np.quantile(boot, [0.025, 0.975], method="linear")
            p = (1 + np.sum(np.abs(boot - observed) >= abs(observed))) / (REPLICATES + 1)
            rows.append(
                {
                    "id": comparison["id"], "priority": comparison["priority"],
                    "left": comparison["left"], "right": comparison["right"], "metric": metric,
                    "effect": observed, "ci_low": float(low), "ci_high": float(high),
                    "bootstrap_std": float(np.std(boot, ddof=1)),
                    "p_centered_two_sided": float(p),
                    "holm_joint_primary_auc_logloss": None,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
            draws[f'comparison_{comparison["id"]}_{metric}'] = boot

    primary = [
        position for position, row in enumerate(rows)
        if row["priority"] == "primary" and row["metric"] in ("auc", "logloss")
    ]
    for position, value in zip(primary, holm([rows[p]["p_centered_two_sided"] for p in primary])):
        rows[position]["holm_joint_primary_auc_logloss"] = float(value)

    pl.DataFrame(rows).write_csv(TEST_STUDY / "bootstrap_intervals.csv")
    pl.DataFrame(draws).write_csv(TEST_STUDY / "bootstrap_deltas.csv")
    write_json(
        manifest_path,
        {
            "complete": True, "valid_replicates": REPLICATES, "invalid_replicates": 0,
            "seed": SEED, "threads": THREADS, "rows": context.height, "users": n_users,
            "eligible_users": int(eligible.sum()),
            "distinct_predictors": 16, "zero_predictors": 1,
            "f4_reference_predictors": 3,
            "elapsed_seconds": round(elapsed, 2),
            "pilot_reproduces_full_prefix": True,
            "unweighted_replicate_reproduces_saved_point_metrics": True,
            "intervals": "marginal percentile 95%, not simultaneous",
            "p_resolution": 1 / (REPLICATES + 1),
            "holm_scope": "the two primary comparisons on AUC and LogLoss, four hypotheses",
            "uncertainty": (
                "User-sampling conditional on all fixed checkpoints and the single "
                "test day. Not training-seed uncertainty, which src/54 reports "
                "separately over three residual seeds on one shared backbone."
            ),
            "predictors": records,
            "binary_sha256": sha256(executable),
            "raw_results_sha256": sha256(DATA / "full.bin"),
            "environment": environment,
            "total_seconds": round(time.perf_counter() - started, 2),
        },
    )
    print(pl.DataFrame(rows).filter(pl.col("metric").is_in(["auc", "logloss"])).select(
        "id", "priority", "left", "right", "metric", "effect", "ci_low", "ci_high",
        "p_centered_two_sided", "holm_joint_primary_auc_logloss", "excludes_zero"), flush=True)


if __name__ == "__main__":
    main()
