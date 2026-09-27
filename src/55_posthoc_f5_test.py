"""F5 post-hoc: validation-fitted calibration, fixed groups, and donor sensitivity.

Run `--stage fit` completely before `--stage apply`. Neither stage loads a CTR
model, runs inference, or uses any test metric to choose anything.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import time

import joblib
import numpy as np
import polars as pl
import sklearn

from ctr_posthoc import calibration, evaluator
from f5_sequences import MAX_LENGTH, ROOT, UNKNOWN_INDEX, sha256, write_json
from f5_test_common import (
    BUILD, F4_BUILD, STUDY, TEST_DATE, TEST_ROWS, TEST_STUDY, VALIDATION_DATE,
)
from project_environment import require_tested_environment

CALIBRATORS = ROOT / "artifacts/calibrators/f5_sequence_test_v1"
TEST_FREEZE = STUDY / "test_freeze_manifest.json"
METRIC_KEYS = ("auc", "logloss", "pr_auc", "gauc", "copc", "ece")


def distinct_predictors() -> list[dict]:
    """Twelve learned runs plus one zero reference; the three zero runs are identical."""
    freeze = json.loads(TEST_FREEZE.read_text())
    records = [r for r in freeze["checkpoints"] if not r["zero_sequence"] or r["seed"] == 42]
    if len(records) != 13:
        raise ValueError(f"expected 13 distinct predictors, found {len(records)}")
    return records


def select_and_refit(ids, labels, probabilities):
    """Validation-only: the existing recipe, reused unchanged."""
    select = calibration.deterministic_selection_mask(ids.astype(np.uint64), 5)
    fit = ~select
    platt = calibration.fit_platt(labels[fit], probabilities[fit])
    isotonic = calibration.fit_isotonic(labels[fit], probabilities[fit])
    candidates = {
        "raw": probabilities[select],
        "platt": calibration.apply_platt(platt, probabilities[select]),
        "isotonic": isotonic.predict(probabilities[select]),
    }
    rows = [
        calibration.metric_row("validation_selection", name, labels[select], p, 20)
        for name, p in candidates.items()
    ]
    method = (
        pl.DataFrame(rows).filter(pl.col("method") != "raw").sort("logloss").row(0, named=True)["method"]
    )
    final = (calibration.fit_platt if method == "platt" else calibration.fit_isotonic)(
        labels, probabilities
    )
    return method, final, rows


def load_split(split: str, run: str) -> pl.DataFrame:
    return evaluator.load_and_validate(split, run, "raw")


def fit_stage() -> None:
    path = TEST_STUDY / "calibration_fit_manifest.json"
    if path.exists() or CALIBRATORS.exists():
        raise FileExistsError("Calibration fitting already started; preserve existing products")
    CALIBRATORS.mkdir(parents=True)
    selected, selections = [], []
    started = time.perf_counter()
    for record in distinct_predictors():
        frame = load_split("val", record["run_name"])
        method, final, rows = select_and_refit(
            frame["impression_id"].to_numpy(), frame["clk"].to_numpy(),
            frame["prediction"].to_numpy(),
        )
        target = CALIBRATORS / f'{record["run_name"]}.joblib'
        joblib.dump(final, target)
        selected.append(
            {
                "run_name": record["run_name"], "condition": record["condition"],
                "seed": record["seed"], "method": method,
                "calibrator_path": str(target.relative_to(ROOT)),
                "calibrator_sha256": sha256(target),
            }
        )
        selections.extend(
            [{**row, "condition": record["condition"], "seed": record["seed"]} for row in rows]
        )
        print(f'Validation only: {record["condition"]} seed {record["seed"]}: {method}', flush=True)
    pl.DataFrame(selections).write_csv(TEST_STUDY / "calibration_selection.csv")
    write_json(
        path,
        {
            "complete": True, "calibrators": selected, "count": len(selected), "zero_count": 1,
            "test_used_for_fitting_or_selection": False,
            "recipe": "validation impression-hash 80/20, Platt vs isotonic by selection LogLoss, refit on full validation",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "numpy": np.__version__, "sklearn": sklearn.__version__,
            "code_sha256": {
                name: sha256(ROOT / "src" / name)
                for name in ("14_calibrate_predictions.py", "55_posthoc_f5_test.py")
            },
        },
    )


def group_frame() -> pl.DataFrame:
    """The seven pre-fixed partitions, all declared before any test metric existed."""
    base = pl.read_parquet(
        ROOT / "data/processed/base/test.parquet",
        columns=["impression_id", "clk", "user_raw", "adgroup_id_raw"],
    ).sort("impression_id")
    context = evaluator.add_cold_start_flags(base).sort("impression_id")

    f5 = pl.read_parquet(
        BUILD / f"test_{TEST_DATE}.parquet",
        columns=["impression_id", "start", "length", "raw_length", "fallback"],
    ).sort("impression_id")
    f4 = pl.read_parquet(
        F4_BUILD / f"test_{TEST_DATE}.parquet", columns=["impression_id", "length"]
    ).sort("impression_id").rename({"length": "f4_length"})
    history = pl.read_parquet(
        BUILD / f"history_{TEST_DATE}.parquet", columns=["user_raw", "recency_bucket"]
    )
    activity = pl.read_parquet(
        ROOT / "data/processed/behavior/window_features/user_behavior_2017-05-13.parquet",
        columns=["user_raw", "event_count_14d"],
    )

    events = pl.read_parquet(BUILD / "events.parquet", columns=["cate_idx", "type_idx"])
    categories = events["cate_idx"].to_numpy()
    types = events["type_idx"].to_numpy()
    del events
    starts = f5["start"].to_numpy()
    lengths = f5["length"].to_numpy()
    non_pv = np.zeros(len(starts), dtype=np.int32)
    unknown = np.zeros(len(starts), dtype=np.int32)
    span = np.arange(MAX_LENGTH)
    for offset in range(0, len(starts), 65536):
        block = lengths[offset:offset + 65536]
        mask = span[None, :] < block[:, None]
        index = np.where(mask, starts[offset:offset + 65536, None] + span[None, :], 0)
        non_pv[offset:offset + 65536] = np.sum(mask & (types[index] != 2), axis=1)
        unknown[offset:offset + 65536] = np.sum(mask & (categories[index] == UNKNOWN_INDEX), axis=1)

    frame = (
        context.join(f5, on="impression_id", validate="1:1")
        .join(f4, on="impression_id", validate="1:1")
        .join(history, on="user_raw", how="left", validate="m:1")
        .join(activity, on="user_raw", how="left", validate="m:1")
        .with_columns(
            pl.Series("non_pv_count", non_pv),
            pl.Series("unknown_tokens", unknown),
            pl.col("event_count_14d").fill_null(0),
        )
        .with_columns(
            pl.col("fallback").alias("bucket_relaxation"),
            (pl.col("fallback") > 0).alias("relaxed"),
            pl.when(pl.col("length") == 0).then(pl.lit(-1.0))
            .otherwise(pl.col("unknown_tokens") / pl.col("length")).alias("unknown_share"),
        )
        .with_columns(
            pl.when(pl.col("raw_length") == 0).then(pl.lit("0"))
            .when(pl.col("raw_length") == 1).then(pl.lit("1"))
            .when(pl.col("raw_length") <= 5).then(pl.lit("2-5"))
            .when(pl.col("raw_length") <= 10).then(pl.lit("6-10"))
            .when(pl.col("raw_length") <= 20).then(pl.lit("11-20"))
            .when(pl.col("raw_length") <= 50).then(pl.lit("21-50"))
            .when(pl.col("raw_length") <= 100).then(pl.lit("51-100"))
            .when(pl.col("raw_length") <= 199).then(pl.lit("101-199"))
            .when(pl.col("raw_length") == 200).then(pl.lit("200"))
            .otherwise(pl.lit(">200")).alias("f5_history_length"),
            pl.when(pl.col("length") == 0).then(pl.lit("empty"))
            .when(pl.col("non_pv_count") == 0).then(pl.lit("0"))
            .when(pl.col("non_pv_count") <= 5).then(pl.lit("1-5"))
            .when(pl.col("non_pv_count") <= 20).then(pl.lit("6-20"))
            .otherwise(pl.lit(">20")).alias("behavior_type_composition"),
            pl.when(pl.col("recency_bucket").is_null()).then(pl.lit("empty"))
            .otherwise(pl.col("recency_bucket").cast(pl.String)).alias("recency"),
            pl.when(pl.col("event_count_14d") == 0).then(pl.lit("0"))
            .when(pl.col("event_count_14d") <= 9).then(pl.lit("1-9"))
            .when(pl.col("event_count_14d") <= 99).then(pl.lit("10-99"))
            .when(pl.col("event_count_14d") <= 999).then(pl.lit("100-999"))
            .otherwise(pl.lit(">=1000")).alias("user_activity"),
            pl.when(pl.col("unknown_share") < 0).then(pl.lit("empty"))
            .when(pl.col("unknown_share") == 0).then(pl.lit("0"))
            .when(pl.col("unknown_share") <= 0.05).then(pl.lit("(0,0.05]"))
            .when(pl.col("unknown_share") <= 0.20).then(pl.lit("(0.05,0.20]"))
            .otherwise(pl.lit(">0.20")).alias("unknown_category_rate"),
            pl.concat_str([
                pl.when(pl.col("f4_length") == 0).then(pl.lit("F4empty")).otherwise(pl.lit("F4nonempty")),
                pl.when(pl.col("length") == 0).then(pl.lit("F5empty")).otherwise(pl.lit("F5nonempty")),
            ], separator="+").alias("f4_by_f5_emptiness"),
        )
    )
    if frame.height != TEST_ROWS:
        raise ValueError(f"group frame has {frame.height} rows")
    return frame


PARTITIONS = (
    "cold_start_segment", "f5_history_length", "behavior_type_composition",
    "recency", "user_activity", "unknown_category_rate", "f4_by_f5_emptiness",
)


def apply_stage() -> None:
    manifest_path = TEST_STUDY / "posthoc_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Post-hoc analysis already complete")
    fit_manifest = json.loads((TEST_STUDY / "calibration_fit_manifest.json").read_text())
    calibrators = {r["run_name"]: r for r in fit_manifest["calibrators"]}
    inference = json.loads((TEST_STUDY / "manifest.json").read_text())
    started = time.perf_counter()

    groups = group_frame()
    groups.select(
        "impression_id", "clk", *PARTITIONS, "bucket_relaxation", "relaxed"
    ).write_parquet(TEST_STUDY.parent.parent / "data/processed/f5_sequence_test_v1_groups.parquet",
                    compression="zstd")

    counts = []
    for partition in PARTITIONS:
        for row in groups.group_by(partition).agg(
            pl.len().alias("rows"), pl.col("clk").mean().alias("ctr")
        ).sort(partition).iter_rows(named=True):
            counts.append({"partition": partition, "value": str(row[partition]),
                           "rows": row["rows"], "ctr": row["ctr"]})
    pl.DataFrame(counts).write_csv(TEST_STUDY / "group_counts.csv")

    calibrated_rows, group_rows, sensitivity_rows = [], [], []
    for record in inference["records"]:
        run = record["run_name"]
        training_run = record["training_run"]
        frame = load_split("test", run).join(
            groups.drop("clk"), on="impression_id", validate="1:1"
        )
        raw = frame["prediction"].to_numpy()

        # Calibration: the method was selected on validation and is applied as-is.
        source = calibrators[training_run if training_run in calibrators else
                            training_run.rsplit("_s", 1)[0] + "_s42_v1"]
        calibrator = joblib.load(ROOT / source["calibrator_path"])
        transformed = (
            calibration.apply_platt(calibrator, raw) if source["method"] == "platt"
            else calibrator.predict(raw)
        )
        for label, values in (("raw", raw), ("calibrated", transformed)):
            scored = frame.with_columns(pl.Series("prediction", values))
            row = evaluator.metric_row("test", label, scored)
            gauc, _ = evaluator.build_gauc("test", scored)
            row["ece"] = evaluator.weighted_ece(evaluator.build_reliability("test", scored, 10))
            row.update({k: gauc[k] for k in ("gauc", "eligible_users", "eligible_row_coverage")})
            calibrated_rows.append(
                {"run_name": run, "condition": record["condition"], "seed": record["seed"],
                 "calibration": label, "method": source["method"], **row}
            )

        # Pre-fixed groups, on raw probabilities.
        for partition in PARTITIONS:
            for value in sorted(groups[partition].unique().to_list()):
                cell = frame.filter(pl.col(partition) == value)
                if cell.height == 0:
                    continue
                row = evaluator.metric_row("test", f"{partition}={value}", cell)
                gauc, _ = evaluator.build_gauc("test", cell)
                row.update({"gauc": gauc["gauc"], "eligible_users": gauc["eligible_users"]})
                group_rows.append(
                    {"run_name": run, "condition": record["condition"], "seed": record["seed"],
                     "partition": partition, "value": str(value), **row}
                )

        # Donor sensitivity: the same metrics with relaxed rows excluded.
        retained = frame.filter(~pl.col("relaxed"))
        row = evaluator.metric_row("test", "bucket_relaxation_excluded", retained)
        gauc, _ = evaluator.build_gauc("test", retained)
        row["gauc"] = gauc["gauc"]
        sensitivity_rows.append(
            {"run_name": run, "condition": record["condition"], "seed": record["seed"],
             "sample": "relaxation_excluded", **row}
        )
        print(f"Post-hoc: {record['condition']} seed {record['seed']}", flush=True)

    pl.DataFrame(calibrated_rows).write_csv(TEST_STUDY / "calibrated_metrics.csv")
    pl.DataFrame(group_rows).write_csv(TEST_STUDY / "group_metrics.csv")
    pl.DataFrame(sensitivity_rows).write_csv(TEST_STUDY / "sensitivity_metrics.csv")

    # Matched minus mismatched, main sample versus relaxation-excluded sample.
    main = pl.read_csv(TEST_STUDY / "test_metrics.csv")
    sensitivity = pl.DataFrame(sensitivity_rows)

    def mean_of(frame: pl.DataFrame, condition: str, key: str) -> float:
        values = frame.filter(pl.col("condition") == condition)[key].to_list()
        return stats.mean(values)

    comparison = []
    for pooling in ("mean", "attention"):
        left = f"behavior_{pooling}_matched"
        right = f"behavior_{pooling}_mismatched"
        for key in ("auc", "logloss", "pr_auc", "gauc"):
            comparison.append(
                {
                    "pooling": pooling, "metric": key,
                    "main_effect": mean_of(main, left, key) - mean_of(main, right, key),
                    "relaxation_excluded_effect": mean_of(sensitivity, left, key) - mean_of(sensitivity, right, key),
                }
            )
    for row in comparison:
        row["absolute_change"] = row["relaxation_excluded_effect"] - row["main_effect"]
        row["sign_preserved"] = bool(
            np.sign(row["main_effect"]) == np.sign(row["relaxation_excluded_effect"])
        )
    pl.DataFrame(comparison).write_csv(TEST_STUDY / "donor_sensitivity.csv")

    write_json(
        manifest_path,
        {
            "complete": True, "split": "test",
            "calibration": {
                "selected_on": "validation only",
                "applied_to": "frozen test predictions",
                "methods": sorted({r["method"] for r in fit_manifest["calibrators"]}),
                "fit_manifest_sha256": sha256(TEST_STUDY / "calibration_fit_manifest.json"),
            },
            "partitions": list(PARTITIONS),
            "partitions_declared_before_test": True,
            "donor_sensitivity": {
                "excluded_rows": int(groups["relaxed"].sum()),
                "retained_rows": int((~groups["relaxed"]).sum()),
                "status": "diagnostic only; it does not replace the main analysis",
                "all_signs_preserved": bool(all(r["sign_preserved"] for r in comparison)),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "environment": require_tested_environment("F5 post-hoc"),
        },
    )
    print(pl.DataFrame(comparison), flush=True)


def main() -> None:
    require_tested_environment("F5 post-hoc")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "apply"], required=True)
    args = parser.parse_args()
    fit_stage() if args.stage == "fit" else apply_stage()


if __name__ == "__main__":
    main()
