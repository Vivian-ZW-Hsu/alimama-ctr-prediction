"""Audit the F5 validation freeze and report validation-only diagnostics.

Mirrors src/24_audit_sequence_freeze.py. Everything here is validation-only and
runs after the immutable selection, so it cannot influence any configuration.

Two checks matter most:

* every empty-history row must reproduce the frozen DeepFM baseline exactly, and
* the zero control must reproduce that baseline on **all** rows, which is what
  makes it a pipeline null rather than a weak model.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date

import numpy as np
import polars as pl

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER
from f5_sequences import MAX_LENGTH, midnight, sha256, source_fingerprints, write_json
from project_environment import require_tested_environment

STUDY = ROOT / "results" / "f5_selection_v1"
BUILD = ROOT / "data/processed/f5_behavior_v1"
VALIDATION_DAY = date(2017, 5, 12)
VAL_ROWS = 3_234_051

# Declared at freeze-audit time, before any F5 test result exists. These are NOT
# preregistered bins; the preregistration fixed the configuration, not this
# diagnostic partition.
LENGTH_BINS = [
    ("0", 0, 0), ("1", 1, 1), ("2-5", 2, 5), ("6-10", 6, 10), ("11-20", 11, 20),
    ("21-50", 21, 50), ("51-100", 51, 100), ("101-199", 101, 199),
    ("200", 200, 200), (">200", 201, 10 ** 9),
]


def length_bin_expression() -> pl.Expr:
    expression = pl.when(pl.col("raw_length") < 0).then(pl.lit("invalid"))
    for label, low, high in LENGTH_BINS:
        expression = expression.when(
            (pl.col("raw_length") >= low) & (pl.col("raw_length") <= high)
        ).then(pl.lit(label))
    return expression.otherwise(pl.lit("unbinned")).alias("history_bin")


def control_balance() -> dict:
    """Describe the confounding the donor rule does not remove. Donors are unchanged."""
    refs = pl.read_parquet(BUILD / f"val_{VALIDATION_DAY}.parquet")
    events = pl.read_parquet(BUILD / "events.parquet", columns=["cate_idx", "time_stamp"])
    categories = events["cate_idx"].to_numpy()
    timestamps = events["time_stamp"].to_numpy()
    del events
    size = int(categories.max()) + 1
    counts = {source: np.zeros(size, dtype=np.int64) for source in ("matched", "mismatched")}
    age_sums = {source: 0 for source in counts}
    cutoff = midnight(VALIDATION_DAY)
    for chunk in refs.iter_slices(65536):
        lengths = chunk["length"].to_numpy()
        mask = np.arange(MAX_LENGTH)[None, :] < lengths[:, None]
        for source, key in (("matched", "start"), ("mismatched", "donor_start")):
            index = (chunk[key].to_numpy()[:, None] + np.arange(MAX_LENGTH)[None, :])[mask]
            counts[source] += np.bincount(categories[index], minlength=size)
            ages = cutoff - timestamps[index]
            if not (ages > 0).all():
                raise ValueError("a retained event is not strictly before the cutoff")
            age_sums[source] += int(ages.sum())
    if counts["matched"].sum() != counts["mismatched"].sum():
        raise ValueError("matched and mismatched token budgets differ")

    history = pl.read_parquet(BUILD / f"history_{VALIDATION_DAY}.parquet")
    exposures = (
        pl.scan_parquet(ROOT / "data/processed/base/train.parquet")
        .group_by("user_raw").agg(pl.len().alias("past_exposures"))
        .collect(engine="streaming")
    )
    lookup = history.select("user_raw", "activity_bucket", "recency_bucket").join(
        exposures, on="user_raw", how="left", validate="1:1"
    ).with_columns(pl.col("past_exposures").fill_null(0))
    nonempty = refs.filter(pl.col("length") > 0)
    joined = nonempty.join(lookup, on="user_raw", validate="m:1").join(
        lookup.rename(
            {
                "user_raw": "donor_user_raw", "activity_bucket": "donor_activity",
                "recency_bucket": "donor_recency", "past_exposures": "donor_past_exposures",
            }
        ),
        on="donor_user_raw", validate="m:1",
    )
    if joined.height != nonempty.height:
        raise ValueError("donor join lost rows")
    activity_mismatch = joined.filter(pl.col("activity_bucket") != pl.col("donor_activity")).height
    recency_mismatch = joined.filter(pl.col("recency_bucket") != pl.col("donor_recency")).height
    distributions = {s: counts[s] / counts[s].sum() for s in counts}
    return {
        "split": "val",
        "computed_after_configuration_selection": True,
        "tokens_per_condition": int(counts["matched"].sum()),
        "impression_weighted_category_total_variation": float(
            np.abs(distributions["matched"] - distributions["mismatched"]).sum() / 2
        ),
        "mean_event_age_days": {
            s: age_sums[s] / int(counts[s].sum()) / 86400 for s in counts
        },
        "activity_bucket_mismatch_rows": activity_mismatch,
        "recency_bucket_mismatch_rows": recency_mismatch,
        "coarsened_rows_by_level": {
            str(level): int((refs["fallback"] == level).sum()) for level in range(4)
        },
        "nonempty_exposure_weighted_past_activity_mean": {
            "matched": float(joined["past_exposures"].mean()),
            "mismatched": float(joined["donor_past_exposures"].mean()),
        },
        "limitation": (
            "Exact retained length is always preserved, and the activity and recency "
            "buckets match except on the counted coarsened rows. Impression-weighted "
            "category popularity, event age and within-bucket activity need not be "
            "identical. This is not a complete causal adjustment."
        ),
    }


def main() -> None:
    environment = require_tested_environment("E5 F5 freeze audit")
    if (STUDY / "freeze_audit.json").exists():
        raise FileExistsError("Freeze audit already exists")
    freeze_path = STUDY / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if not freeze["complete"] or freeze["test_inference_performed"] or freeze["test_authorized"]:
        raise ValueError("freeze manifest is not in the pre-test state")
    for name, digest in (
        ("preregistration.json", freeze["preregistration_sha256"]),
        ("selection.json", freeze["selection_sha256"]),
    ):
        if sha256(STUDY / name) != digest:
            raise ValueError(f"{name} changed after the freeze")
    if sha256(BASE_MODEL) != freeze["base_model_sha256"]:
        raise ValueError("frozen backbone changed")
    if sha256(BASE_SCALER) != freeze["base_scaler_sha256"]:
        raise ValueError("frozen scaler changed")
    for name, digest in freeze["code_sha256"].items():
        if sha256(ROOT / "src" / name) != digest:
            raise ValueError(f"{name} changed after the freeze")
    build = json.loads((ROOT / "results/f5_behavior_v1/build_config.json").read_text())
    if source_fingerprints() != build["inputs"]:
        raise ValueError("F5 build inputs changed since the build")

    spec = importlib.util.spec_from_file_location(
        "shared_eval", ROOT / "src/10_evaluate_predictions.py"
    )
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)

    lengths = pl.read_parquet(
        BUILD / f"val_{VALIDATION_DAY}.parquet", columns=["impression_id", "raw_length"]
    ).with_columns(length_bin_expression())
    if lengths.filter(pl.col("history_bin") == "unbinned").height:
        raise ValueError("a raw length fell outside every declared bin")

    baseline = pl.read_parquet(
        ROOT / "artifacts/predictions/deepfm_f1_f2_f3/val_predictions.parquet"
    ).sort("impression_id")
    empty_ids = lengths.filter(pl.col("raw_length") == 0)["impression_id"]
    baseline_empty = baseline.filter(
        pl.col("impression_id").is_in(empty_ids.implode())
    )["prediction"].to_numpy()

    rows = []
    empty_checks = []
    zero_control_checks = []
    evidence = {}
    conditions = [
        {"run_name": "deepfm_f1_f2_f3", "condition": "frozen_backbone",
         "pooling": "none", "source": "none", "seed": 42, "zero_sequence": False},
        *freeze["checkpoints"],
    ]
    for condition in conditions:
        name = condition["run_name"]
        if name != "deepfm_f1_f2_f3":
            checkpoint = ROOT / "artifacts/models" / name / "best_model.pt"
            prediction_path = ROOT / "artifacts/predictions" / name / "val_predictions.parquet"
            if sha256(checkpoint) != condition["checkpoint_sha256"]:
                raise ValueError(f"{name} checkpoint changed")
            if sha256(prediction_path) != condition["validation_prediction_sha256"]:
                raise ValueError(f"{name} validation prediction changed")
            if (prediction_path.parent / "test_predictions.parquet").exists():
                raise ValueError(f"{name} has a test prediction; test is not authorized")
            prediction = pl.read_parquet(prediction_path).sort("impression_id")
            np.testing.assert_array_equal(
                prediction["impression_id"].to_numpy(), baseline["impression_id"].to_numpy()
            )
            np.testing.assert_array_equal(
                prediction["clk"].to_numpy(), baseline["clk"].to_numpy()
            )
            empty = prediction.filter(
                pl.col("impression_id").is_in(empty_ids.implode())
            )["prediction"].to_numpy()
            np.testing.assert_array_equal(empty, baseline_empty)
            empty_checks.append(
                {"run_name": name, "rows": len(empty), "max_prediction_error": 0.0}
            )
            if condition.get("zero_sequence"):
                # A pipeline null must equal the frozen backbone on every row.
                difference = float(
                    np.max(np.abs(
                        prediction["prediction"].to_numpy() - baseline["prediction"].to_numpy()
                    ))
                )
                if difference != 0.0:
                    raise ValueError(
                        f"{name} is not a pipeline null: max deviation {difference}"
                    )
                zero_control_checks.append(
                    {"run_name": name, "rows": prediction.height,
                     "max_deviation_from_frozen_backbone": difference}
                )
            for path in sorted((ROOT / "results" / name).rglob("*")):
                if path.is_file() and path.suffix in (".json", ".csv"):
                    evidence[str(path.relative_to(ROOT))] = sha256(path)

        frame = shared.load_and_validate("val", name, "raw").join(
            lengths, on="impression_id", validate="1:1"
        )
        if frame.height != VAL_ROWS or frame["history_bin"].null_count():
            raise ValueError(f"{name} validation frame is incomplete")
        for label, _, _ in LENGTH_BINS:
            group = frame.filter(pl.col("history_bin") == label)
            if group.height == 0:
                continue
            row = shared.metric_row("val", label, group)
            gauc, _ = shared.build_gauc("val", group)
            row.update(
                {
                    "run_name": name, "condition": condition.get("condition"),
                    "pooling": condition["pooling"], "source": condition["source"],
                    "seed": condition["seed"], "gauc": gauc["gauc"],
                    "eligible_users": gauc["eligible_users"],
                    "eligible_row_coverage": gauc["eligible_row_coverage"],
                }
            )
            rows.append(row)
        print(f"Passed validation identity/empty-history audit: {name}", flush=True)

    pl.DataFrame(rows).write_csv(STUDY / "validation_history_length.csv")
    write_json(STUDY / "validation_control_balance.json", control_balance())
    for path in sorted(STUDY.glob("*")):
        if path.is_file() and path.suffix in (".json", ".csv"):
            evidence[str(path.relative_to(ROOT))] = sha256(path)
    write_json(
        STUDY / "freeze_audit.json",
        {
            "passed": True,
            "freeze_manifest_sha256": sha256(freeze_path),
            "inputs_unchanged": True,
            "no_new_test_predictions": True,
            "selected_checkpoints_verified": len(freeze["checkpoints"]),
            "empty_history_exact_baseline": empty_checks,
            "zero_control_equals_frozen_backbone": zero_control_checks,
            "length_bins_declared_at": "freeze audit, before any F5 test result; not preregistered",
            "environment": environment,
            "evidence_sha256": evidence,
            "audit_code_sha256": sha256(ROOT / "src/51_audit_f5_freeze.py"),
            "history_diagnostics": "Validation only, after immutable selection; no further tuning",
            "scope": "No test inference, no test scoring, no configuration change",
        },
    )
    print("F5 freeze audit passed.", flush=True)


if __name__ == "__main__":
    main()
