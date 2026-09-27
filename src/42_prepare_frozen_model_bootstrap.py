"""Stage E3 preparation: pre-declare and bind the frozen tabular bootstrap.

This stage reuses the four already frozen F123 tabular test predictions. It does
not retrain anything, does not run model inference, and does not touch any
existing result directory. It writes the pre-declared comparison protocol and the
exact rank/loss/GAUC inputs consumed by src/cluster_bootstrap_metrics.cpp.

The protocol is written here and committed BEFORE src/43_bootstrap_frozen_models.py
is allowed to run, so the primary hypothesis set is fixed before any interval
exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from project_environment import require_tested_environment

from ctr_posthoc import (
    ROOT,
    evaluator,
    sha256,
    write_common,
    write_json,
    write_rank_model,
)

STUDY = ROOT / "results" / "frozen_model_bootstrap_v1"
DATA = ROOT / "data" / "processed" / "frozen_model_bootstrap_v1"
SPLIT = "test"

# The four frozen single-seed F123 tabular families, in their existing run names.
MODELS = [
    {"key": "lr_f123", "family": "Logistic Regression", "run_name": "lr_f1_f2_f3"},
    {"key": "lightgbm_f123", "family": "LightGBM", "run_name": "lightgbm_f1_f2_f3"},
    {"key": "wide_deep_f123", "family": "Wide & Deep", "run_name": "wide_deep_f1_f2_f3"},
    {"key": "deepfm_f123", "family": "DeepFM", "run_name": "deepfm_f1_f2_f3"},
]

METRICS = ("auc", "logloss", "pr_auc", "gauc")

# Pre-declared comparisons. Only the primary pair carries a confirmatory claim and
# Holm correction; everything else is explicitly exploratory and is reported
# without a family-wise adjustment.
PRIMARY = {
    "id": 1,
    "left": "lr_f123",
    "right": "deepfm_f123",
    "metrics": ["auc", "logloss"],
    "interpretation": "confirmatory_primary",
}
SECONDARY = [
    {
        "id": 2,
        "left": "lr_f123",
        "right": "lightgbm_f123",
        "metrics": list(METRICS),
        "interpretation": "exploratory_secondary",
    },
    {
        "id": 3,
        "left": "lr_f123",
        "right": "wide_deep_f123",
        "metrics": list(METRICS),
        "interpretation": "exploratory_secondary",
    },
]

BOOTSTRAP = {
    "replicates": 1000,
    "pilot_replicates": 5,
    "seed": 20260926,
    "threads": 8,
    "resample_unit": "user_raw cluster, exact multinomial over all test users",
    "interval": "marginal percentile 95%, not simultaneous",
    "p_value": "centered two-sided, resolution 1/(B+1)",
    "holm_scope": "the two primary metrics of comparison 1 only",
    "conditionality": (
        "Conditional on the four frozen fitted models and the single frozen test "
        "day. This estimates user-sampling uncertainty only. It is not an estimate "
        "of training-seed uncertainty: each tabular family has exactly one seed, "
        "so re-training variability is not represented at all."
    ),
}


def main() -> None:
    environment = require_tested_environment("E3 bootstrap preparation")
    if DATA.exists() or (STUDY / "input_manifest.json").exists():
        raise FileExistsError(
            "Stage E3 inputs are immutable; inspect the existing directory instead "
            "of regenerating it"
        )
    STUDY.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True)
    started = time.perf_counter()

    fingerprints: dict[str, str] = {}

    base_path = ROOT / "data" / "processed" / "base" / f"{SPLIT}.parquet"
    base = pl.read_parquet(
        base_path,
        columns=["impression_id", "clk", "user_raw", "adgroup_id_raw"],
    ).sort("impression_id")
    fingerprints[str(base_path.relative_to(ROOT))] = sha256(base_path)

    context = evaluator.add_cold_start_flags(base).sort("impression_id")
    context.write_parquet(DATA / f"{SPLIT}_groups.parquet", compression="zstd")

    user_values, user = np.unique(context["user_raw"].to_numpy(), return_inverse=True)
    user = user.astype(np.uint32)
    n_users = len(user_values)
    n_rows = context.height
    labels = context["clk"].to_numpy().astype(np.uint8)

    counts = np.bincount(user, minlength=n_users)
    clicks = np.bincount(user, weights=labels, minlength=n_users)
    eligible = (clicks > 0) & (clicks < counts)
    eligible_counts = counts * eligible

    write_common(DATA / "common.bin", n_rows, n_users, len(MODELS), counts, eligible_counts)

    records = []
    for index, model in enumerate(MODELS):
        path = ROOT / "artifacts" / "predictions" / model["run_name"] / f"{SPLIT}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"missing frozen prediction {path}; refusing to synthesise a "
                "bootstrap from metric summaries"
            )
        digest = sha256(path)
        fingerprints[str(path.relative_to(ROOT))] = digest

        prediction = pl.read_parquet(path).sort("impression_id")
        if prediction.height != n_rows:
            raise ValueError(f"{path} has {prediction.height} rows, expected {n_rows}")
        for column in ("impression_id", "clk"):
            np.testing.assert_array_equal(
                prediction[column].to_numpy(), context[column].to_numpy()
            )
        probabilities = prediction["prediction"].to_numpy().astype(np.float64)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{path} contains non-finite probabilities")
        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError(f"{path} contains probabilities outside [0, 1]")

        scored = context.with_columns(pl.Series("prediction", probabilities))
        gauc, users = evaluator.build_gauc(SPLIT, scored)
        positions = np.searchsorted(user_values, users["user_raw"].to_numpy())
        numerator = np.zeros(n_users, dtype=float)
        numerator[positions] = users["rows"].to_numpy() * users["user_auc"].to_numpy()

        write_rank_model(
            DATA / f"model_{index}.bin", user, labels, probabilities, n_users, numerator
        )
        records.append(
            {
                **model,
                "index": index,
                "prediction_path": str(path.relative_to(ROOT)),
                "prediction_sha256": digest,
                "saved_gauc": gauc["gauc"],
                "eligible_users": gauc["eligible_users"],
                "eligible_rows": gauc["eligible_rows"],
            }
        )
        print(f"Prepared exact rank input {index + 1}/{len(MODELS)}: {model['key']}", flush=True)

    write_json(
        STUDY / "protocol.json",
        {
            "stage": "E3 frozen tabular paired user-cluster bootstrap",
            "split": SPLIT,
            "models": [{k: m[k] for k in ("key", "family", "run_name")} for m in MODELS],
            "metrics": list(METRICS),
            "comparisons": [PRIMARY, *SECONDARY],
            "bootstrap": BOOTSTRAP,
            "reuses": "saved frozen test predictions only; no retraining, no inference",
            "declared_before_results": True,
        },
    )
    write_json(
        STUDY / "input_manifest.json",
        {
            "complete": True,
            "rows": n_rows,
            "users": n_users,
            "eligible_users": int(eligible.sum()),
            "eligible_rows": int(eligible_counts.sum()),
            "models": records,
            "input_sha256": fingerprints,
            "derived_sha256": {
                str(p.relative_to(ROOT)): sha256(p) for p in sorted(DATA.iterdir())
            },
            "code_sha256": {
                name: sha256(ROOT / "src" / name)
                for name in ("ctr_posthoc.py", "42_prepare_frozen_model_bootstrap.py")
            },
            "environment": environment,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        },
    )
    print(f"\nrows {n_rows:,} | users {n_users:,} | eligible users {int(eligible.sum()):,}")
    print(f"wrote {STUDY.relative_to(ROOT)}/protocol.json and input_manifest.json")


if __name__ == "__main__":
    main()
