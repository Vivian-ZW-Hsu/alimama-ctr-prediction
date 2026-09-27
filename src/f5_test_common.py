"""Shared constants and helpers for the F5 formal test stage.

Kept out of `f5_sequences.py` on purpose: that module's SHA256 is bound by the
validation freeze manifest, so editing it would invalidate the freeze audit.

Terminology, as declared in results/f5_selection_v1/test_analysis_plan.json: the
physical column named `fallback` in the frozen F5 reference files denotes DONOR
BUCKET RELAXATION LEVEL and nothing else. It is never a retained-length mismatch,
an invalid or missing donor, a self match, a future event or an ownership error;
all of those are verified zero at build time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]

STUDY = ROOT / "results" / "f5_selection_v1"
TEST_STUDY = ROOT / "results" / "f5_sequence_test_v1"
BUILD = ROOT / "data" / "processed" / "f5_behavior_v1"
F4_BUILD = ROOT / "data" / "processed" / "f4_click_daily_v1"

TEST_DATE = "2017-05-13"
VALIDATION_DATE = "2017-05-12"
TEST_ROWS = 3_308_665
VALIDATION_ROWS = 3_234_051

# Canonical name for the stored `fallback` column.
BUCKET_RELAXATION_COLUMN = "fallback"
BUCKET_RELAXATION_NAME = "bucket_relaxation"
BUCKET_RELAXATION_LEVELS = {
    0: "exact retained length, same activity bucket, same recency bucket",
    1: "exact retained length and activity bucket; recency bucket relaxed",
    2: "exact retained length only; activity and recency buckets relaxed",
    3: "donor retained length at least the target's; never used, zero rows",
}

METRICS = ("auc", "logloss", "pr_auc", "gauc")


def bucket_relaxation_markers(date: str = TEST_DATE, split: str = "test") -> pl.DataFrame:
    """Per-impression relaxation level, read from the frozen reference file.

    The frozen file is opened read-only; the marker table is a derived artifact.
    """
    refs = pl.read_parquet(
        BUILD / f"{split}_{date}.parquet",
        columns=["impression_id", BUCKET_RELAXATION_COLUMN, "length", "raw_length"],
    ).sort("impression_id")
    return refs.rename({BUCKET_RELAXATION_COLUMN: BUCKET_RELAXATION_NAME}).with_columns(
        (pl.col(BUCKET_RELAXATION_NAME) > 0).alias("relaxed")
    )


def relaxed_rows_only(markers: pl.DataFrame) -> pl.DataFrame:
    """The committed marker artifact: only the rows that needed relaxation.

    A full per-impression level table is 15 MiB and would duplicate information
    that already exists: the complete mapping IS the `fallback` column of the
    frozen reference file, reproducible in one line by
    `bucket_relaxation_markers()`. Only 1,457 of 3,308,665 test rows are relaxed,
    so the exclusion list is what the sensitivity analysis actually needs.
    """
    return markers.filter(pl.col("relaxed")).select(
        "impression_id", BUCKET_RELAXATION_NAME, "length", "raw_length"
    )


def relaxation_summary(markers: pl.DataFrame) -> dict:
    counts = (
        markers.group_by(BUCKET_RELAXATION_NAME)
        .agg(pl.len().alias("rows"))
        .sort(BUCKET_RELAXATION_NAME)
    )
    total = markers.height
    relaxed = int(markers["relaxed"].sum())
    return {
        "rows": total,
        "levels": {
            str(row[BUCKET_RELAXATION_NAME]): row["rows"] for row in counts.iter_rows(named=True)
        },
        "level_meanings": {str(k): v for k, v in BUCKET_RELAXATION_LEVELS.items()},
        "relaxed_rows": relaxed,
        "relaxed_share": relaxed / total,
        "retained_rows": total - relaxed,
        "all_relaxed_rows_have_nonempty_history": bool(
            markers.filter(pl.col("relaxed"))["length"].min() is None
            or markers.filter(pl.col("relaxed"))["length"].min() > 0
        ),
        "terminology": (
            "bucket_relaxation is the stored `fallback` column; it denotes donor "
            "bucket relaxation only, never a length mismatch or an invalid donor"
        ),
    }


def paired_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Left minus right, with no metric-orientation flipping."""
    return float(left - right)
