"""Stage E1: full-population cross-source namespace and behavior-window audit.

Purpose.  The existing F4 study uses past clicked-ad categories because an
earlier design pass could only check a non-random 200,000-row prefix of
behavior_log.csv and therefore recorded the cross-source namespace contract as
unverified.  This stage settles that question on the full population, decides
whether an F5 behavior-sequence budget is admissible, and measures how much
ordered history a completed-day cutoff actually makes available.

Inputs are the raw CSV tables plus the verified lossless Parquet copy of
behavior_log written by src/40_convert_behavior_log.py.  Nothing here reads
model inputs, checkpoints, predictions or any frozen result, and nothing is
overwritten outside this stage's own result directory.

Pre-declared admissibility thresholds are fixed in THRESHOLDS below, before the
coverage numbers are known to this script, so the verdict cannot be tuned to the
observed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from datetime import date
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BEHAVIOR_CSV = RAW_DIR / "behavior_log.csv"
BEHAVIOR_PARQUET = ROOT / "data" / "processed" / "behavior_raw_v1" / "behavior_log.parquet"
AD_CSV = RAW_DIR / "ad_feature.csv"
SAMPLE_CSV = RAW_DIR / "raw_sample.csv"
BASE_DIR = ROOT / "data" / "processed" / "base"
RESULT_DIR = ROOT / "results" / "behavior_namespace_audit_v1"

BEIJING = "Asia/Shanghai"

# Declared source schemas.  brand is integer-valued with empty cells for missing
# values; an earlier derived Parquet inferred Float64 for it, which is the one
# real dtype conflict between the raw table and that historical artifact.
BEHAVIOR_SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "btag": pl.String,
    "cate": pl.Int64,
    "brand": pl.Int64,
}
AD_SCHEMA = {
    "adgroup_id": pl.Int64,
    "cate_id": pl.Int64,
    "campaign_id": pl.Int64,
    "customer": pl.Int64,
    "brand": pl.Int64,
    "price": pl.Float64,
}
SAMPLE_SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "adgroup_id": pl.Int64,
    "pid": pl.String,
    "nonclk": pl.Int8,
    "clk": pl.Int8,
}

# The formal F5 window.  Earlier dates carry only sporadic records and are
# excluded from the formal experiment; the audit still reports them.
FORMAL_WINDOW_START = date(2017, 4, 22)
FORMAL_WINDOW_END_EXCLUSIVE = date(2017, 5, 14)

# Target impression dates, in Beijing time.
TARGET_DATES = [date(2017, 5, day) for day in range(6, 14)]

# Primary sequence length, fixed from the published Alimama protocol rather than
# from any metric computed in this project.
PRIMARY_SEQUENCE_LENGTH = 200

THRESHOLDS = {
    "ad_category_forward_coverage_min": 0.95,
    "impression_user_forward_coverage_min": 0.95,
    "ad_brand_forward_coverage_min": 0.90,
    "note": (
        "Category and user coverage are hard preconditions for F5. Brand is a "
        "secondary token field: below its threshold F5 proceeds category-only."
    ),
}

# ad_feature.csv encodes a missing brand as the literal four-character token
# "NULL" rather than an empty field; behavior_log.csv and raw_sample.csv contain
# no missing cells at all. Declaring this explicitly keeps the three tables on
# identical integer semantics for the coverage comparison.
NULL_VALUES = ["NULL"]
AD_BRAND_NULL_TOKEN_ROWS = 246330

HASH_BLOCK_BYTES = 8 * 1024 * 1024

# Sporadic pre-window dates are listed in full in daily_events.csv; the JSON keeps
# only the final ramp so the committed evidence file stays small.
PRE_WINDOW_RAMP_DATES = 14


def verdict_for(namespaces: dict[str, dict[str, object]]) -> dict[str, object]:
    """Apply the pre-declared thresholds in THRESHOLDS to measured coverage.

    Category and user coverage are hard preconditions for F5. Brand is a
    secondary token field: failing it downgrades F5 to category-only tokens
    instead of blocking the budget.
    """
    category_ok = (
        namespaces["category"]["forward_coverage"]
        >= THRESHOLDS["ad_category_forward_coverage_min"]
    )
    user_ok = (
        namespaces["user"]["forward_coverage"]
        >= THRESHOLDS["impression_user_forward_coverage_min"]
    )
    brand_ok = (
        namespaces["brand"]["forward_coverage"]
        >= THRESHOLDS["ad_brand_forward_coverage_min"]
    )
    return {
        "category_precondition_met": bool(category_ok),
        "user_precondition_met": bool(user_ok),
        "brand_token_admissible": bool(brand_ok),
        "f5_admissible": bool(category_ok and user_ok),
        "brand_policy": (
            "brand tokens admissible" if brand_ok
            else "F5 proceeds category-only; brand coverage below threshold"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": stat.st_size,
        "mtime_epoch": int(stat.st_mtime),
        "sha256": sha256_file(path) if include_hash else None,
    }


def beijing_date(column: str = "time_stamp") -> pl.Expr:
    """Epoch seconds -> Beijing calendar date."""
    return (
        pl.from_epoch(pl.col(column), time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(BEIJING)
        .dt.date()
    )


def behavior() -> pl.LazyFrame:
    return pl.scan_parquet(BEHAVIOR_PARQUET)


def distinct_ids(frame: pl.LazyFrame, column: str) -> set[int]:
    series = (
        frame.select(pl.col(column))
        .drop_nulls()
        .unique()
        .collect(engine="streaming")
        .to_series()
    )
    return set(series.to_list())


def coverage_record(
    left_name: str,
    left: set[int],
    right_name: str,
    right: set[int],
    sample_size: int,
) -> dict[str, object]:
    shared = left & right
    left_only = sorted(left - right)
    right_only = sorted(right - left)
    return {
        "left": left_name,
        "right": right_name,
        "left_distinct": len(left),
        "right_distinct": len(right),
        "shared_distinct": len(shared),
        "forward_coverage": len(shared) / len(left) if left else None,
        "reverse_coverage": len(shared) / len(right) if right else None,
        "left_min": min(left) if left else None,
        "left_max": max(left) if left else None,
        "right_min": min(right) if right else None,
        "right_max": max(right) if right else None,
        "left_only_count": len(left_only),
        "right_only_count": len(right_only),
        "left_only_sample": left_only[:sample_size],
        "right_only_sample": right_only[:sample_size],
    }


# --------------------------------------------------------------------------
# 1-3. Namespace coverage
# --------------------------------------------------------------------------
def audit_namespaces(sample_size: int) -> dict[str, object]:
    ads = pl.scan_csv(AD_CSV, schema=AD_SCHEMA, null_values=NULL_VALUES)
    impressions = pl.scan_csv(SAMPLE_CSV, schema=SAMPLE_SCHEMA, null_values=NULL_VALUES)
    events = behavior()

    ad_categories = distinct_ids(ads, "cate_id")
    ad_brands = distinct_ids(ads, "brand")
    impression_users = distinct_ids(impressions, "user")
    behavior_categories = distinct_ids(events, "cate")
    behavior_brands = distinct_ids(events, "brand")
    behavior_users = distinct_ids(events, "user")

    return {
        "category": coverage_record(
            "ad_feature.cate_id", ad_categories,
            "behavior_log.cate", behavior_categories, sample_size,
        ),
        "brand": coverage_record(
            "ad_feature.brand", ad_brands,
            "behavior_log.brand", behavior_brands, sample_size,
        ),
        "user": coverage_record(
            "raw_sample.user", impression_users,
            "behavior_log.user", behavior_users, sample_size,
        ),
    }


# --------------------------------------------------------------------------
# 4. Time distribution across every date present in the source
# --------------------------------------------------------------------------
def audit_time_distribution() -> tuple[dict[str, object], pl.DataFrame]:
    events = behavior()
    daily = (
        events.group_by(beijing_date().alias("event_date_bj"))
        .agg(
            pl.len().alias("events"),
            pl.col("user").n_unique().alias("users"),
        )
        .collect(engine="streaming")
        .sort("event_date_bj")
    )
    bounds = (
        events.select(
            pl.len().alias("rows"),
            pl.col("time_stamp").min().alias("min_epoch"),
            pl.col("time_stamp").max().alias("max_epoch"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    in_window = daily.filter(
        (pl.col("event_date_bj") >= FORMAL_WINDOW_START)
        & (pl.col("event_date_bj") < FORMAL_WINDOW_END_EXCLUSIVE)
    )
    before_window = daily.filter(pl.col("event_date_bj") < FORMAL_WINDOW_START)
    after_window = daily.filter(pl.col("event_date_bj") >= FORMAL_WINDOW_END_EXCLUSIVE)

    summary = {
        "rows": int(bounds["rows"]),
        "min_epoch": int(bounds["min_epoch"]),
        "max_epoch": int(bounds["max_epoch"]),
        "distinct_dates": daily.height,
        "formal_window": {
            "start": str(FORMAL_WINDOW_START),
            "end_exclusive": str(FORMAL_WINDOW_END_EXCLUSIVE),
            "dates": in_window.height,
            "events": int(in_window["events"].sum()),
        },
        "before_formal_window": {
            "dates": before_window.height,
            "events": int(before_window["events"].sum()) if before_window.height else 0,
            "max_events_on_a_single_date": (
                int(before_window["events"].max()) if before_window.height else 0
            ),
            "note": (
                "Full per-date counts for every date are in daily_events.csv; "
                f"only the final {PRE_WINDOW_RAMP_DATES} dates before the formal "
                "window are inlined here."
            ),
            "ramp_before_window": [
                {"date": str(row["event_date_bj"]), "events": int(row["events"])}
                for row in before_window.tail(PRE_WINDOW_RAMP_DATES).iter_rows(named=True)
            ],
        },
        "after_formal_window": {
            "dates": after_window.height,
            "events": int(after_window["events"].sum()) if after_window.height else 0,
            "note": "Full per-date counts are in daily_events.csv.",
            "first_dates_after_window": [
                {"date": str(row["event_date_bj"]), "events": int(row["events"])}
                for row in after_window.head(PRE_WINDOW_RAMP_DATES).iter_rows(named=True)
            ],
        },
        "decision": (
            "Formal F5 uses only completed Beijing dates in "
            f"[{FORMAL_WINDOW_START}, {FORMAL_WINDOW_END_EXCLUSIVE}). Records "
            "outside this window are sporadic or out-of-range and are excluded."
        ),
    }
    return summary, daily


# --------------------------------------------------------------------------
# 5. Behavior types and per-user event volume
# --------------------------------------------------------------------------
def audit_behavior_types() -> tuple[dict[str, object], pl.DataFrame, pl.DataFrame]:
    events = behavior().filter(
        (beijing_date() >= FORMAL_WINDOW_START)
        & (beijing_date() < FORMAL_WINDOW_END_EXCLUSIVE)
    )
    by_type = (
        events.group_by("btag")
        .agg(
            pl.len().alias("events"),
            pl.col("user").n_unique().alias("users"),
            beijing_date().min().alias("first_date"),
            beijing_date().max().alias("last_date"),
        )
        .collect(engine="streaming")
        .sort("events", descending=True)
    )
    per_user = (
        events.group_by("user")
        .agg(pl.len().alias("events"))
        .collect(engine="streaming")
    )
    quantiles = {
        "users": per_user.height,
        "mean": float(per_user["events"].mean()),
        "median": float(per_user["events"].median()),
        "p90": float(per_user["events"].quantile(0.90)),
        "p95": float(per_user["events"].quantile(0.95)),
        "p99": float(per_user["events"].quantile(0.99)),
        "max": int(per_user["events"].max()),
    }
    quantile_frame = pl.DataFrame([{"statistic": k, "value": v} for k, v in quantiles.items()])
    return {"per_user_events": quantiles}, by_type, quantile_frame


# --------------------------------------------------------------------------
# 6. Available ordered history under a completed-day cutoff, impression-weighted
# --------------------------------------------------------------------------
def audit_available_history() -> tuple[dict[str, object], pl.DataFrame]:
    """How many ordered behavior tokens each target impression could actually use.

    This is the number that decides whether an F5 budget is worth building: it is
    directly comparable to the existing F4 statistics (mean 4.69 retained tokens,
    30.2% empty on the test day).
    """
    daily_counts = (
        behavior()
        .filter(
            (beijing_date() >= FORMAL_WINDOW_START)
            & (beijing_date() < FORMAL_WINDOW_END_EXCLUSIVE)
        )
        .group_by(["user", beijing_date().alias("event_date_bj")])
        .agg(pl.len().alias("events"))
        .collect(engine="streaming")
    )

    rows: list[dict[str, object]] = []
    for target in TARGET_DATES:
        split = "train" if target <= date(2017, 5, 11) else (
            "val" if target == date(2017, 5, 12) else "test"
        )
        impressions = (
            pl.scan_parquet(BASE_DIR / f"{split}.parquet")
            .filter(pl.col("event_date_bj") == target)
            .select("user_raw")
            .collect(engine="streaming")
        )
        available = (
            daily_counts.filter(pl.col("event_date_bj") < target)
            .group_by("user")
            .agg(pl.col("events").sum().alias("available_events"))
        )
        joined = impressions.join(
            available, left_on="user_raw", right_on="user", how="left"
        ).with_columns(pl.col("available_events").fill_null(0))
        volume = joined["available_events"]
        retained = volume.clip(upper_bound=PRIMARY_SEQUENCE_LENGTH)
        rows.append(
            {
                "target_date": str(target),
                "split": split,
                "impressions": joined.height,
                "empty_history_rows": int((volume == 0).sum()),
                "empty_history_share": float((volume == 0).mean()),
                "mean_available": float(volume.mean()),
                "median_available": float(volume.median()),
                "p90_available": float(volume.quantile(0.90)),
                "mean_retained_at_cap": float(retained.mean()),
                "rows_at_cap": int((volume >= PRIMARY_SEQUENCE_LENGTH).sum()),
                "share_at_cap": float((volume >= PRIMARY_SEQUENCE_LENGTH).mean()),
            }
        )
    frame = pl.DataFrame(rows)
    summary = {
        "primary_sequence_length": PRIMARY_SEQUENCE_LENGTH,
        "cutoff_rule": "completed Beijing dates strictly before the target date",
        "f4_reference": {
            "source": "past clicked-ad categories, retained length 20",
            "test_mean_retained_tokens": 4.687,
            "test_empty_share": 0.3021,
            "note": "Recomputed from results/f4_click_daily_v1/audit_2017-05-13.json",
        },
        "per_target_date": rows,
    }
    return summary, frame


# --------------------------------------------------------------------------
# 7. Integrity: nulls, timestamp legality, row-level CSV/Parquet recheck
# --------------------------------------------------------------------------
def audit_integrity(recheck_size: int, skip_csv_recheck: bool) -> dict[str, object]:
    events = behavior()
    nulls = (
        events.select(
            pl.len().alias("rows"),
            *[pl.col(c).null_count().alias(f"{c}_nulls") for c in BEHAVIOR_SCHEMA],
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    legality = (
        events.select(
            (pl.col("time_stamp") < 0).sum().alias("negative_timestamps"),
            (pl.col("cate") <= 0).sum().alias("non_positive_categories"),
            (pl.col("brand") <= 0).sum().alias("non_positive_brands"),
            (~pl.col("btag").is_in(["pv", "cart", "fav", "buy"])).sum().alias("unknown_btag"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )

    recheck: dict[str, object] = {"performed": False}
    if not skip_csv_recheck:
        total = int(nulls["rows"])
        step = max(total // recheck_size, 1)
        indices = list(range(0, total, step))[:recheck_size]
        wanted = pl.Series("row_index", indices, dtype=pl.UInt32)
        from_parquet = (
            events.with_row_index("row_index")
            .filter(pl.col("row_index").is_in(wanted.implode()))
            .collect(engine="streaming")
            .sort("row_index")
        )
        from_csv = (
            pl.scan_csv(BEHAVIOR_CSV, schema=BEHAVIOR_SCHEMA, null_values=NULL_VALUES)
            .with_row_index("row_index")
            .filter(pl.col("row_index").is_in(wanted.implode()))
            .collect(engine="streaming")
            .sort("row_index")
        )
        if from_csv.height != len(indices) or from_parquet.height != len(indices):
            raise ValueError(
                "row-level recheck selected "
                f"{from_csv.height}/{from_parquet.height} of {len(indices)} "
                "requested rows; an empty selection would make the equality "
                "check vacuous"
            )
        recheck = {
            "performed": True,
            "requested_rows": len(indices),
            "sampled_rows": from_csv.height,
            "index_stride": step,
            "selection": "deterministic evenly spaced row indices, not a prefix",
            "exact_match": from_csv.equals(from_parquet),
        }
        if not recheck["exact_match"]:
            raise ValueError("row-level CSV/Parquet recheck mismatch")

    return {
        "null_counts": {k: int(v) for k, v in nulls.items()},
        "value_legality": {k: int(v) for k, v in legality.items()},
        "row_level_recheck": recheck,
        "missing_value_encoding": {
            "ad_feature.brand": (
                f"literal 'NULL' token on {AD_BRAND_NULL_TOKEN_ROWS:,} rows, not an "
                "empty field; an explicit null_values list is required or the "
                "column cannot be parsed as Int64"
            ),
            "behavior_log": "no missing cells in any column",
            "raw_sample": "no missing cells in any column",
        },
        "dtype_conflict": (
            "behavior_log.brand and ad_feature.brand are both integer-valued. A "
            "historical derived Parquet inferred Float64 for ad_feature.brand "
            "because of its NULL token; this audit reads both with an explicit "
            "Int64 schema plus null_values so the comparison is on identical "
            "integer semantics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmatched-sample-size", type=int, default=25)
    parser.add_argument("--recheck-size", type=int, default=500)
    parser.add_argument("--skip-csv-recheck", action="store_true",
                        help="skip the extra streaming pass over the 22 GiB CSV")
    parser.add_argument("--skip-source-hashes", action="store_true",
                        help="skip SHA256 of the raw CSV inputs")
    args = parser.parse_args()

    if not BEHAVIOR_PARQUET.exists():
        raise FileNotFoundError(
            f"missing {BEHAVIOR_PARQUET}; run src/40_convert_behavior_log.py first"
        )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Auditing cross-source namespaces...")
    namespaces = audit_namespaces(args.unmatched_sample_size)

    print("Auditing behavior time distribution...")
    time_summary, daily_frame = audit_time_distribution()

    print("Auditing behavior types and per-user volume...")
    type_summary, type_frame, quantile_frame = audit_behavior_types()

    print("Auditing available ordered history under completed-day cutoff...")
    history_summary, history_frame = audit_available_history()

    print("Auditing integrity...")
    integrity = audit_integrity(args.recheck_size, args.skip_csv_recheck)

    verdict = verdict_for(namespaces)

    conversion_manifest_path = RESULT_DIR / "behavior_log_conversion.json"
    audit = {
        "stage": "E1 cross-source namespace and behavior-window audit",
        "thresholds": THRESHOLDS,
        "verdict": verdict,
        "sources": {
            "behavior_log_csv": file_record(BEHAVIOR_CSV, include_hash=False),
            "behavior_log_parquet_conversion_manifest": (
                str(conversion_manifest_path.relative_to(ROOT))
                if conversion_manifest_path.exists() else None
            ),
            "ad_feature_csv": file_record(AD_CSV, not args.skip_source_hashes),
            "raw_sample_csv": file_record(SAMPLE_CSV, not args.skip_source_hashes),
        },
        "namespaces": namespaces,
        "time_distribution": time_summary,
        "behavior_types": type_summary,
        "available_history": history_summary,
        "integrity": integrity,
        "implementation_sha256": sha256_file(Path(__file__)),
        "polars_version": pl.__version__,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (RESULT_DIR / "namespace_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n"
    )
    daily_frame.write_csv(RESULT_DIR / "daily_events.csv")
    type_frame.write_csv(RESULT_DIR / "behavior_types.csv")
    quantile_frame.write_csv(RESULT_DIR / "per_user_event_quantiles.csv")
    history_frame.write_csv(RESULT_DIR / "available_history.csv")
    pl.DataFrame(
        [
            {
                "relation": key,
                "left": value["left"],
                "right": value["right"],
                "left_distinct": value["left_distinct"],
                "right_distinct": value["right_distinct"],
                "shared_distinct": value["shared_distinct"],
                "forward_coverage": value["forward_coverage"],
                "reverse_coverage": value["reverse_coverage"],
            }
            for key, value in namespaces.items()
        ]
    ).write_csv(RESULT_DIR / "namespace_coverage.csv")

    print("\n=== NAMESPACE COVERAGE ===")
    for key, value in namespaces.items():
        print(
            f"  {key:9s} {value['left']:22s} -> {value['right']:22s} "
            f"forward {value['forward_coverage']:.4%}  reverse {value['reverse_coverage']:.4%}"
        )
    print("\n=== AVAILABLE HISTORY (completed-day cutoff) ===")
    print(history_frame.select(
        "target_date", "split", "impressions", "empty_history_share",
        "mean_available", "mean_retained_at_cap", "share_at_cap",
    ))
    print("\n=== VERDICT ===")
    for key, value in verdict.items():
        print(f"  {key}: {value}")
    print(f"\nelapsed {audit['elapsed_seconds']:.1f} s -> {RESULT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
