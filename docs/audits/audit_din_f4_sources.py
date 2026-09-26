"""Stage-A read-only F4 source audit; JSON goes to stdout, inputs are never written.

Run from the repository root with the existing ctr_project Python environment.
Reads full train/validation base columns and existing behavior aggregates;
reads only a prefix of raw behavior CSV, and only test Parquet metadata.
No training, prediction, feature materialization, or project module imports.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path: Path) -> dict:
    stat = path.stat()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        handle.seek(max(0, stat.st_size - 1024 * 1024))
        tail = handle.read(1024 * 1024)
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "head_tail_sha256": hashlib.sha256(head + tail).hexdigest(),
        "fingerprint_scope": "first and last 1 MiB; NOT a full-file content hash",
    }


def main() -> None:
    started = time.perf_counter()
    report = {
        "audit_version": 2,
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": {
            "train_validation": "complete narrow-column scans",
            "behavior_raw": "first 200000 rows, non-random; not a population estimate",
            "behavior_aggregate": "prior dates through 2017-05-11 only",
            "test": "Parquet footer/schema/row count and stat only; no labels or predictions read",
            "model_execution": False,
        },
        "environment": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("polars", "pyarrow", "numpy", "torch")
            },
        },
        "raw_files": {},
        "parquet_inputs": {},
    }
    for name in ("raw_sample", "ad_feature", "user_profile", "behavior_log"):
        path = ROOT / "data/raw" / f"{name}.csv"
        with path.open() as handle:
            header = next(csv.reader(handle))
        report["raw_files"][name] = {"header": header, **fingerprint(path)}

    for family in ("base", "features", "model_input", "past_ctr", "model_input_f3"):
        for split in ("train", "val", "test"):
            path = ROOT / "data/processed" / family / f"{split}.parquet"
            parquet = pq.ParquetFile(path)
            report["parquet_inputs"][str(path.relative_to(ROOT))] = {
                "rows": parquet.metadata.num_rows,
                "schema": {f.name: str(f.type) for f in parquet.schema_arrow},
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }

    columns = ["impression_id", "user_raw", "event_date_bj", "time_stamp",
               "adgroup_id_raw", "cate_id_raw", "brand_raw", "clk"]
    frames = [
        pl.scan_parquet(ROOT / f"data/processed/base/{split}.parquet")
        .select(*columns, pl.lit(split).alias("dataset"))
        for split in ("train", "val")
    ]
    impressions = pl.concat(frames).collect()
    assert impressions.height == 23_249_296
    assert impressions["impression_id"].n_unique() == impressions.height
    assert impressions["clk"].is_in([0, 1]).all()
    assert impressions.select(
        (pl.from_epoch("time_stamp", time_unit="s").dt.replace_time_zone("UTC")
         .dt.convert_time_zone("Asia/Shanghai").dt.date() != pl.col("event_date_bj")).sum()
    ).item() == 0
    clicks = impressions.filter(pl.col("clk") == 1)
    assert clicks.height == 1_199_566
    report["click_event_summary"] = clicks.group_by("dataset").agg(
        pl.len().alias("click_events"), pl.col("user_raw").n_unique().alias("users"),
        pl.col("adgroup_id_raw").n_unique().alias("ads"),
        pl.col("cate_id_raw").n_unique().alias("categories"),
        pl.col("cate_id_raw").null_count().alias("missing_category"),
        pl.col("brand_raw").is_null().mean().alias("brand_missing_rate"),
        pl.col("time_stamp").min().alias("min_timestamp"),
        pl.col("time_stamp").max().alias("max_timestamp"),
    ).sort("dataset").to_dicts()
    ads = pl.read_csv(ROOT / "data/raw/ad_feature.csv", schema_overrides={"brand": pl.Float64},
                      null_values=["", "NULL", "null"])
    assert ads["adgroup_id"].n_unique() == ads.height
    linked = clicks.join(ads.select(
        pl.col("adgroup_id").alias("adgroup_id_raw"),
        pl.col("cate_id").alias("source_category"),
        pl.col("brand").cast(pl.Int64).alias("source_brand"),
    ), on="adgroup_id_raw", how="left", validate="m:1")
    mismatches = linked.filter(
        ~pl.col("cate_id_raw").eq_missing(pl.col("source_category"))
        | ~pl.col("brand_raw").eq_missing(pl.col("source_brand"))
    ).height
    assert mismatches == 0
    report["clicked_ad_static_alignment"] = {
        "audited_clicks": linked.height, "category_or_brand_mismatches": mismatches,
        "ad_table_rows": ads.height, "ad_key_unique": True,
    }
    duplicated = clicks.group_by("user_raw", "time_stamp").len().filter(pl.col("len") > 1)
    report["click_timestamp_ties"] = {
        "user_second_groups_with_multiple_clicks": duplicated.height,
        "events_in_tied_groups": int(duplicated["len"].sum()),
        "max_events_per_user_second": int(duplicated["len"].max() or 0),
        "rule": "stable source impression_id breaks serialization ties only, never availability ties",
    }

    daily = impressions.group_by("user_raw", "event_date_bj", "dataset").agg(
        pl.len().alias("rows"), pl.col("clk").sum().cast(pl.UInt64).alias("clicks")
    ).sort("user_raw", "event_date_bj").with_columns(
        (pl.col("clicks").cum_sum().over("user_raw") - pl.col("clicks")).alias("prior_clicks")
    )
    summary = [
        pl.col("rows").sum().alias("rows"),
        pl.col("rows").filter(pl.col("prior_clicks") > 0).sum().alias("covered_rows"),
        pl.col("prior_clicks").max().alias("max_prior_clicks"),
        (pl.col("rows") * pl.col("prior_clicks")).sum().alias("impression_weighted_tokens"),
        *[pl.col("rows").filter(pl.col("prior_clicks") > n).sum().alias(f"truncated_rows_L{n}")
          for n in (10, 20, 50)],
    ]
    report["click_history_by_date"] = daily.group_by("dataset", "event_date_bj").agg(
        summary).sort("event_date_bj").to_dicts()
    report["click_history_by_split"] = daily.group_by("dataset").agg(
        summary).sort("dataset").to_dicts()
    report["click_history_user_date_entries"] = daily.height
    report["length_stratum_singletons"] = daily.with_columns(
        pl.col("prior_clicks").clip(upper_bound=20).alias("length")
    ).filter(pl.col("length") > 0).group_by("event_date_bj", "length").len().filter(
        pl.col("len") < 2).to_dicts()

    # Candidate/history equality is audited on raw IDs, before encoding.
    alignment = []
    for day in impressions["event_date_bj"].unique().sort().to_list():
        target = impressions.filter(pl.col("event_date_bj") == day)
        past = clicks.filter(pl.col("event_date_bj") < day)
        row = {"date": day, "rows": target.height}
        for field in ("cate_id_raw", "adgroup_id_raw"):
            keys = past.select("user_raw", field).unique()
            row[f"rows_with_prior_same_{field}"] = target.join(
                keys, on=["user_raw", field], how="semi").height
        alignment.append(row)
    report["candidate_alignment_untruncated_history"] = alignment

    report["existing_behavior_coverage"] = [
        {"split": split, **pl.scan_parquet(ROOT / f"data/processed/features/{split}.parquet")
         .select(pl.len().alias("rows"),
                 pl.col("has_behavior_history_14d").mean().alias("coverage_14d"),
                 pl.col("event_count_14d").median().alias("median_events_14d"),
                 pl.col("event_count_14d").quantile(0.99).alias("p99_events_14d"))
         .collect().row(0, named=True)} for split in ("train", "val")
    ]
    daily_path = ROOT / "data/processed/behavior/user_daily_behavior.parquet"
    report["daily_behavior_schema"] = str(pq.ParquetFile(daily_path).schema_arrow)
    report["daily_behavior_prior_to_validation"] = pl.scan_parquet(daily_path).filter(
        pl.col("event_date_bj") < date(2017, 5, 12)
    ).select(pl.len().alias("user_days"), pl.col("event_date_bj").min().alias("min_date"),
             pl.col("event_date_bj").max().alias("max_date"),
             *[pl.col(c).cast(pl.UInt64).sum() for c in
               ("event_count", "pv_count", "cart_count", "fav_count", "buy_count")]
    ).collect().row(0, named=True)

    behavior = pl.read_csv(ROOT / "data/raw/behavior_log.csv", n_rows=200_000,
                           schema_overrides={"user": pl.Int64, "time_stamp": pl.Int64,
                                             "btag": pl.String, "cate": pl.Int64, "brand": pl.Int64})
    start = int(datetime(2017, 4, 21, 16, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2017, 5, 11, 16, tzinfo=timezone.utc).timestamp())
    eligible = behavior.filter((pl.col("time_stamp") >= start) & (pl.col("time_stamp") < end))
    report["behavior_prefix_sample"] = {
        "rows_read": behavior.height,
        "rows_in_allowed_prevalidation_window": eligible.height,
        "btag_counts_in_window": eligible.group_by("btag").len().sort("btag").to_dicts(),
        "distinct_users_in_window": eligible["user"].n_unique(),
        "category_null_rows": eligible["cate"].null_count(),
        "brand_null_rows": eligible["brand"].null_count(),
        "brand_zero_rows": eligible.filter(pl.col("brand") == 0).height,
    }
    for raw, target, mapping in (("cate", "cate_id_raw", "cate_id_raw"),
                                  ("brand", "brand_raw", "brand_raw")):
        vocabulary = pl.read_parquet(ROOT / f"artifacts/encoders/mappings/{mapping}_mapping.parquet")
        report["behavior_prefix_sample"][f"{raw}_train_vocabulary_match_rate"] = (
            eligible.join(vocabulary.select(pl.col(target).alias(raw)), on=raw, how="semi").height
            / eligible.height
        )
    report["limitations"] = [
        "CSV prefix is not a random sample; do not extrapolate category/brand coverage.",
        "No separate click occurrence or feedback arrival timestamp exists in exposure schema.",
        "Behavior category/brand numeric overlap does not prove shared identifier semantics.",
        "Daily aggregate cannot reconstruct ordered category/brand events.",
        "Resource measurements describe this audit process, not a sequence build or training benchmark.",
    ]
    report["checks_passed"] = ["train_val_row_count", "unique_impression_id", "binary_labels",
                               "timestamp_date_consistency", "train_val_click_count",
                               "clicked_ad_category_brand_matches_raw_ad_table"]
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_bytes_macos"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
