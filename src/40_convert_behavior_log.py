"""Losslessly convert data/raw/behavior_log.csv to Parquet for auditing and F5.

The 22 GiB source CSV is re-read for every streaming aggregation, which makes the
namespace audit and the F5 event store expensive. This stage performs exactly one
conversion and then proves it lossless by computing an identical fingerprint
bundle from the CSV and from the Parquet copy and requiring exact equality.

Nothing here reads model inputs, predictions or any frozen result. The Parquet
copy is a derived artifact under the Git-ignored data/ tree; the committed
manifest is evidence about it, not a backup of it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import polars as pl

from project_environment import require_tested_environment

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "raw" / "behavior_log.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "behavior_raw_v1"
OUTPUT_PARQUET = OUTPUT_DIR / "behavior_log.parquet"
RESULT_DIR = ROOT / "results" / "behavior_namespace_audit_v1"

# The published schema of the source file.  Declaring it explicitly prevents a
# silent dtype inference change between runs.
SCHEMA = {
    "user": pl.Int64,
    "time_stamp": pl.Int64,
    "btag": pl.String,
    "cate": pl.Int64,
    "brand": pl.Int64,
}

NUMERIC_COLUMNS = ["user", "time_stamp", "cate", "brand"]

HASH_BLOCK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Full-content SHA256, streamed so a 22 GiB file never enters memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fingerprint(frame: pl.LazyFrame) -> dict[str, object]:
    """Aggregate bundle used to compare the CSV and the Parquet copy.

    Column sums are accumulated in Int128 because the time_stamp sum exceeds the
    exact range of Float64 and would silently lose precision.
    """
    expressions: list[pl.Expr] = [pl.len().alias("rows")]
    for column in NUMERIC_COLUMNS:
        expressions += [
            pl.col(column).null_count().alias(f"{column}__nulls"),
            pl.col(column).min().alias(f"{column}__min"),
            pl.col(column).max().alias(f"{column}__max"),
            pl.col(column).cast(pl.Int128).sum().alias(f"{column}__sum"),
        ]
    expressions += [
        pl.col("btag").null_count().alias("btag__nulls"),
        pl.col("btag").n_unique().alias("btag__n_unique"),
    ]
    row = frame.select(expressions).collect(engine="streaming").row(0, named=True)

    # btag is tiny, so an exact per-value count is cheap and catches row loss
    # that a total row count alone could mask.
    btag_counts = (
        frame.group_by("btag")
        .agg(pl.len().alias("events"))
        .collect(engine="streaming")
        .sort("btag")
    )
    row["btag__counts"] = {
        str(name): int(events) for name, events in btag_counts.iter_rows()
    }
    return {key: (int(value) if isinstance(value, bool) is False and isinstance(value, int) else value)
            for key, value in row.items()}


def scan_source() -> pl.LazyFrame:
    return pl.scan_csv(SOURCE_CSV, schema=SCHEMA)


def scan_parquet() -> pl.LazyFrame:
    return pl.scan_parquet(OUTPUT_PARQUET)


def main() -> None:
    environment = require_tested_environment("E1.0 behavior_log conversion")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="skip the full SHA256 of the 22 GiB source (recorded as null)",
    )
    parser.add_argument(
        "--reverify",
        action="store_true",
        help="do not convert; only recompute and compare fingerprints",
    )
    args = parser.parse_args()

    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"missing source: {SOURCE_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if OUTPUT_PARQUET.exists() and not args.reverify:
        raise FileExistsError(
            f"{OUTPUT_PARQUET} already exists; pass --reverify to re-check it "
            "instead of rewriting a derived artifact"
        )

    if not args.reverify:
        print(f"Converting {SOURCE_CSV.name} ({SOURCE_CSV.stat().st_size / 1024**3:.2f} GiB)...")
        convert_started = time.perf_counter()
        scan_source().sink_parquet(OUTPUT_PARQUET, compression="zstd")
        print(f"  wrote {OUTPUT_PARQUET.name} in {time.perf_counter() - convert_started:.1f} s")

    print("Fingerprinting Parquet copy...")
    parquet_started = time.perf_counter()
    parquet_fingerprint = fingerprint(scan_parquet())
    print(f"  done in {time.perf_counter() - parquet_started:.1f} s")

    print("Fingerprinting source CSV (independent streaming pass)...")
    csv_started = time.perf_counter()
    csv_fingerprint = fingerprint(scan_source())
    print(f"  done in {time.perf_counter() - csv_started:.1f} s")

    differences = {
        key: {"csv": csv_fingerprint[key], "parquet": parquet_fingerprint.get(key)}
        for key in csv_fingerprint
        if csv_fingerprint[key] != parquet_fingerprint.get(key)
    }
    if differences:
        raise ValueError(f"conversion is not lossless; differing fields: {differences}")
    print("Fingerprints match exactly.")

    source_hash = None if args.skip_source_hash else sha256_file(SOURCE_CSV)

    manifest = {
        "stage": "E1.0 lossless behavior_log conversion",
        "source": {
            "path": str(SOURCE_CSV.relative_to(ROOT)),
            "bytes": SOURCE_CSV.stat().st_size,
            "mtime_epoch": int(SOURCE_CSV.stat().st_mtime),
            "sha256": source_hash,
        },
        "output": {
            "path": str(OUTPUT_PARQUET.relative_to(ROOT)),
            "bytes": OUTPUT_PARQUET.stat().st_size,
            "compression": "zstd",
        },
        "declared_schema": {name: str(dtype) for name, dtype in SCHEMA.items()},
        "fingerprint": csv_fingerprint,
        "fingerprints_match": True,
        "verification": (
            "The same aggregate bundle (row count, per-column null/min/max, "
            "Int128 column sums, exact btag value counts) was computed "
            "independently from the CSV and from the Parquet copy and compared "
            "for exact equality."
        ),
        "implementation_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "polars_version": pl.__version__,
        "environment": environment,
    }
    manifest_path = RESULT_DIR / "behavior_log_conversion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(f"\nrows                {csv_fingerprint['rows']:,}")
    print(f"parquet size        {OUTPUT_PARQUET.stat().st_size / 1024**3:.2f} GiB")
    print(f"btag counts         {csv_fingerprint['btag__counts']}")
    print(f"elapsed             {manifest['elapsed_seconds']:.1f} s")
    print(f"manifest            {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
