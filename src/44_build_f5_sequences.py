"""Build and audit the F5 behavior-log sequence references. No model execution.

F5 is a new information budget beside F4. This entry point writes an immutable
event store plus per-date history references, and refuses to continue if any
integrity check fails. It never reads F4 artifacts, predictions or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import date
from pathlib import Path

import polars as pl

from f5_sequences import (
    ASSUMPTION, FORMAL_END_EXCLUSIVE, FORMAL_START, MAX_LENGTH, PAD_INDEX, ROOT,
    SEED, TYPE_TOKENS, TYPE_VOCABULARY_SIZE, UNKNOWN_INDEX, assert_f123_identity,
    attach_histories, audit_references, build_event_store, daily_histories,
    impression_activity, load_event_arrays, sha256, source_fingerprints,
    targets_for_day, verify_event_store, write_json,
)

TOTAL_IMPRESSIONS = 26_557_961


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-certificate", type=Path)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--partitions", type=int, default=16)
    parser.add_argument(
        "--strict-no-fallback",
        action="store_true",
        help="fail if any donor required bucket coarsening (not achievable at L=200)",
    )
    args = parser.parse_args()
    if not args.run_name.replace("_", "").replace("-", "").isalnum():
        parser.error("Invalid run name")
    if args.max_length != MAX_LENGTH:
        parser.error(f"This authorized build is fixed to L={MAX_LENGTH}")
    started = time.perf_counter()

    inputs = source_fingerprints()
    config = {
        "version": 1,
        "max_length": args.max_length,
        "seed": SEED,
        "smoke": args.smoke_test,
        "partitions": args.partitions,
        "activity_threshold": 100,
        "recency_buckets": "whole Beijing days since the last retained event, clipped at 2",
        "window_start": str(FORMAL_START),
        "window_end_exclusive": str(FORMAL_END_EXCLUSIVE),
        "inputs": inputs,
        "implementation_sha256": sha256(Path(__file__)),
        "library_sha256": sha256(ROOT / "src/f5_sequences.py"),
    }

    output = ROOT / "data/processed" / args.run_name
    result = ROOT / "results" / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)

    config_path = result / "build_config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise ValueError("Resume config/input mismatch; inspect before rebuilding")
        if (result / "manifest.json").exists():
            manifest = json.loads((result / "manifest.json").read_text())
            for name, digest in manifest["output_sha256"].items():
                if sha256(ROOT / name) != digest:
                    raise ValueError(f"{name} changed since the build completed")
            print("Verified complete immutable build; nothing changed.", flush=True)
            return
    else:
        write_json(config_path, config)

    if not args.smoke_test:
        if not args.smoke_certificate:
            parser.error("Full build requires a passed reproducible smoke certificate")
        certificate = json.loads(args.smoke_certificate.read_text())
        if not (certificate["passed"] and certificate["reproducible"]):
            raise ValueError("Smoke certificate did not pass")
        for key in ("max_length", "seed", "partitions", "activity_threshold",
                    "implementation_sha256", "library_sha256"):
            if certificate["config"][key] != config[key]:
                raise ValueError(f"Smoke certificate disagrees on {key}")

    store_path = output / "events.parquet"
    store_report_path = result / "event_store.json"
    if store_report_path.exists():
        store_report = json.loads(store_report_path.read_text())
        if sha256(store_path) != store_report["sha256"]:
            raise ValueError("event store changed since it was audited")
        print("Reusing audited event store.", flush=True)
    else:
        print("Building partition-sorted behavior event store...", flush=True)
        store_report = build_event_store(store_path, args.partitions, args.smoke_test)
        write_json(store_report_path, store_report)
    print(
        f"  events={store_report['events']:,} users={store_report['users']:,} "
        f"unknown_category_share={store_report['unknown_category_share']:.6%}",
        flush=True,
    )

    print("Loading narrow event arrays for the reference audit...", flush=True)
    arrays = load_event_arrays(store_path)
    events = pl.scan_parquet(store_path)
    activity = impression_activity()

    reports = []
    outputs = [store_path]
    for day_number in range(6, 14):
        day = date(2017, 5, day_number)
        split = "train" if day_number < 12 else "val" if day_number == 12 else "test"
        refs_path = output / f"{split}_{day}.parquet"
        history_path = output / f"history_{day}.parquet"
        audit_path = result / f"audit_{day}.json"

        if audit_path.exists():
            report = json.loads(audit_path.read_text())
            if sha256(refs_path) != report["refs_sha256"]:
                raise ValueError(f"{refs_path} changed since it was audited")
            if sha256(history_path) != report["history_sha256"]:
                raise ValueError(f"{history_path} changed since it was audited")
        else:
            history = daily_histories(events, activity, day, args.max_length, SEED)
            # Integrity failures (self-match, length mismatch, future events,
            # identity errors) must be zero and are enforced in audit_references.
            # Donor *bucket coarsening* cannot be forced to zero at L=200 without
            # abandoning exact retained-length matching, which is the stronger
            # guarantee, so it is counted by level instead of prohibited.
            fallback_levels = {
                level: int((history["fallback"] == level).sum()) for level in range(4)
            }
            coarsened = sum(fallback_levels[level] for level in (1, 2, 3))
            if coarsened and args.strict_no_fallback:
                raise ValueError(
                    f"{day}: donor bucket coarsening on {coarsened:,} users "
                    f"{fallback_levels}; --strict-no-fallback forbids it"
                )
            targets = targets_for_day(split, day, args.smoke_test)
            refs = attach_histories(targets, history)
            assert_f123_identity(refs, split, day, args.smoke_test)
            report = audit_references(refs, arrays, day, args.max_length)
            report.update(
                {
                    "split": split,
                    "f123_id_label_parity": True,
                    "donor_pool_users": history.height,
                    "donor_pool_fallback_levels": fallback_levels,
                    "donor_pool_coarsened_users": coarsened,
                    "reference_schema": {k: str(v) for k, v in refs.schema.items()},
                }
            )
            for frame, destination in [(history, history_path), (refs, refs_path)]:
                temporary = destination.with_suffix(".partial")
                frame.write_parquet(temporary, compression="zstd", row_group_size=100000)
                temporary.replace(destination)
            report.update(
                {
                    "refs_sha256": sha256(refs_path),
                    "history_sha256": sha256(history_path),
                }
            )
            write_json(audit_path, report)
        reports.append(report)
        outputs.extend([refs_path, history_path])
        print(
            f'{day} {split} rows={report["rows"]:,} mean_retained={report["mean_retained"]:.2f} '
            f'empty={report["empty_share"]:.4%} '
            f'unknown={report["token_stats"]["matched"]["unknown_share"]:.4%} '
            f'coarsened={report["donor_pool_coarsened_users"]:,}; '
            "cutoff/identity/order/length checks passed",
            flush=True,
        )

    if not args.smoke_test:
        total = sum(r["rows"] for r in reports)
        if total != TOTAL_IMPRESSIONS:
            raise ValueError(f"covered {total:,} impressions, expected {TOTAL_IMPRESSIONS:,}")

    for name, before in inputs.items():
        path = ROOT / name
        if (path.stat().st_size, path.stat().st_mtime_ns) != (before["bytes"], before["mtime_ns"]):
            raise ValueError(f"input {name} changed during the build")

    manifest = {
        "run_name": args.run_name,
        "complete": True,
        "config": config,
        "source": "behavior_log events; category + behavior-type tokens",
        "budget": "F5, a new information budget; F4 is neither read nor modified",
        "availability_assumption": ASSUMPTION,
        "cutoff": "event Beijing date strictly less than target Beijing date; no same-day events",
        "vocabulary": (
            "category reuses the existing train-only ad-category mapping so history "
            "and candidate share one space; behavior-only categories map to UNKNOWN. "
            "Behavior type is a fixed four-value enumeration from the dataset "
            "documentation, not learned from any split."
        ),
        "reserved_indices": {"PAD": PAD_INDEX, "UNKNOWN": UNKNOWN_INDEX},
        "type_tokens": TYPE_TOKENS,
        "type_vocabulary_size": TYPE_VOCABULARY_SIZE,
        "brand_tokens": "admissible per the E1 audit but deliberately unused in the primary budget",
        "smoke_selector": "user_raw % 4000 == 0 on both the event store and every target date" if args.smoke_test else None,
        "test_scope": "data construction and ID/label integrity only; no model or test metrics",
        "event_store": store_report,
        "total_rows": sum(r["rows"] for r in reports),
        "days": reports,
        "output_directory": str(output.relative_to(ROOT)),
        "output_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
        "output_bytes": sum(p.stat().st_size for p in outputs),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(result / "manifest.json", manifest)
    print(f'Complete: {result / "manifest.json"}', flush=True)


if __name__ == "__main__":
    main()
