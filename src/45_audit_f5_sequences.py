"""Independent persisted-artifact audit and reproducible-smoke certification for F5.

Re-derives every per-date reference audit from the written Parquet files rather
than trusting the build's own report, and certifies that two independent smoke
builds produced byte-identical artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from project_environment import require_tested_environment

from f5_sequences import (
    ROOT, assert_f123_identity, audit_references, load_event_arrays, sha256,
    verify_event_store, write_json,
)


def audit(name: str) -> dict:
    manifest_path = ROOT / "results" / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["complete"]:
        raise ValueError(f"{name} is not a complete build")
    output = ROOT / manifest["output_directory"]

    for file, digest in manifest["output_sha256"].items():
        if sha256(ROOT / file) != digest:
            raise ValueError(f"{file} changed since the build completed")

    # Re-verify the two structural properties the pointer scheme relies on.
    store = verify_event_store(
        output / "events.parquet", manifest["config"]["partitions"]
    )
    for key in ("events", "users", "unknown_category_tokens", "sha256"):
        if store[key] != manifest["event_store"][key]:
            raise ValueError(f"event store {key} disagrees with the build manifest")

    arrays = load_event_arrays(output / "events.parquet")
    for previous in manifest["days"]:
        day = date.fromisoformat(previous["date"])
        refs = pl.read_parquet(output / f'{previous["split"]}_{day}.parquet')
        assert_f123_identity(refs, previous["split"], day, manifest["config"]["smoke"])
        observed = audit_references(refs, arrays, day, manifest["config"]["max_length"])
        for key, value in observed.items():
            if value != previous[key]:
                raise ValueError(f'{previous["date"]} {key}: {value} != {previous[key]}')
    return manifest


def main() -> None:
    environment = require_tested_environment("E4 F5 independent audit")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--compare-smoke")
    parser.add_argument("--certificate", type=Path)
    args = parser.parse_args()

    first = audit(args.run_name)
    if args.compare_smoke:
        second = audit(args.compare_smoke)
        if not (first["config"]["smoke"] and second["config"]["smoke"]):
            raise ValueError("both runs must be smoke builds")
        # The run name differs, so compare everything else in the config.
        if first["config"] != second["config"]:
            raise ValueError("smoke configs differ")
        one = {Path(k).name: v for k, v in first["output_sha256"].items()}
        two = {Path(k).name: v for k, v in second["output_sha256"].items()}
        if one != two:
            differing = sorted(k for k in one if one[k] != two.get(k))
            raise ValueError(f"smoke Parquet files are not byte-reproducible: {differing}")
        if args.certificate is None or args.certificate.exists():
            raise ValueError("certificate path must be new; creation is exclusive")
        write_json(
            args.certificate,
            {
                "passed": True,
                "reproducible": True,
                "runs": [args.run_name, args.compare_smoke],
                "config": first["config"],
                "artifact_sha256": one,
                "checked_dates": len(first["days"]),
                "event_store_reverified": True,
            },
        )
        print(f"Smoke reproducibility certified: {args.certificate}", flush=True)
    print("Independent F5 integrity audit passed.", flush=True)


if __name__ == "__main__":
    main()
