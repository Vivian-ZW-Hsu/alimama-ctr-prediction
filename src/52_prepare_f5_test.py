"""Assemble the authorized F5 test freeze and the bucket-relaxation markers.

Reads the validation freeze, verifies every hash, binds the pre-committed test
analysis plan, and records the per-impression donor bucket-relaxation level needed
by the predefined sensitivity analysis. No inference is performed here.
"""

from __future__ import annotations

import json
import subprocess

import polars as pl

from ctr_nn_common import BASE_MODEL, BASE_SCALER
from f5_sequences import MAX_LENGTH, ROOT, sha256, source_fingerprints, write_json
from f5_test_common import (
    BUILD, STUDY, TEST_DATE, TEST_ROWS, TEST_STUDY, bucket_relaxation_markers,
    relaxation_summary, relaxed_rows_only,
)
from project_environment import require_tested_environment

TEST_FREEZE = STUDY / "test_freeze_manifest.json"

AUTHORIZATION = (
    "The user authorized the first formal F5 test on 2026-09-27, after the "
    "validation freeze and after the test analysis plan was committed. One "
    "inference pass per unique frozen checkpoint. No retuning, no changed "
    "retained length, learning rate, donor assignment or primary model, and no "
    "retraining."
)


def main() -> None:
    environment = require_tested_environment("F5 test preparation")
    if TEST_FREEZE.exists():
        raise FileExistsError("Test freeze already assembled; it is immutable")

    freeze_path = STUDY / "freeze_manifest.json"
    freeze = json.loads(freeze_path.read_text())
    if not freeze["complete"] or freeze["test_inference_performed"]:
        raise ValueError("validation freeze is not in the pre-test state")

    plan_path = STUDY / "test_analysis_plan.json"
    plan = json.loads(plan_path.read_text())
    if not plan["declared_before_first_formal_f5_test"]:
        raise ValueError("the analysis plan does not declare itself pre-test")
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(plan_path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError("the test analysis plan must be committed before the test")

    # Every frozen input must still be byte-identical.
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
    build_config = json.loads((ROOT / "results/f5_behavior_v1/build_config.json").read_text())
    if source_fingerprints() != build_config["inputs"]:
        raise ValueError("F5 build inputs changed")
    f5_path = ROOT / "results/f5_behavior_v1/manifest.json"
    if sha256(f5_path) != freeze["f5_manifest_sha256"]:
        raise ValueError("F5 build manifest changed")
    f5 = json.loads(f5_path.read_text())
    for relative in (f"data/processed/f5_behavior_v1/test_{TEST_DATE}.parquet",
                     "data/processed/f5_behavior_v1/events.parquet"):
        if sha256(ROOT / relative) != f5["output_sha256"][relative]:
            raise ValueError(f"{relative} changed since the build")

    records = []
    for checkpoint in freeze["checkpoints"]:
        path = ROOT / "artifacts/models" / checkpoint["run_name"] / "best_model.pt"
        if sha256(path) != checkpoint["checkpoint_sha256"]:
            raise ValueError(f"{checkpoint['run_name']} checkpoint changed")
        records.append({**checkpoint, "test_run_name": checkpoint["run_name"] + "_test_v1"})
    if len(records) != 15:
        raise ValueError(f"expected 15 checkpoints, found {len(records)}")
    if len({r["test_run_name"] for r in records}) != 15:
        raise ValueError("test run names are not unique")

    TEST_STUDY.mkdir(parents=True, exist_ok=True)
    markers = bucket_relaxation_markers()
    if markers.height != TEST_ROWS:
        raise ValueError(f"markers cover {markers.height} rows, expected {TEST_ROWS}")
    marker_path = TEST_STUDY / "bucket_relaxation_excluded_rows.parquet"
    if not marker_path.exists():
        relaxed_rows_only(markers).write_parquet(marker_path, compression="zstd")
    summary = relaxation_summary(markers)
    write_json(TEST_STUDY / "bucket_relaxation_summary.json", summary)

    code = [
        "f5_sequences.py", "f5_sequence_ctr.py", "ctr_nn_common.py",
        "47_train_f5_sequence_ctr.py", "10_evaluate_predictions.py",
        "f5_test_common.py", "52_prepare_f5_test.py", "53_infer_f5_test.py",
    ]
    write_json(
        TEST_FREEZE,
        {
            "version": 1,
            "complete": True,
            "budget": "F5",
            "authorization": AUTHORIZATION,
            "test_authorized": True,
            "test_inference_performed_at_freeze": False,
            "passes_authorized_per_checkpoint": 1,
            "validation_freeze": str(freeze_path.relative_to(ROOT)),
            "validation_freeze_sha256": sha256(freeze_path),
            "test_analysis_plan_sha256": sha256(plan_path),
            "comparison_plan": freeze["comparison_plan"],
            "primary_model": freeze["primary_model"],
            "selected_learning_rate": freeze["selected_learning_rate"],
            "max_length": MAX_LENGTH,
            "base_model_sha256": freeze["base_model_sha256"],
            "base_scaler_sha256": freeze["base_scaler_sha256"],
            "f5_manifest_sha256": freeze["f5_manifest_sha256"],
            "cache_manifest_sha256": freeze["cache_manifest_sha256"],
            "checkpoints": records,
            "bucket_relaxation": {
                **summary,
                "excluded_rows_path": str(marker_path.relative_to(ROOT)),
                "excluded_rows_sha256": sha256(marker_path),
                "full_per_row_levels": (
                    "identical to the `fallback` column of "
                    f"data/processed/f5_behavior_v1/test_{TEST_DATE}.parquet; "
                    "reproduced by f5_test_common.bucket_relaxation_markers(). A "
                    "15 MiB copy is not committed because only the exclusion list "
                    "is needed and the source is already frozen and hashed."
                ),
            },
            "code_sha256": {name: sha256(ROOT / "src" / name) for name in code},
            "environment": environment,
            "gate": (
                "The formal pass additionally requires a passing inference smoke "
                "certificate and a clean worktree pushed to the tracking branch."
            ),
        },
    )
    print(f"Assembled the F5 test freeze: {len(records)} checkpoints", flush=True)
    print(
        f"bucket_relaxation: {summary['relaxed_rows']:,} of {summary['rows']:,} rows "
        f"({summary['relaxed_share']:.4%}) | levels {summary['levels']}",
        flush=True,
    )
    print(f"excluded rows: {marker_path.relative_to(ROOT)} "
          f"({marker_path.stat().st_size / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
