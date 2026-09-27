"""One immutable streaming F5 inference pass. Smoke mode uses validation only.

The formal pass requires a passing smoke certificate and a clean worktree pushed
to the tracking branch. An exclusive start marker is written before the first
forward call, so a partial run can never be replayed silently.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ctr_nn_common import BASE_MODEL, BASE_SCALER, load_frozen, tabular
from f5_sequences import MAX_LENGTH, ROOT, TYPE_VOCABULARY_SIZE, sha256, source_fingerprints, write_json
from f5_sequence_ctr import F5SequenceResidual, ZeroF5SequenceResidual, gather_f5_tokens
from f5_test_common import BUILD, STUDY, TEST_DATE, TEST_ROWS, VALIDATION_DATE
from project_environment import require_tested_environment

TEST_FREEZE = STUDY / "test_freeze_manifest.json"


def main() -> None:
    require_tested_environment("F5 formal test inference")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    split = "val" if args.smoke_test else "test"
    date = VALIDATION_DATE if args.smoke_test else TEST_DATE
    study = ROOT / "results" / ("f5_sequence_test_v1_smoke" if args.smoke_test else "f5_sequence_test_v1")
    marker_path = study / "inference_started.json"
    if marker_path.exists():
        raise FileExistsError("Inference already started; automatic replay is forbidden")

    freeze = json.loads(TEST_FREEZE.read_text())
    if not (freeze["complete"] and freeze["test_authorized"]):
        raise ValueError("the test freeze does not authorize inference")
    if sha256(STUDY / "test_analysis_plan.json") != freeze["test_analysis_plan_sha256"]:
        raise ValueError("the test analysis plan changed after the freeze")
    if sha256(STUDY / "freeze_manifest.json") != freeze["validation_freeze_sha256"]:
        raise ValueError("the validation freeze changed")
    for name, digest in freeze["code_sha256"].items():
        if sha256(ROOT / "src" / name) != digest:
            raise ValueError(f"{name} changed after the test freeze")
    if sha256(BASE_MODEL) != freeze["base_model_sha256"]:
        raise ValueError("frozen backbone changed")
    if sha256(BASE_SCALER) != freeze["base_scaler_sha256"]:
        raise ValueError("frozen scaler changed")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not args.smoke_test:
        certificate_path = STUDY / "f5_inference_smoke_certificate.json"
        certificate = json.loads(certificate_path.read_text())
        if not certificate["passed"] or certificate["freeze_sha256"] != sha256(TEST_FREEZE):
            raise ValueError("the smoke certificate does not match this freeze")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
            raise RuntimeError("the worktree must be clean before the formal pass")
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True
        ).strip()
        remote = subprocess.check_output(
            ["git", "ls-remote", "origin", f"refs/heads/{branch}"], cwd=ROOT, text=True
        ).split()
        if not remote or remote[0] != head:
            raise RuntimeError("the pre-test commit must be pushed to the tracking branch first")

    build_config = json.loads((ROOT / "results/f5_behavior_v1/build_config.json").read_text())
    if source_fingerprints() != build_config["inputs"]:
        raise ValueError("F5 build inputs changed")
    f5_path = ROOT / "results/f5_behavior_v1/manifest.json"
    if sha256(f5_path) != freeze["f5_manifest_sha256"]:
        raise ValueError("F5 build manifest changed")
    f5 = json.loads(f5_path.read_text())
    reference_path = BUILD / f"{split}_{date}.parquet"
    for path in (reference_path, BUILD / "events.parquet"):
        if sha256(path) != f5["output_sha256"][str(path.relative_to(ROOT))]:
            raise ValueError(f"{path} changed since the build")

    references = pl.read_parquet(reference_path).sort("impression_id")
    if args.smoke_test:
        references = references.filter(pl.col("impression_id") % 1000 == 0)
    reference_ids = references["impression_id"].to_numpy()
    arrays = {key: references[key].to_numpy() for key in ("start", "donor_start", "length", "clk")}
    events = pl.read_parquet(BUILD / "events.parquet", columns=["cate_idx", "type_idx"])
    categories = events["cate_idx"].to_numpy()
    types = events["type_idx"].to_numpy()
    del events

    device = tabular.choose_device("auto")
    backbone, base_config, mean, scale = load_frozen(device)
    categorical, binary, numeric = [
        base_config[k] for k in ("categorical_features", "binary_features", "numeric_features")
    ]
    category_position = categorical.index("cate_idx")
    embedding = backbone.embeddings[category_position].weight.detach()
    if backbone.training or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("backbone must be frozen and in eval mode")

    models = []
    for record in freeze["checkpoints"]:
        path = ROOT / "artifacts/models" / record["run_name"] / "best_model.pt"
        if sha256(path) != record["checkpoint_sha256"]:
            raise ValueError(f"{record['run_name']} checkpoint changed")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != record["best_epoch"]:
            raise ValueError(f"{record['run_name']} epoch disagrees with the freeze")
        if checkpoint["base_model_sha256"] != freeze["base_model_sha256"]:
            raise ValueError(f"{record['run_name']} was trained on a different backbone")
        model_class = ZeroF5SequenceResidual if record["zero_sequence"] else F5SequenceResidual
        model = model_class(
            embedding, pooling=record["pooling"], type_vocabulary_size=TYPE_VOCABULARY_SIZE
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.requires_grad_(False)
        model.assert_neutral()
        run = record["test_run_name"] + ("_smoke" if args.smoke_test else "")
        destination = ROOT / "artifacts/predictions" / run
        if destination.exists():
            raise FileExistsError(f"{destination} already exists")
        models.append((record, model, run, destination))

    # Exclusive marker before the first forward call.
    study.mkdir(parents=True, exist_ok=True)
    with marker_path.open("x") as out:
        json.dump(
            {
                "split": split, "smoke": args.smoke_test,
                "freeze_sha256": sha256(TEST_FREEZE), "git_head": head,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "passes_authorized_per_checkpoint": 1,
            },
            out, indent=2,
        )

    writers = {}
    seen = np.zeros(references.height, dtype=bool)
    rows = 0
    started = time.perf_counter()
    columns = ["impression_id", "clk", *categorical, *binary, *numeric]
    try:
        for batch in tabular.iter_record_batches(
            ROOT / f"data/processed/model_input_f3/{split}.parquet", columns, 8192, None
        ):
            ids = tabular.arrow_column_to_numpy(batch, "impression_id", np.uint32)
            if args.smoke_test:
                batch = batch.filter(pa.array(ids % 1000 == 0))
                if batch.num_rows == 0:
                    continue
                ids = tabular.arrow_column_to_numpy(batch, "impression_id", np.uint32)
            position = np.searchsorted(reference_ids, ids)
            if not (position < len(reference_ids)).all():
                raise ValueError("an impression is missing from the F5 references")
            if not np.array_equal(reference_ids[position], ids) or seen[position].any():
                raise ValueError("reference alignment failure")
            seen[position] = True
            labels = tabular.arrow_column_to_numpy(batch, "clk", np.uint8)
            np.testing.assert_array_equal(labels, arrays["clk"][position])
            encoded = tabular.categorical_numpy(batch, categorical)
            dense = tabular.dense_numpy(batch, binary, numeric, mean, scale)
            candidate = torch.from_numpy(encoded[:, category_position].copy()).to(device)
            tokens = {}
            for source, key in (("matched", "start"), ("mismatched", "donor_start")):
                category_tokens, type_tokens = gather_f5_tokens(
                    categories, types, arrays[key][position], arrays["length"][position], MAX_LENGTH
                )
                tokens[source] = (
                    torch.from_numpy(category_tokens).to(device),
                    torch.from_numpy(type_tokens).to(device),
                )
            with torch.inference_mode():
                base = backbone(
                    torch.from_numpy(encoded).to(device), torch.from_numpy(dense).to(device)
                )
                base_probability = torch.sigmoid(base).cpu().numpy()
                for record, model, run, destination in models:
                    history, history_type = tokens[record["source"]]
                    delta = model(candidate, history, history_type)
                    empty = ~(history > 1).any(dim=1)
                    if torch.count_nonzero(delta[empty]).item():
                        raise ValueError(f"{run}: empty history produced a nonzero residual")
                    if record["zero_sequence"] and torch.count_nonzero(delta).item():
                        raise ValueError(f"{run}: zero control produced a nonzero residual")
                    predictions = torch.sigmoid(base + delta).cpu().numpy()
                    if not np.isfinite(predictions).all():
                        raise ValueError(f"{run}: non-finite prediction")
                    if record["zero_sequence"]:
                        np.testing.assert_array_equal(predictions, base_probability)
                    table = pa.table(
                        {"impression_id": ids, "clk": labels, "prediction": predictions}
                    )
                    if run not in writers:
                        destination.mkdir(parents=True)
                        writers[run] = pq.ParquetWriter(
                            destination / f"{split}_predictions.partial", table.schema,
                            compression="zstd",
                        )
                    writers[run].write_table(table)
            rows += len(ids)
    finally:
        for writer in writers.values():
            writer.close()

    if rows != references.height or not seen.all():
        raise ValueError(f"covered {rows} of {references.height} rows")
    if not args.smoke_test and rows != TEST_ROWS:
        raise ValueError(f"formal pass covered {rows} rows, expected {TEST_ROWS}")

    records = []
    for record, _, run, destination in models:
        path = destination / f"{split}_predictions.parquet"
        (destination / f"{split}_predictions.partial").replace(path)
        error = None
        if args.smoke_test:
            expected = pl.read_parquet(
                ROOT / "artifacts/predictions" / record["run_name"] / "val_predictions.parquet"
            ).filter(pl.col("impression_id") % 1000 == 0).sort("impression_id")
            observed = pl.read_parquet(path).sort("impression_id")
            np.testing.assert_array_equal(
                expected["impression_id"].to_numpy(), observed["impression_id"].to_numpy()
            )
            error = float(
                np.max(np.abs(expected["prediction"].to_numpy() - observed["prediction"].to_numpy()))
            )
            if error > 1e-6:
                raise ValueError(f"{run} does not reproduce its saved validation predictions: {error}")
        records.append(
            {
                "run_name": run, "training_run": record["run_name"],
                "condition": record["condition"], "seed": record["seed"],
                "pooling": record["pooling"], "source": record["source"],
                "zero_sequence": record["zero_sequence"], "rows": rows,
                "prediction_sha256": sha256(path),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "validation_smoke_max_error": error, "inference_passes": 1,
            }
        )

    if sha256(BASE_MODEL) != freeze["base_model_sha256"] or sha256(BASE_SCALER) != freeze["base_scaler_sha256"]:
        raise ValueError("frozen backbone changed during inference")

    write_json(
        study / "manifest.json",
        {
            "complete": True, "split": split, "smoke": args.smoke_test,
            "freeze_sha256": sha256(TEST_FREEZE), "git_head": head,
            "records": records,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "exact_zero_residual_assertions_passed": True,
            "f123_id_label_parity": True, "device": str(device),
            "passes_per_checkpoint": 1,
        },
    )
    if args.smoke_test:
        certificate = STUDY / "f5_inference_smoke_certificate.json"
        if certificate.exists():
            raise FileExistsError("a smoke certificate already exists")
        write_json(
            certificate,
            {
                "passed": True, "freeze_sha256": sha256(TEST_FREEZE),
                "manifest_sha256": sha256(study / "manifest.json"), "split": "val",
                "rows_per_condition": rows, "conditions": len(records),
                "max_error": max(r["validation_smoke_max_error"] for r in records),
            },
        )
    print(f"Completed {split}: {len(records)} checkpoints x {rows:,} rows, one pass each", flush=True)


if __name__ == "__main__":
    main()
