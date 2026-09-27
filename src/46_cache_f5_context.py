"""Cache frozen DeepFM logits and compact F5 references for TRAIN/VAL ONLY.

Mirrors src/21_cache_sequence_context.py. The frozen backbone, its scaler and the
existing saved validation predictions are read only; test is never opened.
"""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular, load_frozen
from f5_sequences import MAX_LENGTH, sha256, write_json

ID_SPACE = 26_557_961


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f5-run", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not args.run_name.replace("_", "").isalnum():
        parser.error("Invalid run name")

    output = ROOT / "data/processed" / args.run_name
    result = ROOT / "results" / args.run_name
    if output.exists() or result.exists():
        raise FileExistsError("Cache names are immutable; choose a new run")

    f5_path = ROOT / "results" / args.f5_run / "manifest.json"
    f5 = json.loads(f5_path.read_text())
    if not f5["complete"]:
        raise ValueError("F5 build is not complete")
    if f5["config"]["max_length"] != MAX_LENGTH:
        raise ValueError(f"expected max_length {MAX_LENGTH}, found {f5['config']['max_length']}")
    directory = ROOT / f5["output_directory"]
    for name, digest in f5["output_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"{name} changed since the F5 build completed")

    device = tabular.choose_device("auto")
    model, checkpoint, mean, scale = load_frozen(device)
    categorical, binary, numeric = [
        checkpoint[key] for key in ("categorical_features", "binary_features", "numeric_features")
    ]
    if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
        raise ValueError("backbone must be frozen and in eval mode")
    category_position = categorical.index("cate_idx")
    baseline_digest = sha256(BASE_MODEL)
    scaler_digest = sha256(BASE_SCALER)

    output.mkdir(parents=True)
    result.mkdir(parents=True)
    np.save(
        output / "category_embedding.npy",
        model.embeddings[category_position].weight.detach().cpu().numpy(),
    )

    started = time.perf_counter()
    records = []
    source_columns = ["impression_id", "clk", *categorical, *binary, *numeric]
    for split in ("train", "val"):
        present = np.zeros(ID_SPACE, dtype=bool)
        starts = np.zeros(ID_SPACE, dtype=np.uint32)
        donor = np.zeros(ID_SPACE, dtype=np.uint32)
        lengths = np.zeros(ID_SPACE, dtype=np.uint8)
        labels = np.zeros(ID_SPACE, dtype=np.uint8)
        for path in sorted(directory.glob(f"{split}_*.parquet")):
            refs = pl.read_parquet(path)
            ids = refs["impression_id"].to_numpy()
            if present[ids].any():
                raise ValueError(f"{path} repeats an impression_id")
            present[ids] = True
            starts[ids] = refs["start"].to_numpy()
            donor[ids] = refs["donor_start"].to_numpy()
            lengths[ids] = refs["length"].to_numpy()
            labels[ids] = refs["clk"].to_numpy()
        expected = int(present.sum())

        previous = None
        if split == "val":
            previous = np.full(ID_SPACE, np.nan, dtype=np.float32)
            saved = pl.read_parquet(
                ROOT / "artifacts/predictions/deepfm_f1_f2_f3/val_predictions.parquet"
            )
            previous[saved["impression_id"].to_numpy()] = saved["prediction"].to_numpy()

        writer = None
        rows = 0
        max_error = 0.0
        temporary = output / f"{split}.partial"
        seen = np.zeros(ID_SPACE, dtype=bool)
        try:
            for batch in tabular.iter_record_batches(
                ROOT / f"data/processed/model_input_f3/{split}.parquet",
                source_columns, 8192, None,
            ):
                ids = tabular.arrow_column_to_numpy(batch, "impression_id", np.uint32)
                if f5["config"]["smoke"]:
                    batch = batch.filter(pa.array(present[ids]))
                    if batch.num_rows == 0:
                        continue
                    ids = tabular.arrow_column_to_numpy(batch, "impression_id", np.uint32)
                if not present[ids].all() or seen[ids].any():
                    raise ValueError("cache row set disagrees with the F5 references")
                seen[ids] = True
                y = tabular.arrow_column_to_numpy(batch, "clk", np.uint8)
                np.testing.assert_array_equal(y, labels[ids])
                encoded = tabular.categorical_numpy(batch, categorical)
                dense = tabular.dense_numpy(batch, binary, numeric, mean, scale)
                with torch.no_grad():
                    logits = model(
                        torch.from_numpy(encoded).to(device), torch.from_numpy(dense).to(device)
                    )
                    probabilities = torch.sigmoid(logits).cpu().numpy()
                    logits = logits.cpu().numpy()
                if previous is not None:
                    if not np.isfinite(previous[ids]).all():
                        raise ValueError("missing saved validation prediction")
                    error = float(np.max(np.abs(probabilities - previous[ids])))
                    max_error = max(max_error, error)
                    if error > 1e-6:
                        raise ValueError(f"frozen baseline prediction mismatch: {error}")
                table = pa.table(
                    {
                        "impression_id": ids,
                        "clk": y,
                        "candidate_category": encoded[:, category_position].astype(np.uint32),
                        "start": starts[ids],
                        "donor_start": donor[ids],
                        "length": lengths[ids],
                        "base_logit": logits.astype(np.float32),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                rows += len(ids)
        finally:
            if writer:
                writer.close()
        if rows != expected or not np.array_equal(seen, present):
            raise ValueError(f"{split}: cached {rows} rows, expected {expected}")
        temporary.replace(output / f"{split}.parquet")
        records.append(
            {
                "split": split,
                "rows": rows,
                "baseline_max_probability_error": max_error,
                "id_label_parity": True,
                "sha256": sha256(output / f"{split}.parquet"),
            }
        )
        print(records[-1], flush=True)

    if sha256(BASE_MODEL) != baseline_digest or sha256(BASE_SCALER) != scaler_digest:
        raise ValueError("frozen backbone changed during caching")
    write_json(
        result / "manifest.json",
        {
            "complete": True,
            "run_name": args.run_name,
            "smoke": f5["config"]["smoke"],
            "splits": ["train", "val"],
            "test_read": False,
            "budget": "F5",
            "max_length": MAX_LENGTH,
            "f5_manifest": str(f5_path.relative_to(ROOT)),
            "f5_manifest_sha256": sha256(f5_path),
            "base_model_sha256": baseline_digest,
            "scaler_sha256": scaler_digest,
            "source_code_sha256": sha256(ROOT / "src/16_train_tabular_nn.py"),
            "category_embedding_sha256": sha256(output / "category_embedding.npy"),
            "device": str(device),
            "records": records,
            "directory": str(output.relative_to(ROOT)),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    print(f'Complete: {result / "manifest.json"}', flush=True)


if __name__ == "__main__":
    main()
