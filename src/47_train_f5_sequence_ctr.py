"""Train the F5 behavior-sequence residual on frozen DeepFM context; never opens test.

Mirrors src/22_train_sequence_ctr.py. The only intended differences are the
history source (behavior_log events), the retained length (200), and the extra
behavior-type token per position.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import resource
import subprocess
import time

import numpy as np
import polars as pl
import torch
from torch import nn

from ctr_nn_common import ROOT, tabular
from f5_sequences import MAX_LENGTH, TYPE_VOCABULARY_SIZE, sha256, write_json
from f5_sequence_ctr import F5SequenceResidual, ZeroF5SequenceResidual, gather_f5_tokens

RSS_BUDGET_BYTES = 30 * 1024 ** 3


def load_columns(path):
    frame = pl.read_parquet(path)
    return {name: frame[name].to_numpy() for name in frame.columns}


def tensors(data, index, categories, types, source, device):
    key = "start" if source == "matched" else "donor_start"
    category_tokens, type_tokens = gather_f5_tokens(
        categories, types, data[key][index], data["length"][index], MAX_LENGTH
    )
    return (
        torch.from_numpy(data["candidate_category"][index].astype(np.int64)).to(device),
        torch.from_numpy(category_tokens).to(device),
        torch.from_numpy(type_tokens).to(device),
        torch.from_numpy(data["base_logit"][index].copy()).to(device),
        torch.from_numpy(data["clk"][index].astype(np.float32)).to(device),
    )


def predict(model, data, categories, types, source, device, batch_size):
    model.eval()
    output = np.empty(len(data["clk"]), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            index = slice(start, start + batch_size)
            candidate, history, history_type, base, _ = tensors(
                data, index, categories, types, source, device
            )
            delta = model(candidate, history, history_type)
            empty = ~(history > 1).any(dim=1)
            if torch.count_nonzero(delta[empty]).item():
                raise ValueError("empty history produced a nonzero residual")
            output[index] = torch.sigmoid(base + delta).cpu().numpy()
    if not np.isfinite(output).all():
        raise ValueError("non-finite prediction")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-run", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--pooling", choices=["mean", "attention"], required=True)
    parser.add_argument("--sequence-source", choices=["matched", "mismatched"], required=True)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--pilot", action="store_true",
                        help="One full train epoch; no validation scoring")
    parser.add_argument("--zero-sequence", action="store_true",
                        help="Hard-zero the whole residual through the same loop")
    args = parser.parse_args()
    if not args.run_name.replace("_", "").isalnum():
        parser.error("Invalid run name")
    if args.learning_rate <= 0 or args.batch_size <= 0 or args.epochs < 1:
        parser.error("Training limits must be positive")

    result = ROOT / "results" / args.run_name
    model_dir = ROOT / "artifacts/models" / args.run_name
    prediction_dir = ROOT / "artifacts/predictions" / args.run_name
    if any(path.exists() for path in (result, model_dir, prediction_dir)):
        raise FileExistsError("Runs are immutable; choose a new name")

    cache_path = ROOT / "results" / args.cache_run / "manifest.json"
    cache = json.loads(cache_path.read_text())
    if not (cache["complete"] and cache["splits"] == ["train", "val"]):
        raise ValueError("cache is incomplete")
    if cache["smoke"] != args.smoke_test:
        raise ValueError("cache smoke flag disagrees with this run")
    if cache.get("max_length") != MAX_LENGTH:
        raise ValueError("cache was not built for the F5 retained length")
    directory = ROOT / cache["directory"]
    for record in cache["records"]:
        if sha256(directory / f"{record['split']}.parquet") != record["sha256"]:
            raise ValueError(f"cache {record['split']} changed since it was written")
    if sha256(directory / "category_embedding.npy") != cache["category_embedding_sha256"]:
        raise ValueError("frozen category embedding changed")

    f5_path = ROOT / cache["f5_manifest"]
    if sha256(f5_path) != cache["f5_manifest_sha256"]:
        raise ValueError("F5 manifest changed since caching")
    f5 = json.loads(f5_path.read_text())
    events_path = ROOT / f5["output_directory"] / "events.parquet"
    if sha256(events_path) != f5["output_sha256"][str(events_path.relative_to(ROOT))]:
        raise ValueError("F5 event store changed since the build")

    events = pl.read_parquet(events_path, columns=["cate_idx", "type_idx"])
    categories = events["cate_idx"].to_numpy()
    types = events["type_idx"].to_numpy()
    del events
    train = load_columns(directory / "train.parquet")
    validation = None if args.pilot else load_columns(directory / "val.parquet")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = tabular.choose_device(args.device)
    model_class = ZeroF5SequenceResidual if args.zero_sequence else F5SequenceResidual
    model = model_class(
        torch.from_numpy(np.load(directory / "category_embedding.npy")),
        pooling=args.pooling,
        type_vocabulary_size=TYPE_VOCABULARY_SIZE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    loss_function = nn.BCEWithLogitsLoss()

    result.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    config = {
        **vars(args), "budget": "F5", "max_length": MAX_LENGTH, "embedding_dim": 8,
        "hidden_dims": [64, 32], "attention_hidden_dim": 32, "dropout": 0.3,
        "weight_decay": 1e-5, "optimizer": "AdamW", "patience": 2,
        "context": "frozen_DeepFM_F123",
        "type_vocabulary_size": TYPE_VOCABULARY_SIZE,
        "type_combination": "additive, zero-initialised type embedding",
        "validity_mask": "category > 1, identical to F4; UNKNOWN categories carry no information",
        "cache_manifest_sha256": sha256(cache_path),
        "device_resolved": str(device),
        "code_sha256": {
            name: sha256(ROOT / "src" / name)
            for name in ("47_train_f5_sequence_ctr.py", "f5_sequence_ctr.py",
                         "sequence_ctr.py", "ctr_nn_common.py", "f5_sequences.py")
        },
        "control_purpose": (
            "Isolate residual seed, batch, interface dimensions and training pipeline; "
            "the frozen backbone is shared by all conditions and is not refitted"
            if args.zero_sequence else None
        ),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "versions": {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "polars": pl.__version__,
        },
    }
    write_json(result / "config.json", config)

    epochs = 1 if args.smoke_test or args.pilot else args.epochs
    best = float("inf")
    best_epoch = None
    stale = 0
    history_rows = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        # O(N) permutation only; histories are expanded one minibatch at a time.
        order = np.random.default_rng(args.seed + epoch).permutation(len(train["clk"]))
        loss_sum = torch.zeros((), device=device)
        for offset in range(0, len(order), args.batch_size):
            index = order[offset:offset + args.batch_size]
            candidate, tokens, token_types, base, labels = tensors(
                train, index, categories, types, args.sequence_source, device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(base + model(candidate, tokens, token_types), labels)
            loss.backward()
            if args.zero_sequence:
                for parameter in model.parameters():
                    if parameter.grad is not None and torch.count_nonzero(parameter.grad).item():
                        raise ValueError("zero control produced a nonzero gradient")
            optimizer.step()
            loss_sum += loss.detach() * len(index)
            if offset and offset // args.batch_size % 1000 == 0:
                print(f"epoch={epoch} train_rows={offset:,}", flush=True)
        train_seconds = time.perf_counter() - epoch_start
        train_loss = float(loss_sum.cpu()) / len(order)
        if not np.isfinite(train_loss):
            raise ValueError("non-finite training loss")
        model.assert_neutral()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss >= RSS_BUDGET_BYTES:
            raise ValueError(f"RSS exceeds the declared budget: {rss}")
        row = {
            "epoch": epoch, "train_rows": len(order), "train_logloss": train_loss,
            "train_seconds": train_seconds, "peak_rss_bytes_macos": rss,
        }
        if not args.pilot:
            predictions = predict(
                model, validation, categories, types, args.sequence_source,
                device, args.batch_size * 2,
            )
            metrics = tabular.metric_row(args.run_name, "val", validation["clk"], predictions)
            row.update({f"val_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            if metrics["log_loss"] < best:
                best = metrics["log_loss"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "config": config,
                     "epoch": epoch, "base_model_sha256": cache["base_model_sha256"]},
                    model_dir / "best_model.pt",
                )
                tabular.save_predictions(
                    prediction_dir / "val_predictions.parquet",
                    validation["impression_id"], validation["clk"], predictions,
                )
                write_json(result / "best_validation.json", metrics)
            else:
                stale += 1
        row["epoch_seconds"] = time.perf_counter() - epoch_start
        history_rows.append(row)
        pl.DataFrame(history_rows).write_csv(result / "training_history.csv")
        print(json.dumps(row), flush=True)
        if stale >= 2:
            break

    outputs = {}
    if not args.pilot:
        for path in (model_dir / "best_model.pt", prediction_dir / "val_predictions.parquet"):
            outputs[str(path.relative_to(ROOT))] = sha256(path)
    write_json(
        result / "manifest.json",
        {
            "complete": True, "run_name": args.run_name, "budget": "F5",
            "smoke": args.smoke_test, "pilot": args.pilot,
            "prediction_splits": [] if args.pilot else ["val"],
            "test_inference": False,
            "train_rows": len(train["clk"]),
            "val_rows": 0 if validation is None else len(validation["clk"]),
            "best_epoch": best_epoch,
            "best_validation_logloss": None if args.pilot else best,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "mps_driver_allocated_bytes": (
                torch.mps.driver_allocated_memory() if device.type == "mps" else None
            ),
            "neutral_reserved_rows": True, "empty_residual_exact_zero": True,
            "zero_sequence_same_pipeline": args.zero_sequence,
            "shuffle": "global train row permutation per epoch, seed+epoch",
            "config_sha256": sha256(result / "config.json"),
            "output_sha256": outputs,
        },
    )
    print(f'Complete: {result / "manifest.json"}', flush=True)


if __name__ == "__main__":
    main()
