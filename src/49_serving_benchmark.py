"""Unified serving benchmark for every frozen CTR model, on one fixed setup.

Each model is measured through its own real inference path. Two regimes are
reported so neither favours a model type:

* **pure** — inputs are already prepared in memory; only the model call is timed.
* **end_to_end** — a Parquet batch is read, features are encoded and scaled, the
  sequence window is gathered where applicable, and then the model runs.

Every model is benchmarked in its **own subprocess**, so peak RSS belongs to that
model alone and a previously loaded model cannot inflate it. All timing uses CPU
with a fixed thread count, which is the common ground across sklearn, LightGBM
and PyTorch; the neural models were trained on MPS, which is recorded but not used
here. Labels are read only to locate rows and never used for scoring.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from project_environment import require_tested_environment

from f5_sequences import ROOT, sha256, write_json

STUDY = ROOT / "results" / "serving_benchmark_v1"
SPLIT = "val"
MODEL_INPUT = ROOT / f"data/processed/model_input_f3/{SPLIT}.parquet"
ID_SPACE = 26_557_961

# Fixed measurement plan, declared here rather than tuned per model.
SINGLE_WARMUP = 100
SINGLE_REPEATS = 500
BATCH_WARMUP = 3
BATCH_REPEATS = 10
BATCH_SIZES = (1024, 8192)
THREADS = 8


def percentile(values: list[float], fraction: float) -> float:
    return float(np.quantile(np.asarray(values), fraction, method="linear"))


# --------------------------------------------------------------------------
# Adapters: each exposes load(), prepare(record_batch) and infer(prepared)
# --------------------------------------------------------------------------
class LogisticAdapter:
    key = "lr_f123"
    family = "Logistic Regression"
    path = ROOT / "artifacts/models/lr_f1_f2_f3/best_model.joblib"

    def load(self):
        import joblib
        from ctr_posthoc import module

        self.trainer = module("lr_serving", "08_train_logistic_regression.py")
        checkpoint = joblib.load(self.path)
        self.model = checkpoint["model"]
        self.scaler = checkpoint["scaler"]
        self.layout = checkpoint["feature_layout"]
        self.categorical = checkpoint["categorical_features"]
        self.binary = checkpoint["binary_features"]
        self.numeric = checkpoint["numeric_features"]
        self.vocabulary = checkpoint["vocabulary_sizes"]
        self.columns = ["impression_id", "clk", *self.categorical, *self.binary, *self.numeric]
        return {
            "nonzero_coefficients": int(np.count_nonzero(self.model.coef_)),
            "coefficient_dimension": int(self.model.coef_.size),
            "feature_dimension": int(self.layout["total_dimension"]),
        }

    def prepare(self, batch):
        return self.trainer.build_sparse_matrix(
            batch=batch, permutation=None, categorical_features=self.categorical,
            binary_features=self.binary, numeric_features=self.numeric,
            vocabulary_sizes=self.vocabulary, feature_layout=self.layout,
            scaler=self.scaler,
        )

    def infer(self, prepared):
        return self.model.predict_proba(prepared)[:, 1]


class LightGBMAdapter:
    key = "lightgbm_f123"
    family = "LightGBM"
    path = ROOT / "artifacts/models/lightgbm_f1_f2_f3/best_model.txt"

    def load(self):
        import lightgbm as lgb

        self.booster = lgb.Booster(model_file=str(self.path))
        self.features = self.booster.feature_name()
        self.columns = ["impression_id", "clk", *self.features]
        return {
            "num_trees": int(self.booster.num_trees()),
            "num_features": int(self.booster.num_feature()),
            "node_count": None,
            "node_count_note": "omitted; a full model dump of this 363 MiB booster is memory-expensive",
        }

    def prepare(self, batch):
        return np.column_stack(
            [batch.column(name).to_numpy(zero_copy_only=False).astype(np.float64)
             for name in self.features]
        )

    def infer(self, prepared):
        return self.booster.predict(prepared, num_iteration=self.booster.best_iteration)


class TorchTabularAdapter:
    def load(self):
        import torch
        from ctr_nn_common import tabular

        torch.set_num_threads(THREADS)
        self.torch = torch
        self.tabular = tabular
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        self.categorical = checkpoint["categorical_features"]
        self.binary = checkpoint["binary_features"]
        self.numeric = checkpoint["numeric_features"]
        self.model = tabular.TabularCTRModel(
            checkpoint["model_type"],
            [checkpoint["vocabulary_sizes"][k] for k in self.categorical],
            len(self.binary) + len(self.numeric),
            checkpoint["embedding_dim"], checkpoint["hidden_dims"], checkpoint["dropout"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval().requires_grad_(False)
        scaler = json.loads((ROOT / self.scaler_path).read_text())
        self.mean = np.array(scaler["mean"], dtype=np.float32)
        self.scale = np.array(scaler["scale"], dtype=np.float32)
        self.columns = ["impression_id", "clk", *self.categorical, *self.binary, *self.numeric]
        return {
            "trainable_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "embedding_dim": checkpoint["embedding_dim"],
            "hidden_dims": checkpoint["hidden_dims"],
        }

    def prepare(self, batch):
        encoded = self.tabular.categorical_numpy(batch, self.categorical)
        dense = self.tabular.dense_numpy(batch, self.binary, self.numeric, self.mean, self.scale)
        return (self.torch.from_numpy(encoded), self.torch.from_numpy(dense))

    def infer(self, prepared):
        with self.torch.no_grad():
            return self.torch.sigmoid(self.model(*prepared)).numpy()


class WideDeepAdapter(TorchTabularAdapter):
    key = "wide_deep_f123"
    family = "Wide & Deep"
    path = ROOT / "artifacts/models/wide_deep_f1_f2_f3/best_model.pt"
    scaler_path = "results/wide_deep_f1_f2_f3/numeric_scaler.json"


class DeepFMAdapter(TorchTabularAdapter):
    key = "deepfm_f123"
    family = "DeepFM"
    path = ROOT / "artifacts/models/deepfm_f1_f2_f3/best_model.pt"
    scaler_path = "results/deepfm_f1_f2_f3/numeric_scaler.json"


class SequenceAdapter(TorchTabularAdapter):
    """Frozen DeepFM backbone plus a trained sequence residual."""

    path = ROOT / "artifacts/models/deepfm_f1_f2_f3/best_model.pt"
    scaler_path = "results/deepfm_f1_f2_f3/numeric_scaler.json"

    def load(self):
        info = super().load()
        import torch

        residual_path = ROOT / "artifacts/models" / self.residual_run / "best_model.pt"
        checkpoint = torch.load(residual_path, map_location="cpu", weights_only=False)
        embedding = self.model.embeddings[self.categorical.index("cate_idx")].weight.detach()
        self.residual = self.build_residual(embedding)
        self.residual.load_state_dict(checkpoint["model_state_dict"])
        self.residual.eval().requires_grad_(False)
        self.category_position = self.categorical.index("cate_idx")

        # Pointer tables, indexed by impression_id exactly as the cache builds them.
        self.starts = np.zeros(ID_SPACE, dtype=np.uint32)
        self.lengths = np.zeros(ID_SPACE, dtype=np.uint8)
        for path in sorted((ROOT / self.reference_directory).glob(f"{SPLIT}_*.parquet")):
            refs = pl.read_parquet(path, columns=["impression_id", "start", "length"])
            ids = refs["impression_id"].to_numpy()
            self.starts[ids] = refs["start"].to_numpy()
            self.lengths[ids] = refs["length"].to_numpy()
        events = pl.read_parquet(ROOT / self.events_path, columns=self.event_columns)
        self.event_arrays = [events[name].to_numpy() for name in self.event_columns]
        del events
        info.update(
            {
                "residual_run": self.residual_run,
                "residual_parameters": int(sum(p.numel() for p in self.residual.parameters())),
                "residual_model_bytes": residual_path.stat().st_size,
                "backbone_shared_with": "deepfm_f123",
                "max_length": self.max_length,
            }
        )
        return info

    def prepare(self, batch):
        encoded = self.tabular.categorical_numpy(batch, self.categorical)
        dense = self.tabular.dense_numpy(batch, self.binary, self.numeric, self.mean, self.scale)
        ids = batch.column("impression_id").to_numpy(zero_copy_only=False).astype(np.int64)
        tokens = self.gather(self.starts[ids], self.lengths[ids])
        candidate = self.torch.from_numpy(encoded[:, self.category_position].astype(np.int64))
        return (
            self.torch.from_numpy(encoded), self.torch.from_numpy(dense), candidate,
            *[self.torch.from_numpy(t) for t in tokens],
        )

    def infer(self, prepared):
        encoded, dense, candidate, *tokens = prepared
        with self.torch.no_grad():
            base = self.model(encoded, dense)
            delta = self.residual(candidate, *tokens)
            return self.torch.sigmoid(base + delta).numpy()


class F4SequenceAdapter(SequenceAdapter):
    key = "f4_mean_matched"
    family = "F4 clicked-ad mean matched"
    residual_run = "seq_mean_matched_l20_lr500_s42_v1"
    reference_directory = "data/processed/f4_click_daily_v1"
    events_path = "data/processed/f4_click_daily_v1/events.parquet"
    event_columns = ["cate_idx"]
    max_length = 20

    def build_residual(self, embedding):
        from sequence_ctr import SequenceResidual

        return SequenceResidual(embedding.clone(), pooling="mean")

    def gather(self, starts, lengths):
        from sequence_ctr import gather_tokens

        return (gather_tokens(self.event_arrays[0], starts, lengths, self.max_length),)


class F5SequenceAdapter(SequenceAdapter):
    key = "f5_mean_matched"
    family = "F5 behavior mean matched"
    reference_directory = "data/processed/f5_behavior_v1"
    events_path = "data/processed/f5_behavior_v1/events.parquet"
    event_columns = ["cate_idx", "type_idx"]
    max_length = 200

    def build_residual(self, embedding):
        from f5_sequence_ctr import F5SequenceResidual
        from f5_sequences import TYPE_VOCABULARY_SIZE

        return F5SequenceResidual(
            embedding.clone(), pooling="mean", type_vocabulary_size=TYPE_VOCABULARY_SIZE
        )

    def gather(self, starts, lengths):
        from f5_sequence_ctr import gather_f5_tokens

        return gather_f5_tokens(*self.event_arrays, starts, lengths, self.max_length)


ADAPTERS = {
    cls.key: cls
    for cls in (LogisticAdapter, LightGBMAdapter, WideDeepAdapter, DeepFMAdapter,
                F4SequenceAdapter, F5SequenceAdapter)
}


def read_batches(columns: list[str], batch_size: int, count: int):
    """Deterministic leading batches of the validation model input."""
    reader = pq.ParquetFile(MODEL_INPUT)
    collected = []
    for batch in reader.iter_batches(batch_size=batch_size, columns=columns):
        collected.append(batch)
        if len(collected) >= count:
            break
    return collected


def benchmark(key: str, residual_run: str | None) -> dict:
    adapter = ADAPTERS[key]()
    if residual_run and isinstance(adapter, SequenceAdapter):
        adapter.residual_run = residual_run
    load_started = time.perf_counter()
    details = adapter.load()
    load_seconds = time.perf_counter() - load_started

    result = {
        "model": key,
        "family": adapter.family,
        "model_file": str(adapter.path.relative_to(ROOT)),
        "model_file_bytes": adapter.path.stat().st_size,
        "model_load_seconds": round(load_seconds, 4),
        "details": details,
        "regimes": {},
    }

    # Single-row latency. Both regimes use the same one-row source batch.
    single_source = read_batches(adapter.columns, 1, SINGLE_WARMUP + SINGLE_REPEATS)
    prepared_single = [adapter.prepare(b) for b in single_source[:SINGLE_REPEATS]]
    for _ in range(SINGLE_WARMUP):
        adapter.infer(prepared_single[0])
    pure = []
    for prepared in prepared_single:
        started = time.perf_counter()
        adapter.infer(prepared)
        pure.append((time.perf_counter() - started) * 1000.0)
    end_to_end = []
    for batch in single_source[:SINGLE_REPEATS]:
        started = time.perf_counter()
        adapter.infer(adapter.prepare(batch))
        end_to_end.append((time.perf_counter() - started) * 1000.0)
    result["regimes"]["batch_1_latency_ms"] = {
        regime: {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "mean": float(np.mean(values)),
            "repeats": len(values),
            "warmup": SINGLE_WARMUP,
        }
        for regime, values in (("pure", pure), ("end_to_end", end_to_end))
    }

    # Batched throughput.
    for batch_size in BATCH_SIZES:
        source = read_batches(adapter.columns, batch_size, BATCH_WARMUP + BATCH_REPEATS)
        usable = [b for b in source if b.num_rows == batch_size]
        if len(usable) < BATCH_WARMUP + BATCH_REPEATS:
            usable = (usable * (BATCH_WARMUP + BATCH_REPEATS))[: BATCH_WARMUP + BATCH_REPEATS]
        prepared = [adapter.prepare(b) for b in usable]
        for index in range(BATCH_WARMUP):
            adapter.infer(prepared[index])
        started = time.perf_counter()
        rows = 0
        for index in range(BATCH_WARMUP, len(prepared)):
            adapter.infer(prepared[index])
            rows += usable[index].num_rows
        pure_seconds = time.perf_counter() - started
        started = time.perf_counter()
        end_rows = 0
        for index in range(BATCH_WARMUP, len(usable)):
            adapter.infer(adapter.prepare(usable[index]))
            end_rows += usable[index].num_rows
        end_seconds = time.perf_counter() - started
        result["regimes"][f"batch_{batch_size}_throughput_rows_per_second"] = {
            "pure": rows / pure_seconds,
            "end_to_end": end_rows / end_seconds,
            "batches": len(prepared) - BATCH_WARMUP,
            "warmup_batches": BATCH_WARMUP,
        }

    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def environment() -> dict:
    import torch
    import sklearn
    import lightgbm

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": len(__import__("os").sched_getaffinity(0)) if hasattr(__import__("os"), "sched_getaffinity") else __import__("os").cpu_count(),
        "declared_threads": THREADS,
        "device": "cpu for all timings; the neural models were trained on MPS",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def main() -> None:
    environment = require_tested_environment("E6 serving benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=sorted(ADAPTERS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--f5-residual-run", help="F5 primary residual run name")
    parser.add_argument("--models", nargs="*", default=sorted(ADAPTERS))
    args = parser.parse_args()

    if args.worker:
        if args.output is None:
            parser.error("--output is required in worker mode")
        result = benchmark(args.worker, args.f5_residual_run)
        write_json(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "regimes"}), flush=True)
        return

    STUDY.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in args.models:
        output = STUDY / f"{key}.json"
        if not output.exists():
            print(f"Benchmarking {key} in an isolated subprocess...", flush=True)
            arguments = [sys.executable, "-u", "-B", str(Path(__file__)),
                         "--worker", key, "--output", str(output)]
            if args.f5_residual_run:
                arguments += ["--f5-residual-run", args.f5_residual_run]
            subprocess.run(arguments, cwd=ROOT, check=True)
        record = json.loads(output.read_text())
        single = record["regimes"]["batch_1_latency_ms"]
        rows.append(
            {
                "model": record["model"],
                "family": record["family"],
                "model_file_mib": record["model_file_bytes"] / 1024 ** 2,
                "peak_rss_mib": record["peak_rss_bytes"] / 1024 ** 2,
                "load_seconds": record["model_load_seconds"],
                "pure_p50_ms": single["pure"]["p50"],
                "pure_p95_ms": single["pure"]["p95"],
                "pure_p99_ms": single["pure"]["p99"],
                "end_to_end_p50_ms": single["end_to_end"]["p50"],
                "end_to_end_p95_ms": single["end_to_end"]["p95"],
                "end_to_end_p99_ms": single["end_to_end"]["p99"],
                **{
                    f"{regime}_rps_batch_{size}": record["regimes"][
                        f"batch_{size}_throughput_rows_per_second"
                    ][regime]
                    for size in BATCH_SIZES
                    for regime in ("pure", "end_to_end")
                },
            }
        )
    table = pl.DataFrame(rows)
    table.write_csv(STUDY / "serving_benchmark.csv")
    write_json(
        STUDY / "manifest.json",
        {
            "complete": True,
            "split_used": SPLIT,
            "labels_used_for_scoring": False,
            "regimes": {
                "pure": "inputs already prepared in memory; only the model call is timed",
                "end_to_end": "Parquet batch read, encoding, scaling, sequence gather, model call",
            },
            "isolation": "one subprocess per model, so peak RSS is not contaminated",
            "measurement_plan": {
                "single_row_warmup": SINGLE_WARMUP, "single_row_repeats": SINGLE_REPEATS,
                "batch_warmup": BATCH_WARMUP, "batch_repeats": BATCH_REPEATS,
                "batch_sizes": list(BATCH_SIZES),
            },
            "environment": environment(),
            "models": args.models,
            "per_model_sha256": {
                f"{key}.json": sha256(STUDY / f"{key}.json") for key in args.models
            },
            "caveat": (
                "Sequence rows share one frozen DeepFM backbone with deepfm_f123, so "
                "their cost includes that backbone plus a residual, and their model "
                "file size is reported for the backbone with the residual size listed "
                "separately in details."
            ),
        },
    )
    print(table, flush=True)


if __name__ == "__main__":
    main()
