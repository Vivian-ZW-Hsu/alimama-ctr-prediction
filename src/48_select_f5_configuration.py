"""Execute the preregistered F5 validation-only study and freeze checkpoint hashes.

Mirrors src/23_select_sequence_configuration.py. Validation is the only split
touched. The entry point refuses to run once a freeze manifest exists, so no
further tuning is possible after the freeze.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

from project_environment import require_tested_environment

from ctr_nn_common import ROOT, BASE_MODEL, BASE_SCALER, tabular
from f5_sequences import MAX_LENGTH, sha256, write_json
from f5_sequence_ctr import F5SequenceResidual, ZeroF5SequenceResidual

STUDY = ROOT / "results" / "f5_selection_v1"
TRAIN_ROWS = 20_015_245
VAL_ROWS = 3_234_051


def condition_name(pooling: str, source: str, zero: bool) -> str:
    return "behavior_zero_same_pipeline" if zero else f"behavior_{pooling}_{source}"


def run_name(pooling: str, source: str, rate: float, seed: int, zero: bool = False) -> str:
    if zero:
        return f"f5b_zero_l{MAX_LENGTH}_s{seed}_v1"
    return f"f5b_{pooling}_{source}_l{MAX_LENGTH}_lr{int(rate * 1e6)}_s{seed}_v1"


def select_common(rows: list[dict]) -> tuple[float, list[dict]]:
    """Pair the mechanisms; neither pooling family is optimised independently."""
    candidates = []
    for rate in (5e-4, 2e-4):
        pair = [r for r in rows if r["learning_rate"] == rate]
        if len(pair) != 2 or {r["pooling"] for r in pair} != {"mean", "attention"}:
            raise ValueError(f"incomplete search pair at learning rate {rate}")
        candidates.append(
            {
                "learning_rate": rate,
                "paired_validation_logloss": float(np.mean([r["log_loss"] for r in pair])),
            }
        )
    selected = min(
        candidates, key=lambda r: (r["paired_validation_logloss"], -r["learning_rate"])
    )
    return selected["learning_rate"], candidates


def verify_completed(name: str) -> tuple[dict, dict]:
    directory = ROOT / "results" / name
    manifest = json.loads((directory / "manifest.json").read_text())
    if not (manifest["complete"] and not manifest["smoke"] and not manifest["pilot"]):
        raise ValueError(f"{name} is not a completed formal run")
    if manifest["prediction_splits"] != ["val"] or manifest["test_inference"]:
        raise ValueError(f"{name} touched a split other than validation")
    if manifest["train_rows"] != TRAIN_ROWS or manifest["val_rows"] != VAL_ROWS:
        raise ValueError(f"{name} row counts disagree with the protocol")
    if sha256(directory / "config.json") != manifest["config_sha256"]:
        raise ValueError(f"{name} config changed after the run")
    for path, digest in manifest["output_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"{name} artifact {path} changed after the run")
    config = json.loads((directory / "config.json").read_text())
    for path, digest in config["code_sha256"].items():
        if sha256(ROOT / "src" / path) != digest:
            raise ValueError(f"{name} was produced by different code than is present")
    return config, manifest


def command(args: list[str], log_path: Path, remaining: float) -> None:
    if remaining <= 0:
        raise RuntimeError("Preregistered compute budget exhausted")
    with log_path.open("x") as log:
        subprocess.run(
            [sys.executable, "-u", "-B", *args], cwd=ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=True, timeout=remaining,
        )


def main() -> None:
    environment = require_tested_environment("E5 F5 validation selection")
    preregistration_path = STUDY / "preregistration.json"
    pre = json.loads(preregistration_path.read_text())
    if pre["test_authorized"] is not False or pre["max_length"] != MAX_LENGTH:
        raise ValueError("preregistration disagrees with this entry point")
    if (STUDY / "freeze_manifest.json").exists():
        raise FileExistsError("Study already frozen; no additional tuning is allowed")

    logs = STUDY / "logs"
    logs.mkdir(exist_ok=True)
    start = time.perf_counter()
    completed_before = sum(
        json.loads(p.read_text())["elapsed_seconds"]
        for p in (ROOT / "results").glob(f"f5b_*_l{MAX_LENGTH}_*_v1/manifest.json")
    )

    def remaining() -> float:
        return pre["budget_seconds"] - completed_before - (time.perf_counter() - start)

    def ensure(pooling: str, source: str, rate: float, seed: int, zero: bool = False) -> dict:
        name = run_name(pooling, source, rate, seed, zero)
        if not (ROOT / "results" / name / "manifest.json").exists():
            print(f"Training {name}", flush=True)
            arguments = [
                "src/47_train_f5_sequence_ctr.py", "--cache-run", pre["cache_run"],
                "--run-name", name, "--pooling", pooling, "--sequence-source", source,
                "--learning-rate", str(rate), "--seed", str(seed),
                "--epochs", str(pre["max_epochs"]), "--batch-size", str(pre["batch_size"]),
            ]
            if zero:
                arguments.append("--zero-sequence")
            command(arguments, logs / f"{name}_train.log", remaining())
        config, manifest = verify_completed(name)
        for key, expected in (
            ("pooling", pooling), ("sequence_source", source),
            ("learning_rate", rate), ("seed", seed),
            ("cache_run", pre["cache_run"]), ("batch_size", pre["batch_size"]),
            ("zero_sequence", zero), ("max_length", MAX_LENGTH),
        ):
            if config[key] != expected:
                raise ValueError(f"{name} config {key}={config[key]}, expected {expected}")
        metrics = json.loads((ROOT / "results" / name / "best_validation.json").read_text())
        row = {
            "run_name": name, "condition": condition_name(pooling, source, zero),
            "pooling": pooling, "source": source, "zero_sequence": zero,
            "learning_rate": rate, "seed": seed, "best_epoch": manifest["best_epoch"],
            **{k: v for k, v in metrics.items() if k != "model"},
        }
        print(json.dumps(row), flush=True)
        return row

    # Stage 1: paired learning-rate search on matched sources at the selection seed.
    search = [
        ensure(pooling, "matched", rate, pre["selection_seed"])
        for rate in pre["learning_rates"]
        for pooling in pre["poolings"]
    ]
    selected_rate, pairs = select_common(search)
    selection = {
        "preregistration_sha256": sha256(preregistration_path),
        "criterion": "paired validation LogLoss",
        "selected_learning_rate": selected_rate,
        "max_length": MAX_LENGTH,
        "candidates": pairs,
        "runs": search,
    }
    selection_path = STUDY / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("selection evidence changed on resume")
    else:
        write_json(selection_path, selection)
    print(f"Selected common LR={selected_rate}; controls inherit it", flush=True)

    # Stage 2: the four learned conditions at three seeds, plus the zero control.
    final = [
        ensure(pooling, source, selected_rate, seed)
        for seed in pre["final_seeds"]
        for pooling in pre["poolings"]
        for source in pre["sources"]
    ]
    zero_runs = [
        ensure("mean", "matched", selected_rate, seed, zero=True)
        for seed in pre["zero_control"]["seeds"]
    ]

    all_runs = {row["run_name"]: row for row in [*search, *final, *zero_runs]}
    for name, row in all_runs.items():
        evaluation = ROOT / "results" / name / "evaluation"
        if not (evaluation / "evaluation_manifest.json").exists():
            command(
                ["src/10_evaluate_predictions.py", "--run-name", name, "--splits", "val"],
                logs / f"{name}_evaluate.log", remaining(),
            )
        if json.loads((evaluation / "evaluation_manifest.json").read_text())["splits"] != ["val"]:
            raise ValueError(f"{name} evaluation touched a split other than validation")
        overall = pl.read_csv(evaluation / "overall_metrics.csv").row(0, named=True)
        gauc = pl.read_csv(evaluation / "gauc_summary.csv").row(0, named=True)
        if abs(overall["logloss"] - row["log_loss"]) >= 1e-12:
            raise ValueError(f"{name} evaluator LogLoss disagrees with the trainer")
        row.update({k: overall[k] for k in ("copc", "ece")})
        row.update({k: gauc[k] for k in ("gauc", "eligible_users", "eligible_row_coverage")})

    # Reload every frozen checkpoint and regenerate its saved validation probabilities.
    spec = importlib.util.spec_from_file_location(
        "f5_sequence_trainer", ROOT / "src/47_train_f5_sequence_ctr.py"
    )
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    cache_path = ROOT / "results" / pre["cache_run"] / "manifest.json"
    cache = json.loads(cache_path.read_text())
    f5_path = ROOT / cache["f5_manifest"]
    f5 = json.loads(f5_path.read_text())
    if sha256(BASE_MODEL) != cache["base_model_sha256"] or sha256(BASE_SCALER) != cache["scaler_sha256"]:
        raise ValueError("frozen backbone or scaler changed")
    data = trainer.load_columns(ROOT / cache["directory"] / "val.parquet")
    events = pl.read_parquet(
        ROOT / f5["output_directory"] / "events.parquet", columns=["cate_idx", "type_idx"]
    )
    categories = events["cate_idx"].to_numpy()
    types = events["type_idx"].to_numpy()
    del events
    embedding = torch.from_numpy(np.load(ROOT / cache["directory"] / "category_embedding.npy"))
    device = tabular.choose_device("auto")

    frozen = []
    for row in [*final, *zero_runs]:
        name = row["run_name"]
        config, manifest = verify_completed(name)
        checkpoint_path = ROOT / "artifacts/models" / name / "best_model.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["base_model_sha256"] != cache["base_model_sha256"]:
            raise ValueError(f"{name} was trained against a different backbone")
        model_class = ZeroF5SequenceResidual if row["zero_sequence"] else F5SequenceResidual
        model = model_class(embedding, pooling=row["pooling"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.assert_neutral()
        reproduced = trainer.predict(
            model, data, categories, types, row["source"], device, pre["batch_size"] * 2
        )
        saved = pl.read_parquet(ROOT / "artifacts/predictions" / name / "val_predictions.parquet")
        np.testing.assert_array_equal(saved["impression_id"].to_numpy(), data["impression_id"])
        np.testing.assert_array_equal(saved["clk"].to_numpy(), data["clk"])
        np.testing.assert_array_equal(saved["prediction"].to_numpy(), reproduced)
        if (ROOT / "artifacts/predictions" / name / "test_predictions.parquet").exists():
            raise ValueError(f"{name} already has a test prediction; test is not authorized")
        frozen.append(
            {
                "run_name": name, "condition": row["condition"], "seed": row["seed"],
                "pooling": row["pooling"], "source": row["source"],
                "zero_sequence": row["zero_sequence"], "best_epoch": manifest["best_epoch"],
                "checkpoint_sha256": sha256(checkpoint_path),
                "validation_prediction_sha256": manifest["output_sha256"][
                    f"artifacts/predictions/{name}/val_predictions.parquet"
                ],
                "reload_prediction_max_error": 0.0,
            }
        )

    reported = [all_runs[row["run_name"]] for row in [*final, *zero_runs]]
    pl.DataFrame(reported).write_csv(STUDY / "validation_metrics.csv")
    pl.DataFrame(list(all_runs.values())).write_csv(STUDY / "all_candidates.csv")
    summary = (
        pl.DataFrame(reported)
        .group_by("condition")
        .agg(
            pl.len().alias("seeds"),
            *[
                expression
                for metric in ("auc", "log_loss", "pr_auc", "gauc", "copc", "ece")
                for expression in (
                    pl.col(metric).mean().alias(metric + "_mean"),
                    pl.col(metric).std().alias(metric + "_std"),
                )
            ],
        )
        .sort("condition")
    )
    summary.write_csv(STUDY / "validation_seed_summary.csv")

    write_json(
        STUDY / "freeze_manifest.json",
        {
            "complete": True, "budget": "F5",
            "test_authorized": False, "test_inference_performed": False,
            "preregistration_sha256": sha256(preregistration_path),
            "selection_sha256": sha256(selection_path),
            "cache_manifest_sha256": sha256(cache_path),
            "f5_manifest_sha256": sha256(f5_path),
            "base_model_sha256": cache["base_model_sha256"],
            "base_scaler_sha256": cache["scaler_sha256"],
            "selected_learning_rate": selected_rate,
            "max_length": MAX_LENGTH,
            "seed_list": pre["final_seeds"],
            "primary_model": pre["primary_model"],
            "comparison_plan": pre["comparison_plan"],
            "configuration": pre,
            "checkpoints": frozen,
            "unique_training_runs": len(all_runs),
            "code_sha256": {
                path.name: sha256(path)
                for path in [
                    ROOT / "src/f5_sequence_ctr.py", ROOT / "src/f5_sequences.py",
                    ROOT / "src/ctr_nn_common.py", ROOT / "src/47_train_f5_sequence_ctr.py",
                    Path(__file__), ROOT / "src/10_evaluate_predictions.py",
                ]
            },
            "environment": environment,
            "execution_seconds_this_invocation": round(time.perf_counter() - start, 2),
            "next_gate": (
                "User approval required before the first formal F5 test inference. "
                "No further tuning."
            ),
        },
    )
    print(summary, flush=True)
    print("FROZEN. Stopped before formal test.", flush=True)


if __name__ == "__main__":
    main()
