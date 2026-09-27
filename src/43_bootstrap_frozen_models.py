"""Stage E3: paired user-cluster bootstrap over the frozen tabular predictions.

Runs the existing exact bootstrap kernel (src/cluster_bootstrap_metrics.cpp) on
the inputs bound by src/42_prepare_frozen_model_bootstrap.py. Refuses to start
unless the pre-declared protocol is already committed, so the hypothesis set
cannot be chosen after seeing an interval.

The reported intervals are conditional on the four frozen fitted models and the
single frozen test day. They describe user-sampling uncertainty. They are NOT an
estimate of training-seed uncertainty: each tabular family has exactly one seed,
so re-training variability is not represented at all.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import polars as pl

from project_environment import require_tested_environment

from ctr_posthoc import ROOT, holm, sha256, write_json

STUDY = ROOT / "results" / "frozen_model_bootstrap_v1"
DATA = ROOT / "data" / "processed" / "frozen_model_bootstrap_v1"
METRICS = ("auc", "logloss", "pr_auc", "gauc")

# Saved point metrics the unweighted replicate must reproduce. These are the
# frozen documented values, not values recomputed by this script.
DOCUMENTED = {
    "lr_f123": {"auc": 0.652489, "logloss": 0.192429},
    "lightgbm_f123": {"auc": 0.651318, "logloss": 0.192502},
    "wide_deep_f123": {"auc": 0.646219, "logloss": 0.193161},
    "deepfm_f123": {"auc": 0.645933, "logloss": 0.193176},
}


def validate_replicates(array: np.ndarray, label: str) -> None:
    """Reject a non-finite replicate instead of silently discarding it.

    A bootstrap draw can select only single-class users, in which case the kernel
    emits NaN for AUC. Dropping such draws would bias the interval, so the run
    fails loudly and the situation is inspected.
    """
    if not np.isfinite(array).all():
        raise ValueError(
            f"{label} contains non-finite metrics; a single-class draw must be "
            "inspected, not silently discarded"
        )


def paired_intervals(
    full: np.ndarray,
    index_by_key: dict[str, int],
    comparisons: list[dict],
    replicates: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Percentile intervals and centered p-values for each declared contrast.

    Row 0 of `full` is the unweighted replicate, so the point effect comes from
    it and the interval from rows 1..B. The difference is always left minus
    right and metric orientation is never flipped here, so a negative LogLoss
    effect means the left model has the lower loss.
    """
    rows: list[dict] = []
    draws: dict[str, np.ndarray] = {"replicate": np.arange(replicates, dtype=np.uint32)}
    for comparison in comparisons:
        left = full[:, index_by_key[comparison["left"]], :]
        right = full[:, index_by_key[comparison["right"]], :]
        delta = left - right
        for metric in comparison["metrics"]:
            position = METRICS.index(metric)
            observed = float(delta[0, position])
            boot = delta[1:, position]
            low, high = np.quantile(boot, [0.025, 0.975], method="linear")
            p = (1 + np.sum(np.abs(boot - observed) >= abs(observed))) / (replicates + 1)
            rows.append(
                {
                    "id": comparison["id"],
                    "left": comparison["left"],
                    "right": comparison["right"],
                    "interpretation": comparison["interpretation"],
                    "metric": metric,
                    "effect": observed,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_std": float(np.std(boot, ddof=1)),
                    "p_centered_two_sided": float(p),
                    "holm_primary": None,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
            draws[f"comparison_{comparison['id']}_{metric}"] = boot
    return rows, draws


def apply_holm(rows: list[dict]) -> list[dict]:
    """Holm-correct the confirmatory primary rows only.

    Exploratory contrasts keep a null correction so they cannot be read as
    family-wise controlled evidence.
    """
    positions = [
        position for position, row in enumerate(rows)
        if row["interpretation"] == "confirmatory_primary"
    ]
    if positions:
        corrected = holm([rows[position]["p_centered_two_sided"] for position in positions])
        for position, value in zip(positions, corrected):
            rows[position]["holm_primary"] = float(value)
    return rows


def require_committed(path: Path) -> None:
    """Refuse to run unless the protocol is already recorded in Git history."""
    relative = str(path.relative_to(ROOT))
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError(f"{relative} is not tracked; commit the protocol first")
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
    if dirty.returncode != 0:
        raise RuntimeError(f"{relative} has uncommitted changes; commit it first")


def main() -> None:
    environment = require_tested_environment("E3 frozen bootstrap")
    if (STUDY / "bootstrap_manifest.json").exists():
        raise FileExistsError("Stage E3 bootstrap already complete")
    protocol_path = STUDY / "protocol.json"
    manifest_path = STUDY / "input_manifest.json"
    require_committed(protocol_path)
    require_committed(manifest_path)

    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    settings = protocol["bootstrap"]

    # Inputs must be byte-identical to what the committed protocol bound.
    for name, digest in manifest["derived_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"{name} changed since preparation")
    for name, digest in manifest["input_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"frozen input {name} changed since preparation")

    executable = DATA / "cluster_bootstrap_metrics"
    if executable.exists():
        raise FileExistsError("Existing executable; inspect before resuming")
    subprocess.run(
        ["clang++", "-O3", "-std=c++17", "-pthread",
         str(ROOT / "src" / "cluster_bootstrap_metrics.cpp"), "-o", str(executable)],
        check=True,
    )

    index_by_key = {record["key"]: record["index"] for record in manifest["models"]}
    n_models = len(manifest["models"])
    replicates = settings["replicates"]
    seed = settings["seed"]
    threads = settings["threads"]

    def run(output: Path, count: int) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        subprocess.run(
            [str(executable), str(DATA), str(output), str(count), str(seed), str(threads)],
            check=True,
        )
        elapsed = time.perf_counter() - started
        raw = np.fromfile(output, dtype="<f8").reshape(-1, n_models, len(METRICS))
        return raw, elapsed

    pilot, pilot_seconds = run(DATA / "bootstrap_pilot.bin", settings["pilot_replicates"])
    projected = pilot_seconds * replicates / settings["pilot_replicates"]
    print(f"Pilot {pilot_seconds:.2f}s; projected full <= {projected:.1f}s", flush=True)
    if projected >= 6 * 3600:
        raise RuntimeError("Projected bootstrap exceeds six hours; ask before more compute")
    validate_replicates(pilot, "pilot")

    # Replicate 0 carries unit weights, so it must reproduce the saved metrics.
    for record in manifest["models"]:
        observed = pilot[0, record["index"]]
        expected = DOCUMENTED[record["key"]]
        for position, metric in enumerate(METRICS):
            if metric in expected:
                np.testing.assert_allclose(
                    observed[position], expected[metric], rtol=0, atol=5e-7,
                    err_msg=f"{record['key']} {metric} does not reproduce the documented value",
                )
        np.testing.assert_allclose(
            observed[METRICS.index("gauc")], record["saved_gauc"], rtol=0, atol=1e-9
        )
    write_json(
        STUDY / "bootstrap_pilot.json",
        {
            "passed": True,
            "replicates": settings["pilot_replicates"],
            "seconds": round(pilot_seconds, 3),
            "projected_seconds": round(projected, 1),
            "unweighted_replicate_reproduces_documented_point_metrics": True,
        },
    )

    full, elapsed = run(DATA / "bootstrap_full.bin", replicates)
    np.testing.assert_array_equal(full[: len(pilot)], pilot)
    validate_replicates(full, "full bootstrap")

    rows, draws = paired_intervals(full, index_by_key, protocol["comparisons"], replicates)
    rows = apply_holm(rows)

    pl.DataFrame(rows).write_csv(STUDY / "bootstrap_intervals.csv")
    pl.DataFrame(draws).write_csv(STUDY / "bootstrap_deltas.csv")

    write_json(
        STUDY / "bootstrap_manifest.json",
        {
            "complete": True,
            "valid_replicates": replicates,
            "invalid_replicates": 0,
            "seed": seed,
            "threads": threads,
            "rows": manifest["rows"],
            "users": manifest["users"],
            "user_draws_per_replicate": manifest["users"],
            "distinct_predictors": n_models,
            "training_seeds_per_family": 1,
            "elapsed_seconds": round(elapsed, 2),
            "pilot_reproduces_full_prefix": True,
            "unweighted_replicate_reproduces_documented_point_metrics": True,
            "intervals": settings["interval"],
            "p_resolution": 1 / (replicates + 1),
            "holm_scope": settings["holm_scope"],
            "uncertainty": settings["conditionality"],
            "protocol_sha256": sha256(protocol_path),
            "input_manifest_sha256": sha256(manifest_path),
            "binary_sha256": sha256(executable),
            "raw_results_sha256": sha256(DATA / "bootstrap_full.bin"),
            "code_sha256": {
                name: sha256(ROOT / "src" / name)
                for name in (
                    "cluster_bootstrap_metrics.cpp",
                    "42_prepare_frozen_model_bootstrap.py",
                    "43_bootstrap_frozen_models.py",
                    "ctr_posthoc.py",
                )
            },
            "compiler": subprocess.check_output(["clang++", "--version"], text=True),
            "environment": environment,
            "flags": ["-O3", "-std=c++17", "-pthread"],
        },
    )
    print(
        pl.DataFrame(rows).select(
            "id", "left", "right", "metric", "effect", "ci_low", "ci_high",
            "p_centered_two_sided", "holm_primary", "excludes_zero",
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
