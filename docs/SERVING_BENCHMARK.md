# Unified serving benchmark

Every frozen model measured through **its own real inference path**, on one fixed
setup, in **its own subprocess** so peak memory belongs to that model alone.

No test label is read and no test inference is performed. The input is a
deterministic leading slice of the validation model input; labels are present in
the file but are never used for scoring.

## Why two regimes

| Regime | What is timed |
|---|---|
| **pure** | inputs already prepared in memory; only the model call |
| **end_to_end** | Parquet batch read, encoding, scaling, sequence gather, model call |

Reporting only the pure regime would be misleading, and the size of the gap is the
point. Logistic regression reaches **18.1 million rows/s** pure at batch 8192 but
only **1.28 million** end to end — a **14×** drop, because building its
2,497,016-dimension sparse row dominates everything the model itself does. A
serving decision made on the pure number alone would be wrong.

## Setup

macOS arm64, Apple M4 Max, **CPU for every timing** with 8 declared threads. CPU is
the common ground across scikit-learn, LightGBM and PyTorch; the neural models were
trained on MPS, which is recorded but not used here. Single-row latency: 100 warmup
calls then 500 measured. Throughput: 3 warmup batches then 10 measured, at batch
1024 and 8192. Versions are pinned in the manifest and match
`requirements-tested.txt`.

## Model size and memory

| Model | File MiB | Params / size | Peak RSS MiB | Load s |
|---|---:|---|---:|---:|
| LR F123 | 26.6 | 2,496,993 nonzero of 2,497,016 | 705.7 | 0.18 |
| Wide & Deep F123 | 85.9 | 22,503,138 | 801.0 | 0.31 |
| DeepFM F123 | 85.9 | 22,503,138 | 813.5 | 0.30 |
| F4 mean matched | 85.9 | 22,503,138 + 57,297 residual | 1,034.2 | 0.32 |
| LightGBM F123 | **363.0** | 251 trees | 2,235.8 | 0.28 |
| **F5 mean matched** | 85.9 | 22,503,138 + 57,345 residual | **6,082.5** | 0.72 |

Two rows need reading carefully. **LightGBM has by far the largest model file** at
363 MiB for 251 trees. **F5 has by far the largest memory footprint** at 6.08 GiB,
because serving it requires the 723-million-event store resident to gather each
user's window; the residual itself adds only 57,345 parameters. Both sequence rows
share the frozen DeepFM checkpoint, so their file size is the backbone's and the
residual size is listed separately.

## Single-row latency, milliseconds

| Model | pure p50 | pure p95 | pure p99 | e2e p50 | e2e p95 | **e2e p99** |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM F123 | 0.057 | 0.068 | 0.093 | 0.148 | 0.175 | **0.223** |
| Wide & Deep F123 | 0.141 | 0.165 | 0.236 | 0.244 | 0.331 | **0.404** |
| DeepFM F123 | 0.153 | 0.180 | 0.276 | 0.257 | 0.344 | **0.416** |
| LR F123 | 0.060 | 0.069 | 0.073 | 0.327 | 0.373 | **0.394** |
| F5 mean matched | 0.200 | 0.240 | 0.286 | 0.296 | 0.338 | **0.359** |
| F4 mean matched | 0.195 | 0.215 | 0.333 | 0.307 | 0.364 | **0.491** |

At batch 1 every model is well under a millisecond end to end, so single-request
latency does not separate them. LR has the second-best pure p99 and the
second-worst end-to-end p50, again because of feature construction.

## Throughput, rows per second

| Model | pure @1024 | e2e @1024 | pure @8192 | **e2e @8192** |
|---|---:|---:|---:|---:|
| LR F123 | 7,969,131 | 1,031,335 | 18,083,051 | **1,275,056** |
| Wide & Deep F123 | 1,071,050 | 814,452 | 1,872,188 | **1,497,755** |
| DeepFM F123 | 848,906 | 718,126 | 1,530,445 | **1,239,580** |
| F4 mean matched | 647,125 | 538,124 | 1,021,321 | **842,292** |
| F5 mean matched | 318,527 | 230,643 | 658,608 | **379,078** |
| LightGBM F123 | 231,526 | 291,080 | 347,122 | **333,269** |

## The deployment trade-off

Against the frozen DeepFM backbone it sits on, **F5 costs about 3.3× the
end-to-end batch throughput and about 7.5× the peak memory**, and about 2.2× the
throughput of F4. Its validation AUC increment over the pipeline null is
**+0.0014**. Whether that is worth paying is a decision about the serving budget,
not about the metric: the accuracy gain is real and small, and the memory cost is
large and structural, since it comes from the event store rather than the model.

Two cheaper observations follow from the same table. **Wide & Deep has the highest
end-to-end throughput** of any model here, and **LightGBM the lowest**, which is
the opposite of what its small tree count might suggest. And LR's enormous pure
throughput is almost entirely erased by feature construction, so a sparse-feature
service would need that step optimised before the model mattered at all.

## Limits

- One machine, one thread count, one process per model. These are comparable to
  each other, not to a production serving stack.
- CPU only. The neural models were trained on MPS and would behave differently on
  a GPU or an accelerator.
- The sequence rows include the shared frozen backbone's cost; they are not the
  residual's cost in isolation.
- F5's memory reflects a design where the whole event store is resident. A
  production feature store would page or cache windows instead, so 6.08 GiB is
  the cost of *this* implementation, not a lower bound for the budget.
- The LR checkpoint was serialized under scikit-learn 1.8.0 and is read under
  1.9.1 with a `_loss` module alias. A parity check confirms the loaded model
  reproduces its saved validation predictions with a maximum error of **0.0**
  before any timing is taken.

## Reproduction

```bash
PY=/opt/anaconda3/envs/ctr_project/bin/python
$PY -B src/49_serving_benchmark.py --f5-residual-run f5b_mean_matched_l200_lr500_s42_v1
$PY -B -m unittest tests.test_serving_benchmark -v
```

Per-model JSON, the combined CSV and the environment manifest are in
[`results/serving_benchmark_v1/`](../results/serving_benchmark_v1/). Existing
per-model outputs are reused rather than overwritten, so a rerun measures only
what is missing.
