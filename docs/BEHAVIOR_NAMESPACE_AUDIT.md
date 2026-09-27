# Stage E1: cross-source namespace and behavior-window audit

## Why this stage exists

The completed F4 study represents user history as the ordered categories of a
user's **past clicked ads**. [`DIN_F4_DESIGN.md`](DIN_F4_DESIGN.md) §3 recorded the
alternative source, `behavior_log.csv`, as unusable at that time because the
repository had "no authoritative cross-source namespace contract" and because the
only alignment evidence available was a **non-random 200,000-row prefix**
(96.4472% category overlap, 83.7276% brand overlap). That was a measurement gap,
not a property of the data.

This stage closes the gap on the full population and decides whether an F5
behavior-sequence information budget is admissible. It reads only raw tables and
a verified lossless Parquet copy. No model input, checkpoint, prediction,
calibrator or frozen result is read or modified.

## Lossless Parquet conversion

`src/40_convert_behavior_log.py` converts the 22.10 GiB source CSV once into
`data/processed/behavior_raw_v1/behavior_log.parquet` (4.59 GiB, zstd). Repeated
streaming aggregation over the CSV is the reason the earlier pass settled for a
prefix; one verified conversion removes that constraint.

Losslessness is proven rather than assumed: the same fingerprint bundle is
computed **independently from the CSV and from the Parquet copy** and required to
match exactly. The bundle is the row count, per-column null/min/max, Int128
column sums (Float64 would silently lose precision on the `time_stamp` sum), and
exact `btag` value counts.

| Field | Value |
|---|---|
| Rows | 723,268,134 |
| Source bytes / SHA256 prefix | 23,728,773,580 / `a1b2c9cf719796f9…` |
| Parquet bytes | 4,924,494,245 |
| Fingerprints match | yes, all fields |
| Elapsed | 99.1 s |

Evidence: [`behavior_log_conversion.json`](../results/behavior_namespace_audit_v1/behavior_log_conversion.json).
The Parquet copy is a Git-ignored derived artifact; the committed manifest is
evidence about it, not a backup of it.

## Pre-declared admissibility thresholds

Fixed in `THRESHOLDS` in `src/41_audit_behavior_namespace.py` before the script
observes any coverage number, so the verdict cannot be tuned to the result.

| Threshold | Value | Role |
|---|---:|---|
| `ad_category_forward_coverage_min` | 0.95 | hard precondition for F5 |
| `impression_user_forward_coverage_min` | 0.95 | hard precondition for F5 |
| `ad_brand_forward_coverage_min` | 0.90 | secondary; failing it downgrades F5 to category-only tokens |

## 1. Namespace coverage, full population

| Relation | Left distinct | Right distinct | Shared | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| `ad_feature.cate_id` → `behavior_log.cate` | 6,769 | 12,968 | 6,762 | **99.8966%** | 52.1437% |
| `ad_feature.brand` → `behavior_log.brand` | 99,814 | 460,561 | 99,096 | **99.2807%** | 21.5164% |
| `raw_sample.user` → `behavior_log.user` | 1,141,729 | 1,136,340 | 1,136,340 | **99.5280%** | 100.0000% |

Forward coverage is the quantity that matters: it asks whether a candidate ad's
category, a candidate ad's brand, and an impression's user can be located in the
behavior log. Low reverse coverage is expected and is not a defect — the behavior
log spans the whole Taobao catalogue while advertised items are a subset.

Unmatched identifiers are enumerated by count and sampled in
[`namespace_audit.json`](../results/behavior_namespace_audit_v1/namespace_audit.json):
7 ad categories, 718 ad brands and 5,389 impression users have no behavior-log
presence. `raw_sample.user` ⊇ `behavior_log.user` exactly, so the behavior log
introduces no user outside the impression population.

Missing-value encoding differs across tables and must be declared explicitly:
`ad_feature.csv` writes a missing brand as the literal token `NULL` on 246,330
rows, which fails `Int64` parsing unless declared; `behavior_log.csv` and
`raw_sample.csv` contain no missing cells. A historical derived Parquet inferred
`Float64` for `ad_feature.brand` for this reason. This audit reads both brand
columns with an explicit `Int64` schema plus `null_values`, so the comparison is
on identical integer semantics.

## 2. Effective behavior window

1,162 distinct Beijing dates appear in the raw timestamps, spanning 1902 to 2036.
The usable window is narrow and sharply bounded:

| Segment | Dates | Events | Share |
|---|---:|---:|---:|
| Before 2017-04-22 | 653 | 127,136 | 0.0176% |
| **2017-04-22 … 2017-05-13** | **22** | **723,113,323** | **99.9786%** |
| 2017-05-14 and later | 487 | 27,675 | 0.0038% |

The ramp into the window is abrupt, which is why a single start date is
defensible:

| Date | Events |
|---|---:|
| 2017-04-18 | 1,835 |
| 2017-04-19 | 3,581 |
| 2017-04-20 | 11,932 |
| 2017-04-21 | 82,245 |
| **2017-04-22** | **29,943,250** |

**Correction to an earlier characterisation.** The behavior log does not provide
roughly 40 days of usable history before the first impression date. Records
before 2017-04-22 are sporadic or corrupt. The effective window is 14 completed
days before 2017-05-06 and 21 before 2017-05-13. `src/04_build_daily_behavior.py`
already hard-codes `VALID_START_DATE = 2017-04-22`; this audit confirms that
constant against the full population rather than against a prefix.

Formal F5 uses only completed Beijing dates in `[2017-04-22, 2017-05-14)`.

## 3. Behavior types and per-user volume

Within the formal window:

| `btag` | Events | Users with the type |
|---|---:|---:|
| `pv` | 688,749,534 | 1,135,008 |
| `cart` | 15,946,033 | 913,676 |
| `fav` | 9,301,837 | 538,873 |
| `buy` | 9,115,919 | 997,816 |

All 154,811 events excluded by the window filter are `pv`; the `cart`, `fav` and
`buy` totals are identical inside and outside the window, so every out-of-range
timestamp belongs to a page view.

Per-user event counts over 1,136,328 users: mean 636.36, median 387, p90 1,468,
p95 2,062, p99 3,751, max 43,559.

## 4. Available ordered history under a completed-day cutoff

This is the quantity that decides whether F5 is worth building. For every target
impression date, history is restricted to completed Beijing dates strictly
earlier, then truncated at the primary length **L=200** fixed from the published
Alimama protocol rather than from any metric in this project.

| Target date | Split | Impressions | Empty history | Mean available | Mean retained at L=200 | At cap |
|---|---|---:|---:|---:|---:|---:|
| 2017-05-06 | train | 3,270,348 | 0.599% | 659.7 | 175.12 | 74.79% |
| 2017-05-07 | train | 3,430,002 | 0.553% | 696.3 | 177.13 | 76.64% |
| 2017-05-08 | train | 3,354,523 | 0.483% | 749.6 | 179.26 | 78.53% |
| 2017-05-09 | train | 3,248,016 | 0.441% | 795.9 | 180.98 | 80.32% |
| 2017-05-10 | train | 3,333,752 | 0.384% | 840.7 | 182.47 | 81.62% |
| 2017-05-11 | train | 3,378,604 | 0.367% | 894.0 | 183.96 | 83.07% |
| 2017-05-12 | val | 3,234,051 | 0.394% | 931.5 | 184.48 | 83.66% |
| 2017-05-13 | test | 3,308,665 | 0.435% | 966.7 | **185.19** | 84.44% |

Against the existing F4 budget on the same test day:

| | F4: past clicked-ad categories, L=20 | F5: `behavior_log` events, L=200 |
|---|---:|---:|
| Mean retained tokens | 4.687 | **185.19** |
| Empty-history share | 30.21% | **0.435%** |
| First training day (2017-05-06) | no prior-day clicks exist | 0.599% empty |

F5 retains about **39.5×** more tokens per impression and leaves about **69×**
fewer impressions with no history. The first-training-day gap that affects F3
(all entity histories fall back to the prior) and F4 does not affect F5, because
the behavior window opens 14 days before the first impression date.

These are availability counts, not an effect estimate. They establish that the
small F4 sequence effects were measured on a source with very little history;
they do not by themselves predict that a richer source will produce a larger
effect. Testing that is the purpose of F5.

## 5. Integrity

| Check | Result |
|---|---|
| Null cells, all five columns | 0 |
| Negative timestamps | 5,521 |
| Non-positive `cate` / `brand` | 0 / 0 |
| `btag` outside `{pv, cart, fav, buy}` | 0 |
| Row-level CSV ↔ Parquet recheck | 500 requested, 500 sampled, stride 1,446,536, exact match |

The recheck selects deterministic evenly spaced row indices rather than a prefix,
and asserts that the requested row count was actually returned. Without that
count assertion an index filter matching nothing would leave both frames empty
and make the equality comparison report a vacuous pass.

## Verdict

| Field | Result |
|---|---|
| `category_precondition_met` | true (99.8966% ≥ 95%) |
| `user_precondition_met` | true (99.5280% ≥ 95%) |
| `brand_token_admissible` | true (99.2807% ≥ 90%) |
| **`f5_admissible`** | **true** |

F5 proceeds with category **and** behavior-type tokens; brand is additionally
admissible as a token field. The shared semantic space between a candidate ad's
category and a behavior event's category is established at the population level,
which is the precondition for candidate-conditioned attention to be meaningful.

## Scope and limits

- High identifier overlap establishes that the encodings are almost certainly the
  same namespace. It is consistent with, but does not by itself prove, identical
  editorial semantics; the official dataset documentation and the date/volume
  distribution above are the corroborating evidence.
- The audit measures availability, not usefulness. No model was trained, scored
  or selected in this stage.
- `PRIMARY_SEQUENCE_LENGTH = 200` is fixed from the external published protocol
  before any F5 metric exists, so it is not a length selected on this project's
  validation or test data.
- The completed-day cutoff is the same conservative rule used by F2/F3/F4. The
  behavior events carry usable timestamps, so a finer intra-day cutoff is
  possible; it is deliberately not used, to keep F5 comparable with the existing
  budgets and to avoid a new feedback-availability assumption.

## Reproduction

```bash
python -B src/40_convert_behavior_log.py            # once; refuses to overwrite
python -B src/40_convert_behavior_log.py --reverify  # re-check without rewriting
python -B src/41_audit_behavior_namespace.py
python -B -m unittest tests.test_behavior_namespace -v
```

Results: [`results/behavior_namespace_audit_v1/`](../results/behavior_namespace_audit_v1/)
— `namespace_audit.json`, `namespace_coverage.csv`, `daily_events.csv`,
`behavior_types.csv`, `per_user_event_quantiles.csv`, `available_history.csv`,
`behavior_log_conversion.json`.
