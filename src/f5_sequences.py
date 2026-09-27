"""F5 behavior-log sequences: event store, daily histories, donors and audits.

F5 is a NEW information budget. It does not replace, rewrite or read F4.

Design decisions, all conservative and recorded rather than tuned:

* Source is `behavior_log` restricted to the completed Beijing window
  [2017-04-22, 2017-05-14). Stage E1 established that earlier timestamps are
  sporadic or corrupt.
* Tokens are the event category plus its behavior type. Brand is admissible by
  the E1 audit but is deliberately not used in the primary budget, to keep the
  residual branch's capacity close to F4's.
* Category tokens reuse the existing **train-only ad-category mapping**, so a
  history token and the candidate ad share one semantic space. Categories that
  occur only in the behavior log map to UNKNOWN=1, which covers 4.3029% of
  in-window events.
* The cutoff is the completed-day rule already used by F2/F3/F4: an event may be
  used only if its Beijing date is strictly earlier than the impression's.
* Primary retained length is 200, fixed from the published Alimama protocol
  before any F5 metric exists.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from f4_sequences import ROOT, midnight, sha256, write_json  # noqa: F401  (re-exported)

SEED = 20260926

FORMAL_START = date(2017, 4, 22)
FORMAL_END_EXCLUSIVE = date(2017, 5, 14)
MAX_LENGTH = 200

# Smoke mode keeps one user residue class so both sides select the same users.
SMOKE_USER_MODULUS = 4001

# PAD=0 and UNKNOWN=1 are reserved exactly as in the existing encoders.
PAD_INDEX = 0
UNKNOWN_INDEX = 1

# Fixed enumeration from the dataset documentation, not learned from any split.
TYPE_TOKENS = {"pv": 2, "cart": 3, "fav": 4, "buy": 5}
TYPE_VOCABULARY_SIZE = 2 + len(TYPE_TOKENS)

BEHAVIOR_PARQUET = ROOT / "data" / "processed" / "behavior_raw_v1" / "behavior_log.parquet"
CATEGORY_MAPPING = ROOT / "artifacts" / "encoders" / "mappings" / "cate_id_raw_mapping.parquet"

ASSUMPTION = (
    "Behavior events carry their own timestamps, but F5 still applies the "
    "completed-Beijing-day cutoff used by F2/F3/F4 so the budgets stay "
    "comparable and no new feedback-availability assumption is introduced."
)


def source_fingerprints() -> dict[str, dict[str, object]]:
    paths = [
        BEHAVIOR_PARQUET,
        CATEGORY_MAPPING,
        *[ROOT / f"data/processed/{family}/{split}.parquet"
          for family in ("base", "model_input_f3")
          for split in ("train", "val", "test")],
    ]
    return {
        str(p.relative_to(ROOT)): {
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": sha256(p),
        }
        for p in paths
    }


def category_lookup() -> pl.DataFrame:
    """Train-only ad-category mapping, reused so history and candidate align."""
    mapping = pl.read_parquet(CATEGORY_MAPPING)
    if mapping["cate_idx"].min() < 2:
        raise ValueError("cate_idx must start at 2; 0 is PAD and 1 is UNKNOWN")
    return mapping.rename({"cate_id_raw": "cate"}).with_columns(
        pl.col("cate_idx").cast(pl.UInt16)
    )


def type_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "btag": list(TYPE_TOKENS),
            "type_idx": pl.Series(list(TYPE_TOKENS.values()), dtype=pl.UInt8),
        }
    )


def build_event_store(output: Path, partitions: int = 16, smoke: bool = False) -> dict:
    """Materialise the in-window event store with each user's events contiguous.

    A full global sort of 723 million rows is unnecessary and memory-risky. The
    (start, length) scheme only needs each user's events to be contiguous and
    time-ordered; the order of users relative to each other is irrelevant. So the
    users are split into `partitions` buckets by `user % partitions`, each bucket
    is sorted independently, and the buckets are concatenated in order.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = midnight(FORMAL_START)
    end_epoch = midnight(FORMAL_END_EXCLUSIVE)
    categories = category_lookup()
    types = type_lookup()

    partition_dir = output.parent / "partitions"
    partition_dir.mkdir(exist_ok=True)
    partition_paths = []
    for bucket in range(partitions):
        path = partition_dir / f"part_{bucket:03d}.parquet"
        # The source row index is attached BEFORE filtering so it identifies the
        # position in the original log. Sorting on (user, time_stamp) alone is not
        # deterministic for events sharing a second, and F5 models event order, so
        # the source position is the tiebreaker.
        frame = (
            pl.scan_parquet(BEHAVIOR_PARQUET)
            .with_row_index("source_index")
            .filter(
                (pl.col("time_stamp") >= start_epoch)
                & (pl.col("time_stamp") < end_epoch)
                & ((pl.col("user") % partitions) == bucket)
            )
        )
        if smoke:
            frame = frame.filter(pl.col("user") % SMOKE_USER_MODULUS == 0)
        frame = (
            frame.join(categories.lazy(), on="cate", how="left")
            .join(types.lazy(), on="btag", how="left")
            .select(
                pl.col("user").cast(pl.Int64).alias("user_raw"),
                pl.col("time_stamp").cast(pl.Int64),
                pl.col("cate_idx").fill_null(UNKNOWN_INDEX).cast(pl.UInt16),
                pl.col("type_idx").cast(pl.UInt8),
                pl.col("source_index").cast(pl.UInt32),
            )
            .sort("user_raw", "time_stamp", "source_index")
            .collect(engine="streaming")
            .drop("source_index")
        )
        if frame["type_idx"].null_count():
            raise ValueError("unmapped btag value; the type enumeration is fixed")
        frame.write_parquet(path, compression="zstd")
        partition_paths.append(path)
        print(f"  partition {bucket + 1}/{partitions}: {frame.height:,} events", flush=True)

    # Concatenate in bucket order and assign the global contiguous index.
    pl.concat([pl.scan_parquet(p) for p in partition_paths]).with_row_index(
        "event_index"
    ).sink_parquet(output, compression="zstd")
    for path in partition_paths:
        path.unlink()
    partition_dir.rmdir()

    return verify_event_store(output, partitions)


def verify_event_store(path: Path, partitions: int) -> dict:
    """Prove the two properties the (start, length) scheme depends on."""
    events = pl.scan_parquet(path)
    per_user = (
        events.group_by("user_raw")
        .agg(
            pl.len().alias("events"),
            pl.col("event_index").min().alias("first"),
            pl.col("event_index").max().alias("last"),
            # A descending step inside a user block breaks time ordering.
            (pl.col("time_stamp").diff() < 0).sum().alias("descending_steps"),
        )
        .collect(engine="streaming")
    )
    non_contiguous = per_user.filter(
        (pl.col("last") - pl.col("first") + 1) != pl.col("events")
    ).height
    unordered = int(per_user["descending_steps"].sum())
    if non_contiguous or unordered:
        raise ValueError(
            f"event store invalid: {non_contiguous} users non-contiguous, "
            f"{unordered} descending time steps"
        )
    totals = (
        events.select(
            pl.len().alias("events"),
            pl.col("user_raw").n_unique().alias("users"),
            pl.col("time_stamp").min().alias("min_epoch"),
            pl.col("time_stamp").max().alias("max_epoch"),
            (pl.col("cate_idx") == UNKNOWN_INDEX).sum().alias("unknown_category_tokens"),
            (pl.col("cate_idx") == PAD_INDEX).sum().alias("pad_category_tokens"),
        )
        .collect(engine="streaming")
        .row(0, named=True)
    )
    if totals["pad_category_tokens"]:
        raise ValueError("PAD index must never appear as a stored event token")
    by_type = (
        events.group_by("type_idx").agg(pl.len().alias("events"))
        .collect(engine="streaming").sort("type_idx")
    )
    return {
        "path": str(Path(path).relative_to(ROOT)),
        "partitions": partitions,
        "events": int(totals["events"]),
        "users": int(totals["users"]),
        "min_epoch": int(totals["min_epoch"]),
        "max_epoch": int(totals["max_epoch"]),
        "window_start_epoch": midnight(FORMAL_START),
        "window_end_exclusive_epoch": midnight(FORMAL_END_EXCLUSIVE),
        "unknown_category_tokens": int(totals["unknown_category_tokens"]),
        "unknown_category_share": totals["unknown_category_tokens"] / totals["events"],
        "events_by_type": {int(r["type_idx"]): int(r["events"]) for r in by_type.iter_rows(named=True)},
        "users_contiguous": True,
        "time_ordered_within_user": True,
        "tie_break": (
            "original behavior_log row position, so events sharing a second keep "
            "their source order and the store is byte-reproducible"
        ),
        "bytes": Path(path).stat().st_size,
        "sha256": sha256(path),
    }


def impression_activity() -> pl.DataFrame:
    """Past impression volume per user-date, reused for the donor activity bucket."""
    return (
        pl.concat(
            [
                pl.scan_parquet(ROOT / f"data/processed/base/{split}.parquet")
                .group_by("user_raw", "event_date_bj")
                .agg(pl.len().alias("impressions"))
                for split in ("train", "val", "test")
            ]
        )
        .collect(engine="streaming")
    )


def donor_permutation(history: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Deterministic different-user donor assignment with layered, counted fallback.

    The primary rule matches F4: a donor must be a different user whose **retained
    length is exactly** the target's, inside the same activity and recency bucket.
    At L=200 that rule alone is not always satisfiable. About 84% of rows sit at
    the cap, so the intermediate lengths are sparse and a group can hold a single
    user, which admits no different-user donor at all.

    Rather than silently changing a retained length, the rule is relaxed in
    recorded steps and every step is counted:

    * 0 — exact retained length, same activity bucket, same recency bucket.
    * 1 — exact retained length, same activity bucket (recency dropped).
    * 2 — exact retained length (activity dropped).
    * 3 — donor whose retained length is **at least** the target's, same activity
          and recency; the donor contributes its own most recent `n` events.

    Level 3 still yields a genuine, contiguous, pre-cutoff window of exactly the
    required length belonging to one different user. Its only asymmetry is that
    such donors are at least as long-history as the target, which is recorded
    rather than presented as a balanced match.
    """
    history = history.sort("user_raw")
    size = history.height
    length = history["length"].to_numpy().astype(np.int64)
    end_index = history["end"].to_numpy().astype(np.int64)
    activity = history["activity_bucket"].to_numpy().astype(np.int64)
    recency = history["recency_bucket"].to_numpy().astype(np.int64)
    identity = np.arange(size)

    donor = np.full(size, -1, dtype=np.int64)
    donor_length = np.zeros(size, dtype=np.int64)
    fallback = np.zeros(size, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    def assign(rows: np.ndarray, pool: np.ndarray, level: int, exact: bool) -> np.ndarray:
        """Assign donors from `pool` to `rows`; return the rows still unassigned."""
        if len(rows) == 0:
            return rows
        if len(pool) < 2 and not (len(pool) == 1 and len(rows) == 1 and pool[0] != rows[0]):
            if len(pool) == 0:
                return rows
        unresolved = []
        shuffled = rng.permutation(pool)
        for position, row in enumerate(rows):
            assigned = -1
            for step in range(len(shuffled)):
                candidate = shuffled[(position + step + 1) % len(shuffled)]
                if candidate != row:
                    assigned = candidate
                    break
            if assigned < 0:
                unresolved.append(row)
                continue
            donor[row] = assigned
            donor_length[row] = length[row] if not exact else length[assigned]
            fallback[row] = level
        return np.asarray(unresolved, dtype=np.int64)

    # Levels 0-2: exact retained length, progressively coarser context buckets.
    for n in np.unique(length):
        rows = np.flatnonzero(length == n)
        same_length = rows  # donors with exactly this retained length
        pending = rows
        for level, keys in enumerate(
            [
                activity * 16 + recency,
                activity,
                np.zeros(size, dtype=np.int64),
            ]
        ):
            if len(pending) == 0:
                break
            still = []
            for key in np.unique(keys[pending]):
                group_rows = pending[keys[pending] == key]
                pool = same_length[keys[same_length] == key]
                still.append(assign(group_rows, pool, level, exact=True))
            pending = np.concatenate(still) if still else np.asarray([], dtype=np.int64)
        # Level 3: donors with at least this retained length contribute their last n.
        if len(pending):
            keys = activity * 16 + recency
            still = []
            for key in np.unique(keys[pending]):
                group_rows = pending[keys[pending] == key]
                pool = np.flatnonzero((length >= n) & (keys == key))
                still.append(assign(group_rows, pool, 3, exact=False))
            pending = np.concatenate(still) if still else np.asarray([], dtype=np.int64)
        if len(pending):
            raise ValueError(
                f"No different-user donor available for retained length {n} "
                f"({len(pending)} rows); inspect the donor pool before proceeding"
            )

    if np.any(donor < 0):
        raise ValueError("unassigned donor")
    if np.any(donor == identity):
        raise ValueError("self-match in donor assignment")
    # The donor window is the donor's most recent `donor_length` events.
    donor_start = end_index[donor] - donor_length
    if np.any(donor_start < 0):
        raise ValueError("donor window starts before the event store")
    if not np.array_equal(donor_length, length):
        raise ValueError("donor retained length must equal the target retained length")

    return history.with_columns(
        pl.Series("donor_start", donor_start, dtype=pl.UInt32),
        pl.Series("donor_user_raw", history["user_raw"].to_numpy()[donor], dtype=pl.Int64),
        pl.Series("donor_length", donor_length, dtype=pl.UInt8),
        pl.Series("fallback", fallback, dtype=pl.UInt8),
    )


HISTORY_SCHEMA = {
    "user_raw": pl.Int64, "raw_length": pl.UInt32, "end": pl.UInt32,
    "length": pl.UInt8, "start": pl.UInt32, "activity_bucket": pl.UInt8,
    "recency_bucket": pl.UInt8, "donor_start": pl.UInt32,
    "donor_user_raw": pl.Int64, "donor_length": pl.UInt8, "fallback": pl.UInt8,
}


def load_event_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load only the columns the audit needs, in their narrowest safe dtypes.

    Int32 is exact for both the user identifier (max 1,141,729) and the epoch
    timestamp (max 1.49e9), so the whole store costs 11 bytes per event instead of
    the 23 bytes a full DataFrame of native dtypes would hold.
    """
    frame = pl.read_parquet(
        path, columns=["user_raw", "time_stamp", "cate_idx", "type_idx"]
    )
    arrays = {
        "user_raw": frame["user_raw"].cast(pl.Int32).to_numpy(),
        "time_stamp": frame["time_stamp"].cast(pl.Int32).to_numpy(),
        "cate_idx": frame["cate_idx"].to_numpy(),
        "type_idx": frame["type_idx"].to_numpy(),
    }
    del frame
    return arrays


def daily_histories(
    events: pl.LazyFrame,
    activity: pl.DataFrame,
    day: date,
    max_length: int = MAX_LENGTH,
    seed: int = SEED,
) -> pl.DataFrame:
    """Per-user retained window for one target date, plus its donor assignment.

    `events` stays lazy: the aggregation streams over the store and returns about
    1.1 million rows, so the 723-million-event table is never materialised.

    The cutoff is compared on the epoch timestamp against Beijing midnight, so no
    intermediate date column is needed and no UTC boundary can leak in.
    """
    cutoff = midnight(day)
    history = (
        events.filter(pl.col("time_stamp") < cutoff)
        .group_by("user_raw")
        .agg(
            pl.len().cast(pl.UInt32).alias("raw_length"),
            (pl.col("event_index").max() + 1).cast(pl.UInt32).alias("end"),
            pl.col("time_stamp").max().alias("last_event_epoch"),
        )
        .collect(engine="streaming")
    )
    if history.is_empty():
        return pl.DataFrame(schema=HISTORY_SCHEMA)
    if history["last_event_epoch"].max() >= cutoff:
        raise ValueError("cutoff filter failed")
    history = history.with_columns(
        pl.col("raw_length").clip(upper_bound=max_length).cast(pl.UInt8).alias("length")
    )
    past_impressions = (
        activity.filter(pl.col("event_date_bj") < day)
        .group_by("user_raw")
        .agg(pl.col("impressions").sum().alias("past_impressions"))
    )
    history = (
        history.join(past_impressions, on="user_raw", how="left", validate="1:1")
        .with_columns(
            (pl.col("end") - pl.col("length")).cast(pl.UInt32).alias("start"),
            (pl.col("past_impressions").fill_null(0) >= 100).cast(pl.UInt8).alias("activity_bucket"),
            # Whole Beijing days between the user's last retained event and the cutoff.
            ((cutoff - pl.col("last_event_epoch")) // 86400)
            .clip(upper_bound=2).cast(pl.UInt8).alias("recency_bucket"),
        )
        .drop("past_impressions", "last_event_epoch")
    )
    return donor_permutation(history, seed + day.toordinal())


def targets_for_day(split: str, day: date, smoke: bool) -> pl.DataFrame:
    frame = (
        pl.scan_parquet(ROOT / f"data/processed/base/{split}.parquet")
        .filter(pl.col("event_date_bj") == day)
        .select("impression_id", "user_raw", "event_date_bj", "clk")
    )
    if smoke:
        frame = frame.filter(pl.col("user_raw") % SMOKE_USER_MODULUS == 0)
    return frame.collect().sort("impression_id")


def attach_histories(targets: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    zero_filled = ("start", "donor_start", "length", "donor_length", "raw_length", "fallback")
    joined = targets.join(history, on="user_raw", how="left", validate="m:1").with_columns(
        [pl.col(c).fill_null(0) for c in zero_filled]
        + [pl.col("donor_user_raw").fill_null(-1)]
    )
    return joined.select(
        "impression_id", pl.col("clk").cast(pl.UInt8), "user_raw", "event_date_bj",
        pl.col("start").cast(pl.UInt32), pl.col("donor_start").cast(pl.UInt32),
        pl.col("length").cast(pl.UInt8), pl.col("donor_length").cast(pl.UInt8),
        pl.col("raw_length").cast(pl.UInt32), "donor_user_raw",
        pl.col("fallback").cast(pl.UInt8),
    ).sort("impression_id")


def assert_f123_identity(refs: pl.DataFrame, split: str, day: date, smoke: bool) -> None:
    """F5 must not change which impressions exist or what their labels are.

    The F123 model input is keyed by `impression_id` and carries `user_idx`, not
    `user_raw`, so a smoke run restricts the expected set by the impression ids
    actually present in `refs` and then checks that every reference resolves to an
    F123 row with an identical label. A reference pointing at an impression the
    model input does not contain still fails.
    """
    expected = (
        pl.scan_parquet(ROOT / f"data/processed/model_input_f3/{split}.parquet")
        .filter(pl.col("event_date_bj") == day)
        .select("impression_id", "clk")
    )
    if smoke:
        wanted = refs["impression_id"]
        expected = expected.filter(pl.col("impression_id").is_in(wanted.implode()))
    expected = expected.collect().sort("impression_id")
    if refs.height != expected.height:
        raise ValueError(
            f"{split} {day}: {refs.height} references resolve to {expected.height} "
            "F123 rows"
        )
    if refs["impression_id"].n_unique() != refs.height:
        raise ValueError("duplicate impression_id")
    np.testing.assert_array_equal(
        refs["impression_id"].to_numpy(), expected["impression_id"].to_numpy()
    )
    np.testing.assert_array_equal(refs["clk"].to_numpy(), expected["clk"].to_numpy())


def audit_references(
    refs: pl.DataFrame, arrays: dict[str, np.ndarray], day: date,
    max_length: int = MAX_LENGTH,
) -> dict:
    """Independent verification of every reference produced for one date."""
    lengths = refs["length"].to_numpy()
    starts = refs["start"].to_numpy()
    donor_starts = refs["donor_start"].to_numpy()
    positive = lengths > 0
    user = arrays["user_raw"]
    timestamps = arrays["time_stamp"]
    categories = arrays["cate_idx"]
    types = arrays["type_idx"]
    n_events = len(user)
    cutoff = midnight(day)

    counts = {
        "self_match_count": int(np.sum(positive & (
            refs["user_raw"].to_numpy() == refs["donor_user_raw"].to_numpy()))),
        "length_mismatch_count": int(np.sum(lengths != refs["donor_length"].to_numpy())),
        "future_event_count": 0,
        "identity_mismatch_count": 0,
        "order_violation_count": 0,
        "invalid_type_count": 0,
    }
    if not np.all(lengths <= max_length):
        raise ValueError("retained length exceeds the maximum")
    if not np.all(lengths == np.minimum(refs["raw_length"].to_numpy(), max_length)):
        raise ValueError("retained length is not the truncated raw length")
    if not (np.all(starts[~positive] == 0) and np.all(donor_starts[~positive] == 0)):
        raise ValueError("empty history must carry a zero pointer")

    token_stats = {}
    # Bounded 64K x L gathers; an N x L dense array is never allocated.
    for name, start, owners in [
        ("matched", starts, refs["user_raw"].to_numpy()),
        ("mismatched", donor_starts, refs["donor_user_raw"].to_numpy()),
    ]:
        unknown = tokens = 0
        for offset in range(0, refs.height, 65536):
            block = lengths[offset:offset + 65536]
            mask = np.arange(max_length)[None, :] < block[:, None]
            index = start[offset:offset + 65536, None] + np.arange(max_length)[None, :]
            index = np.where(mask, index, 0)
            if np.any(index >= n_events):
                raise ValueError("pointer past the end of the event store")
            owner_block = owners[offset:offset + 65536, None].astype(np.int32)
            counts["identity_mismatch_count"] += int(np.sum(mask & (user[index] != owner_block)))
            counts["future_event_count"] += int(np.sum(mask & (timestamps[index] >= cutoff)))
            counts["order_violation_count"] += int(np.sum(
                mask[:, 1:] & (timestamps[index[:, 1:]] < timestamps[index[:, :-1]])))
            counts["invalid_type_count"] += int(np.sum(
                mask & ((types[index] < 2) | (types[index] >= TYPE_VOCABULARY_SIZE))))
            unknown += int(np.sum(mask & (categories[index] == UNKNOWN_INDEX)))
            tokens += int(mask.sum())
            if np.any(mask & (categories[index] == PAD_INDEX)):
                raise ValueError("PAD index appeared inside a retained window")
        token_stats[name] = {
            "tokens": tokens,
            "unknown_tokens": unknown,
            "unknown_share": unknown / tokens if tokens else 0.0,
            "padding_slots": refs.height * max_length - tokens,
        }
    if any(value != 0 for value in counts.values()):
        raise ValueError(f"reference audit failed: {counts}")
    if not np.array_equal(
        np.bincount(lengths, minlength=max_length + 1),
        np.bincount(refs["donor_length"].to_numpy(), minlength=max_length + 1),
    ):
        raise ValueError("matched and donor retained-length distributions differ")

    return {
        "date": str(day),
        "rows": refs.height,
        **counts,
        "donor_fallback_levels": {
            "0_exact_length_activity_recency": int((refs["fallback"] == 0).sum()),
            "1_exact_length_activity": int((refs["fallback"] == 1).sum()),
            "2_exact_length_only": int((refs["fallback"] == 2).sum()),
            "3_donor_at_least_length": int((refs["fallback"] == 3).sum()),
        },
        "token_stats": token_stats,
        "empty_rows": int(np.sum(lengths == 0)),
        "empty_share": float(np.mean(lengths == 0)),
        "truncated_rows": int(np.sum(refs["raw_length"].to_numpy() > max_length)),
        "mean_retained": float(np.mean(lengths)),
        "length_histogram": np.bincount(lengths, minlength=max_length + 1).tolist(),
    }
