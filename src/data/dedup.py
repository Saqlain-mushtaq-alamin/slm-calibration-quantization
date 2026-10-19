"""Exact and near-duplicate removal for Phase 1 data prep."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover
    MinHash = None  # type: ignore
    MinHashLSH = None  # type: ignore


def normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def exact_dedup(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Remove exact duplicates after normalization."""
    df = df.copy()
    df["_text_norm"] = normalize_text_series(df[text_col])
    before = len(df)
    # Keep first occurrence; stable by original order
    df = df.drop_duplicates(subset="_text_norm", keep="first")
    after = len(df)
    print(f"Exact dedup: {before} -> {after} ({before - after} removed)")
    return df.drop(columns=["_text_norm"]).reset_index(drop=True)


def _minhash_for_text(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # Character 3-grams are robust for morphologically rich Bengali.
    t = text if text else ""
    if len(t) < 3:
        grams = [t] if t else [""]
    else:
        grams = [t[i : i + 3] for i in range(len(t) - 2)]
    for g in grams:
        m.update(g.encode("utf-8"))
    return m


def near_dedup(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = 0.8,
    num_perm: int = 128,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Remove near-duplicates via MinHash LSH (Jaccard threshold).

    Dedup is global across the dataframe (all domains / splits present).
    The first surviving example of each near-dup cluster is kept.
    """
    if MinHashLSH is None:
        raise ImportError("datasketch is required for near_dedup")
    df = df.copy().reset_index(drop=True)
    if id_col is None:
        id_col = "example_id" if "example_id" in df.columns else "_row_id"
        if id_col == "_row_id":
            df[id_col] = [str(i) for i in range(len(df))]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    texts = df[text_col].astype(str).tolist()
    keys = df[id_col].astype(str).tolist()
    keep_mask = [True] * len(df)

    for i, (key, text) in enumerate(zip(keys, texts)):
        mh = _minhash_for_text(text, num_perm=num_perm)
        try:
            # Query for already-indexed near-duplicates before inserting.
            neighbors = lsh.query(mh)
            if neighbors:
                keep_mask[i] = False
                continue
            lsh.insert(key, mh)
        except ValueError:
            # Same key already present → treat as duplicate.
            keep_mask[i] = False

    removed = int(sum(1 for k in keep_mask if not k))
    print(f"Near dedup (Jaccard>={threshold}): removed {removed} of {len(df)}")
    out = df.loc[keep_mask].copy()
    if id_col == "_row_id":
        out = out.drop(columns=["_row_id"])
    return out.reset_index(drop=True)


def dedup_pipeline(
    df: pd.DataFrame,
    *,
    text_col: str = "text",
    near_threshold: float = 0.8,
    num_perm: int = 128,
    run_near: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Exact + optional near dedup; returns cleaned frame and stats."""
    stats: dict[str, Any] = {"n_raw": int(len(df))}
    df1 = exact_dedup(df, text_col=text_col)
    stats["n_exact"] = int(len(df1))
    if run_near:
        df2 = near_dedup(
            df1, text_col=text_col, threshold=near_threshold, num_perm=num_perm
        )
    else:
        df2 = df1
    stats["n_final"] = int(len(df2))
    stats["removed_exact"] = stats["n_raw"] - stats["n_exact"]
    stats["removed_near"] = stats["n_exact"] - stats["n_final"]
    return df2, stats
