"""Materialize frozen LOCO split definitions into capped per-part dataframes.

Phase 1 freezes splits as bare id lists so they stay independent of any model
code.  This module is the bridge back to usable frames: it maps ids back to rows
of the processed dataset, applies the deterministic subsample caps the compute
budget requires, and refuses to hand back a part that lost a class.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SPLIT_PARTS: tuple[str, ...] = ("train", "id_val", "id_test", "ood_test")


def resolve_model_id(hf_id: str, mirrors: Sequence[str] = ()) -> tuple[str, list[str]]:
    """Return ``(model_id, notes)`` for the first reachable candidate.

    Gated repositories (meta-llama/*, google/gemma-*) cannot be fetched without an
    HF token.  Community mirrors of the *same weights* are listed in the config and
    tried in order; every substitution is returned as a note so it lands in the run
    manifest instead of silently changing the model under the reader.
    """
    from transformers import AutoConfig, AutoTokenizer

    attempted: list[str] = []
    candidates = [hf_id, *mirrors]
    for i, candidate in enumerate(candidates):
        attempted.append(candidate)
        try:
            AutoTokenizer.from_pretrained(candidate)
            AutoConfig.from_pretrained(candidate)
        except Exception:
            continue
        notes: list[str] = []
        if i > 0:
            notes.append(
                f"canonical id {hf_id!r} was unreachable (gated/unavailable); "
                f"used mirror {candidate!r} of the same weights"
            )
        return candidate, notes

    raise RuntimeError(
        f"None of the candidates for {hf_id!r} could be loaded. Tried: {attempted}"
    )


def build_split_frames(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    caps: Mapping[str, int | None] | None = None,
    seed: int = 42,
    id_col: str = "example_id",
    parts: Iterable[str] = SPLIT_PARTS,
) -> dict[str, pd.DataFrame]:
    """Materialize ``parts`` of one LOCO spec, optionally capped by subsampling.

    Capping uses the largest-remainder stratified :func:`src.models.finetune.subsample`
    so the class ratio survives.  A cap that drops a class entirely raises, because a
    single-class training or validation frame makes the run silently meaningless.
    """
    from src.models.finetune import subsample

    if id_col not in df.columns:
        raise ValueError(f"dataframe is missing the id column {id_col!r}")
    frame = df.copy()
    frame[id_col] = frame[id_col].astype(str)
    by_id = frame.set_index(id_col, drop=False)

    caps = dict(caps or {})
    out: dict[str, pd.DataFrame] = {}
    for part in parts:
        key = f"{part}_ids"
        if key not in spec:
            raise KeyError(f"split spec has no {key!r}")
        ids = [str(i) for i in spec[key]]
        missing = [i for i in ids if i not in by_id.index]
        if missing:
            raise KeyError(
                f"{len(missing)} ids from {key} are absent from the processed frame "
                f"(first: {missing[0]}); splits and data are out of sync"
            )
        part_df = by_id.loc[ids].reset_index(drop=True)

        cap = caps.get(part)
        if cap:
            part_df = subsample(part_df, int(cap), seed=seed)

        seen = set(part_df["label"].astype(int).unique())
        if not seen <= {0, 1}:
            raise AssertionError(f"{part} contains labels outside {{0, 1}}: {sorted(seen)}")
        if len(seen) < 2 and part != "ood_test":
            # ood_test is single-class by construction for the bd_shs LOCO axis;
            # train/id_val/id_test must keep both classes.
            raise AssertionError(
                f"{part} subsample lost a class (labels={sorted(seen)}); "
                "raise the cap so both classes survive"
            )
        out[part] = part_df
    return out


def split_sizes(spec: Mapping[str, Any]) -> dict[str, int]:
    """Per-part sizes of a LOCO spec, read from the frozen ``sizes`` block."""
    sizes = spec.get("sizes") or {}
    return {part: int(sizes.get(part, len(spec.get(f"{part}_ids", [])))) for part in SPLIT_PARTS}


def split_frame_meta(
    frames: Mapping[str, pd.DataFrame], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Provenance block attached to every Phase 4 run manifest."""
    from src.models.finetune import frame_hash

    return {
        "target_domain": spec.get("target_domain"),
        "source_domains": list(spec.get("source_domains", [])),
        "sizes_full": split_sizes(spec),
        "sizes_used": {part: int(len(df)) for part, df in frames.items()},
        "class_balance_used": {
            part: {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()}
            for part, df in frames.items()
        },
        "frame_hashes": {part: frame_hash(df) for part, df in frames.items()},
    }