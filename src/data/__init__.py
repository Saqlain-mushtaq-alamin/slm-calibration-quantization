"""Data loading, splits, deduplication, domain-shift utilities."""

from src.data.assemble import (
    build_split_frames,
    resolve_model_id,
    split_frame_meta,
    split_sizes,
)
from src.data.loader import (
    load_bd_shs,
    load_english_control,
    load_hatexplain,
    load_bengali_hs_30k,
)
from src.data.dedup import dedup_pipeline, exact_dedup, near_dedup
from src.data.splits import (
    create_loco_splits,
    freeze_and_hash_splits,
    load_splits,
    verify_split_integrity,
)
from src.data.shift import compute_shift_magnitude, shift_rows_from_splits
from src.data.fertility import measure_all_fertility, measure_tokenizer_fertility

__all__ = [
    "load_bd_shs",
    "load_english_control",
    "load_hatexplain",
    "load_bengali_hs_30k",
    "dedup_pipeline",
    "exact_dedup",
    "near_dedup",
    "create_loco_splits",
    "freeze_and_hash_splits",
    "load_splits",
    "verify_split_integrity",
    "compute_shift_magnitude",
    "shift_rows_from_splits",
    "measure_all_fertility",
    "measure_tokenizer_fertility",
    "build_split_frames",
    "resolve_model_id",
    "split_frame_meta",
    "split_sizes",
]
