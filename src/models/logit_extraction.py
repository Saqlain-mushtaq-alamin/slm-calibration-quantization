"""Per-example logit / probability extraction for HF causal LMs.

Two-way read-out: the model's next-token distribution at the answer position is
restricted to the two verbalizer tokens (``" A"`` hate, ``" B"`` non-hate) and
renormalised.  This keeps the output distribution a proper binary probability
vector, which every downstream metric assumes.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch

from src.models.finetune import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_VERBALIZER,
    build_prompt,
    encode_label_token,
)
from src.utils.atomic import atomic_write_parquet

LOG = logging.getLogger("slm-calib.logits")

REQUIRED_COLUMNS = [
    "example_id",
    "split",
    "y_true",
    "logit_hate",
    "logit_nonhate",
    "prob_hate",
    "prob_nonhate",
    "prediction",
    "confidence",
]

_LOGIT_COLUMNS = [
    "logit_hate",
    "logit_nonhate",
    "prob_hate",
    "prob_nonhate",
    "prediction",
    "confidence",
    "correct",
    "margin",
    "entropy",
]


def build_eval_examples(df: pd.DataFrame, template: str = DEFAULT_PROMPT_TEMPLATE) -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(getattr(row, "example_id")),
            "prompt": build_prompt(row.text, template),
            "y_true": int(row.label),
            "domain": str(getattr(row, "domain", "")),
        }
        for row in df.itertuples(index=False)
    ]





def answer_logits_batched(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    hate_token_id: int,
    nonhate_token_id: int,
    max_len: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Next-token logits at the answer position, restricted to the two labels.

    The answer position is the *last real* token of each sequence.  Batches are
    right-padded, so that index is ``attention_mask.sum(1) - 1`` and differs per
    row; indexing ``logits[:, -1, :]`` reads the padded slot of every row shorter
    than the longest prompt in the batch, which silently corrupts the labels.

    Truncation is forced to the left so an over-long prompt loses its prefix
    rather than the trailing verbalizer cue that carries the answer position.

    Returns ``(logit_hate, logit_nonhate)`` as float32 arrays.
    """
    device = next(model.parameters()).device
    previous_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    bos_id = getattr(tokenizer, "bos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    try:
        input_ids_list: list[list[int]] = []
        for p in prompts:
            tok_out = tokenizer(p, add_special_tokens=False)
            if hasattr(tok_out, "input_ids"):
                ids = tok_out.input_ids
            elif hasattr(tok_out, "__getitem__"):
                ids = tok_out["input_ids"]
            else:
                ids = tok_out
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze(0).tolist() if ids.ndim > 1 else ids.tolist()
            elif isinstance(ids, list) and ids and isinstance(ids[0], list):
                ids = ids[0]
            ids = list(ids)
            if bos_id is not None:
                budget = max_len - 1
                if len(ids) > budget:
                    ids = ids[-budget:] if budget > 0 else []
                ids = [bos_id] + ids
            else:
                if len(ids) > max_len:
                    ids = ids[-max_len:] if max_len > 0 else []
            input_ids_list.append(ids)
    finally:
        tokenizer.truncation_side = previous_side

    max_batch_len = max(len(ids) for ids in input_ids_list)
    batch_input_ids = []
    batch_mask = []
    for ids in input_ids_list:
        pad_len = max_batch_len - len(ids)
        batch_input_ids.append(ids + [pad_token_id] * pad_len)
        batch_mask.append([1] * len(ids) + [0] * pad_len)

    encoded = {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_mask, dtype=torch.long, device=device),
    }
    last_real = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(encoded["input_ids"].shape[0], device=device)

    base = getattr(model, "model", getattr(model, "transformer", None))
    if not hasattr(model, "peft_config") and base is not None and hasattr(model, "lm_head") and callable(getattr(model, "lm_head", None)):
        with torch.no_grad():
            out = base(**encoded, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else getattr(out, "last_hidden_state", out[0])
            last_hidden = hidden[rows, last_real, :]
            answer = model.lm_head(last_hidden)
            cap = getattr(model.config, "final_logit_softcapping", None)
            if cap:
                answer = torch.tanh(answer / cap) * cap
    else:
        with torch.no_grad():
            out = model(**encoded, use_cache=False)
        answer = out.logits[rows, last_real, :]

    return (
        answer[:, hate_token_id].float().cpu().numpy(),
        answer[:, nonhate_token_id].float().cpu().numpy(),
    )


def softmax_pair(logit_hate: np.ndarray, logit_nonhate: np.ndarray) -> np.ndarray:
    """Renormalised two-way probabilities, shape ``(N, 2)`` ordered [nonhate, hate]."""
    stacked = np.stack(
        [np.asarray(logit_nonhate, dtype=np.float64), np.asarray(logit_hate, dtype=np.float64)],
        axis=1,
    )
    shifted = stacked - stacked.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def probs_to_frame(
    example_ids: Iterable[str],
    split: str,
    y_true: Iterable[int],
    logit_hate: np.ndarray,
    logit_nonhate: np.ndarray,
    extra: dict[str, Iterable[Any]] | None = None,
) -> pd.DataFrame:
    """Assemble the canonical logits dataframe from raw two-way logits."""
    probs = softmax_pair(logit_hate, logit_nonhate)
    prob_nonhate, prob_hate = probs[:, 0], probs[:, 1]
    prediction = (prob_hate >= prob_nonhate).astype(np.int64)
    confidence = np.maximum(prob_hate, prob_nonhate)
    y = np.asarray(list(y_true), dtype=np.int64)
    eps = 1e-12
    entropy = -(prob_hate * np.log(prob_hate + eps) + prob_nonhate * np.log(prob_nonhate + eps))

    frame = pd.DataFrame(
        {
            "example_id": list(example_ids),
            "split": split,
            "y_true": y,
            "logit_hate": np.asarray(logit_hate, dtype=np.float64),
            "logit_nonhate": np.asarray(logit_nonhate, dtype=np.float64),
            "prob_hate": prob_hate,
            "prob_nonhate": prob_nonhate,
            "prediction": prediction,
            "confidence": confidence,
            "correct": (prediction == y).astype(np.int64),
            "margin": np.abs(np.asarray(logit_hate, dtype=np.float64) - np.asarray(logit_nonhate, dtype=np.float64)),
            "entropy": entropy,
        }
    )
    for key, values in (extra or {}).items():
        # Scalars (e.g. model_id="") must broadcast; only sequences are per-row.
        if values is None or isinstance(values, (str, bytes, int, float, bool)):
            frame[key] = values
        else:
            column = list(values)
            if len(column) != len(frame):
                raise ValueError(
                    f"extra[{key!r}] has length {len(column)} but the frame has {len(frame)} rows"
                )
            frame[key] = column
    return frame


def extract_logits_hf(
    model: Any,
    tokenizer: Any,
    dataset: pd.DataFrame,
    verbalizer: dict[str, str] | None = None,
    batch_size: int = 16,
    *,
    split: str = "eval",
    max_len: int = 256,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    model_id: str = "",
    device: str | torch.device | None = None,
    progress_every: int = 20,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Extract two-way label logits for every row of ``dataset``.

    ``dataset`` needs ``text``, ``label`` and (for provenance) ``example_id``.
    Returns the canonical logits frame with columns ``REQUIRED_COLUMNS`` plus
    ``correct``/``margin``/``entropy``.
    """
    v = verbalizer or DEFAULT_VERBALIZER
    hate_token_id = encode_label_token(tokenizer, v["hate"])
    nonhate_token_id = encode_label_token(tokenizer, v["nonhate"])
    if hate_token_id == nonhate_token_id:
        raise ValueError(f"Verbalizer tokens collide on id {hate_token_id}")

    if device is not None:
        model = model.to(device)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    if hasattr(model, "use_cache"):
        model.config.use_cache = False

    examples = build_eval_examples(dataset, template)
    prompts = [e["prompt"] for e in examples]
    ids = [e["example_id"] for e in examples]
    labels = [e["y_true"] for e in examples]
    domains = [e["domain"] for e in examples]

    hate_chunks: list[np.ndarray] = []
    nonhate_chunks: list[np.ndarray] = []
    t0 = time.perf_counter()
    n_batches = math_ceil(len(prompts), batch_size)
    for i, start in enumerate(range(0, len(prompts), batch_size)):
        if should_stop is not None and should_stop():
            raise KeyboardInterrupt(f"Interrupted during logit extraction on batch {i + 1}/{n_batches}")
        chunk = prompts[start : start + batch_size]
        lh, ln = answer_logits_batched(
            model, tokenizer, chunk, hate_token_id, nonhate_token_id, max_len=max_len
        )
        hate_chunks.append(lh)
        nonhate_chunks.append(ln)
        if progress_every and (i + 1) % progress_every == 0:
            LOG.info(
                "logit extraction %d/%d batches (%.1fs)", i + 1, n_batches, time.perf_counter() - t0
            )

    frame = probs_to_frame(
        ids,
        split,
        labels,
        np.concatenate(hate_chunks) if hate_chunks else np.zeros(0),
        np.concatenate(nonhate_chunks) if nonhate_chunks else np.zeros(0),
        extra={"domain": domains, "model": model_id, "language": dataset.get("language", pd.Series(["bn"] * len(dataset))).tolist()},
    )
    if was_training and hasattr(model, "train"):
        model.train()
    LOG.info("Extracted %d rows in %.1fs", len(frame), time.perf_counter() - t0)
    return frame


def math_ceil(a: int, b: int) -> int:
    return -(-a // b)


def validate_logits_frame(df: pd.DataFrame, prob_tol: float = 1e-9) -> dict[str, Any]:
    """T3.5/T3.10 gate: columns present, rows complete, probabilities normalised."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required logits columns: {missing}")
    if len(df) == 0:
        raise ValueError("Empty logits frame")
    sums = df["prob_hate"].to_numpy() + df["prob_nonhate"].to_numpy()
    max_dev = float(np.max(np.abs(sums - 1.0)))
    if max_dev > prob_tol:
        raise ValueError(f"prob_hate + prob_nonhate deviates from 1 by {max_dev}")
    if df["example_id"].duplicated().any():
        raise ValueError("Duplicate example_id in logits frame")
    if not np.isfinite(df[_LOGIT_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite values in logits frame")
    return {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "missing_columns": missing,
        "max_prob_sum_deviation": max_dev,
        "finite": True,
        "unique_example_ids": True,
    }


def save_logits(df: pd.DataFrame, path: str | Path, **meta: Any) -> Path:
    """Persist per-example logits as Parquet atomically (recomputable metrics later)."""
    out = Path(path)
    payload = df.copy()
    for k, v in meta.items():
        payload[k] = v
    atomic_write_parquet(out, payload)
    LOG.info("Saved logits -> %s (%.2f MB)", out, out.stat().st_size / 1e6)
    return out


def load_logits(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def agreement_between_frames(a: pd.DataFrame, b: pd.DataFrame, id_col: str = "example_id") -> dict[str, Any]:
    """Prediction agreement between two logits frames aligned on ``id_col``."""
    left = a.set_index(id_col)
    right = b.set_index(id_col).reindex(left.index)
    if right.isna()["prediction"].any():
        missing = int(right["prediction"].isna().sum())
        raise ValueError(f"{missing} example_ids missing from the second logits frame")
    same = (left["prediction"].to_numpy() == right["prediction"].to_numpy())
    return {
        "n": int(len(left)),
        "agreement_rate": float(same.mean()),
        "max_abs_prob_delta": float(
            np.max(np.abs(left["prob_hate"].to_numpy() - right["prob_hate"].to_numpy()))
        ),
        "mean_abs_prob_delta": float(
            np.mean(np.abs(left["prob_hate"].to_numpy() - right["prob_hate"].to_numpy()))
        ),
    }