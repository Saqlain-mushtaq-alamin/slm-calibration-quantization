"""Tokenizer fertility measurement (tokens per word / character)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def measure_tokenizer_fertility(
    model_id: str,
    texts: Sequence[str],
    language: str,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Measure tokens-per-word and tokens-per-character for a tokenizer.

    Higher fertility means the model's representation is weaker / more
    fragmented for that language.
    """
    from transformers import AutoTokenizer

    texts = ["" if t is None else str(t) for t in texts]
    if max_samples is not None and len(texts) > max_samples:
        idx = np.linspace(0, len(texts) - 1, num=max_samples).astype(int)
        texts = [texts[i] for i in idx]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    total_tokens = 0
    total_words = 0
    total_chars = 0
    oov_tokens = 0

    unk_id = tokenizer.unk_token_id
    for text in texts:
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        if unk_id is not None:
            oov_tokens += sum(1 for t in tokens if t == unk_id)
        total_tokens += len(tokens)
        words = text.split()
        total_words += len(words)
        total_chars += len(text)

    return {
        "model": model_id,
        "language": language,
        "tokens_per_word": (total_tokens / total_words) if total_words else float("nan"),
        "tokens_per_char": (total_tokens / total_chars) if total_chars else float("nan"),
        "sample_size": len(texts),
        "total_tokens": total_tokens,
        "unk_token_rate": (oov_tokens / total_tokens) if total_tokens else float("nan"),
        "status": "ok",
    }


def measure_all_fertility(
    model_ids: Sequence[str],
    bn_texts: Sequence[str],
    en_texts: Sequence[str],
    *,
    n_samples: int = 1000,
    skip_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Run fertility measurement for each model on Bengali + English samples."""
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        for lang, texts in (("bn", bn_texts), ("en", en_texts)):
            try:
                row = measure_tokenizer_fertility(
                    model_id, texts, lang, max_samples=n_samples
                )
            except Exception as e:
                if not skip_on_error:
                    raise
                row = {
                    "model": model_id,
                    "language": lang,
                    "status": f"error: {type(e).__name__}: {e}",
                    "sample_size": 0,
                }
            rows.append(row)
            print(f"fertility {model_id} [{lang}]: {row.get('status')} "
                  f"tpw={row.get('tokens_per_word')} n={row.get('sample_size')}")
    return rows
