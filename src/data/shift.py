"""Domain-shift magnitude metrics (embedding + lexical)."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

_WS_RE = re.compile(r"\s+")


def _as_list(texts: Sequence[str]) -> list[str]:
    return ["" if t is None else str(t) for t in texts]


def _tokenize_words(text: str) -> list[str]:
    return [t for t in _WS_RE.split(text.lower().strip()) if t]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    t = _WS_RE.sub(" ", text.lower().strip())
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def compute_mmd(
    x: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    gamma: float | None = None,
) -> float:
    """Unbiased-ish MMD^2 with RBF kernel (median heuristic if gamma is None)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a2 = np.sum(a**2, axis=1)[:, None]
        b2 = np.sum(b**2, axis=1)[None, :]
        return np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)

    d_xx = _sq_dists(x, x)
    d_yy = _sq_dists(y, y)
    d_xy = _sq_dists(x, y)

    if gamma is None:
        all_d = np.concatenate([d_xx.ravel(), d_yy.ravel(), d_xy.ravel()])
        all_d = all_d[all_d > 0]
        med = float(np.median(all_d)) if len(all_d) else 1.0
        gamma = 1.0 / med if med > 0 else 1.0

    if kernel != "rbf":
        raise ValueError(f"Unsupported kernel: {kernel}")

    k_xx = np.exp(-gamma * d_xx)
    k_yy = np.exp(-gamma * d_yy)
    k_xy = np.exp(-gamma * d_xy)

    n = len(x)
    m = len(y)
    # Exclude diagonal for unbiased estimate when possible
    if n > 1:
        term_x = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    else:
        term_x = k_xx.mean()
    if m > 1:
        term_y = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    else:
        term_y = k_yy.mean()
    term_xy = k_xy.mean()
    mmd2 = term_x + term_y - 2.0 * term_xy
    return float(max(mmd2, 0.0))


def centroid_cosine_dist(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    a = np.asarray(emb_a, dtype=np.float64).mean(axis=0)
    b = np.asarray(emb_b, dtype=np.float64).mean(axis=0)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_vocab_overlap(source_texts: Sequence[str], target_texts: Sequence[str]) -> float:
    """Jaccard overlap of word-type sets."""
    src = set()
    tgt = set()
    for t in _as_list(source_texts):
        src.update(_tokenize_words(t))
    for t in _as_list(target_texts):
        tgt.update(_tokenize_words(t))
    if not src and not tgt:
        return 1.0
    if not src or not tgt:
        return 0.0
    return float(len(src & tgt) / len(src | tgt))


def compute_oov_rate(
    source_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    unit: str = "word",
) -> float:
    """Fraction of target tokens/types unseen in the source vocabulary."""
    src = set()
    for t in _as_list(source_texts):
        if unit == "word":
            src.update(_tokenize_words(t))
        else:
            src.update(_char_ngrams(t))
    if not src:
        return 1.0
    total = 0
    oov = 0
    for t in _as_list(target_texts):
        if unit == "word":
            toks = _tokenize_words(t)
        else:
            toks = list(_char_ngrams(t))
        total += len(toks)
        oov += sum(1 for tok in toks if tok not in src)
    if total == 0:
        return 1.0
    return float(oov / total)


def embed_texts(
    texts: Sequence[str],
    model_name: str,
    *,
    batch_size: int = 64,
    max_samples: int | None = 2000,
) -> np.ndarray:
    """Encode texts with a sentence-transformers model (CPU/GPU auto)."""
    from sentence_transformers import SentenceTransformer

    texts = _as_list(texts)
    if max_samples is not None and len(texts) > max_samples:
        idx = np.linspace(0, len(texts) - 1, num=max_samples).astype(int)
        texts = [texts[i] for i in idx]
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float64)


def compute_shift_magnitude(
    source_texts: Sequence[str],
    target_texts: Sequence[str],
    *,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    embeddings: tuple[np.ndarray, np.ndarray] | None = None,
    max_samples: int | None = 2000,
) -> dict[str, Any]:
    """Compute MMD, centroid cosine distance, vocab overlap, and OOV rate."""
    source_texts = _as_list(source_texts)
    target_texts = _as_list(target_texts)
    if embeddings is None:
        src_emb = embed_texts(source_texts, model_name, max_samples=max_samples)
        tgt_emb = embed_texts(target_texts, model_name, max_samples=max_samples)
    else:
        src_emb, tgt_emb = embeddings

    return {
        "mmd": compute_mmd(src_emb, tgt_emb),
        "centroid_cosine_dist": centroid_cosine_dist(src_emb, tgt_emb),
        "vocab_overlap": compute_vocab_overlap(source_texts, target_texts),
        "oov_rate_word": compute_oov_rate(source_texts, target_texts, unit="word"),
        "oov_rate_char3": compute_oov_rate(source_texts, target_texts, unit="char"),
        "n_source": len(source_texts),
        "n_target": len(target_texts),
        "embedding_model": model_name,
    }


def shift_rows_from_splits(
    df,
    splits: dict[str, Any],
    *,
    text_col: str = "text",
    id_col: str = "example_id",
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    max_samples_per_side: int = 1500,
) -> list[dict[str, Any]]:
    """For each LOCO split, measure source(train) → ood_test shift magnitude."""
    id_to_text = dict(zip(df[id_col].astype(str), df[text_col].astype(str)))
    rows: list[dict[str, Any]] = []
    for name, spec in splits.items():
        src_ids = spec["train_ids"]
        tgt_ids = spec["ood_test_ids"]
        src_texts = [id_to_text[i] for i in src_ids if i in id_to_text]
        tgt_texts = [id_to_text[i] for i in tgt_ids if i in id_to_text]
        metrics = compute_shift_magnitude(
            src_texts,
            tgt_texts,
            model_name=model_name,
            max_samples=max_samples_per_side,
        )
        priors = spec.get("class_priors", {})
        prior_src = priors.get("train", {})
        prior_tgt = priors.get("ood_test", {})
        metrics.update(
            {
                "split": name,
                "source_domain": "source",
                "target_domain": spec.get("target_domain"),
                "prior_train_hate": prior_src.get("1", prior_src.get(1)),
                "prior_ood_hate": prior_tgt.get("1", prior_tgt.get(1)),
            }
        )
        rows.append(metrics)
    return rows
