"""Dataset loading and preprocessing for Phase 1.

Primary Bengali dataset: BD-SHS (Romim et al., LREC 2022), ~50,281 samples.
Secondary: 30k Bengali hate-speech dataset (full topic labels for LOCO).
English control: HateXplain + a larger identity-labeled corpus for size matching.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BD_SHS_EXPECTED_N = 50_281
BD_SHS_MIN_N = 45_000
ENGLISH_CONTROL_MIN_N = BD_SHS_MIN_N

# Primary hate-type tokens used as the BD-SHS LOCO domain axis.
BD_SHS_PRIMARY_TYPES: tuple[str, ...] = (
    "callToViolence",
    "gender",
    "religion",
    "slander",
)
NON_HATE_DOMAIN = "non_hate"
OTHER_DOMAIN = "other"

# Candidate decoder SLM tokenizers for fertility (Phase 1 + later phases).
# meta-llama/* is gated (401) in this environment — use public TinyLlama as
# the Llama-family stand-in; re-add meta-llama after HF auth if available.
FERTILITY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "unsloth/gemma-2-2b-it",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "csebuetnlp/banglabert",
)

# Sentence-transformer used for embedding-based shift magnitudes.
SHIFT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip, collapse internal whitespace."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS_RE.sub(" ", str(text).lower().strip())


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}; have {list(df.columns)}")


# ---------------------------------------------------------------------------
# BD-SHS
# ---------------------------------------------------------------------------

def primary_hate_type(value: Any) -> str:
    """Map BD-SHS compound `type` labels onto a small LOCO domain axis.

    Non-hate rows have NaN type/target in the official release.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return NON_HATE_DOMAIN
    tokens = [t for t in re.split(r"[_\s]+", str(value).strip()) if t]
    for tok in tokens:
        if tok in BD_SHS_PRIMARY_TYPES:
            return tok
    return OTHER_DOMAIN


def load_bd_shs(raw_dir: str | Path = "data/raw/bd_shs") -> pd.DataFrame:
    """Load and verify BD-SHS train/val/test CSVs.

    Expected columns: sentence, target, type, hate speech.
    Returns DataFrame with columns:
      text, label, target, type, domain, source_split, example_id
    """
    raw_dir = Path(raw_dir)
    files = {
        "train": raw_dir / "train.csv",
        "val": raw_dir / "val.csv",
        "test": raw_dir / "test.csv",
    }
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"BD-SHS files not found: {missing}. "
            f"Download from Kaggle (e.g. iftekharuddin27/bd-shs) into {raw_dir}."
        )

    frames: list[pd.DataFrame] = []
    for split, path in files.items():
        df = pd.read_csv(path)
        _require_columns(df, ["sentence", "hate speech"], f"BD-SHS {split}")
        df = df.copy()
        df["source_split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(columns={"sentence": "text", "hate speech": "label"})
    _require_columns(df, ["text", "label"], "BD-SHS combined")
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    df["target"] = df.get("target")
    df["type"] = df.get("type")
    df["domain"] = df["type"].map(primary_hate_type)
    df["language"] = "bn"
    df["dataset"] = "bd_shs"
    df["example_id"] = [f"bd_shs_{i}" for i in range(len(df))]

    # T1.1 / T1.2 acceptance checks
    if len(df) < BD_SHS_MIN_N:
        raise AssertionError(f"Expected >= {BD_SHS_MIN_N} BD-SHS samples, got {len(df)}")
    if "text" not in df.columns or "label" not in df.columns:
        raise AssertionError("BD-SHS must expose text and label columns")
    label_counts = df["label"].value_counts().to_dict()
    print(f"BD-SHS loaded: {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Available columns: {list(df.columns)}")
    for col in ["domain", "type", "target", "source_split"]:
        print(f"  {col}: {df[col].nunique()} unique values")
        print(f"    {df[col].value_counts().head(12)}")
    print(f"  label counts: {label_counts}")

    meta_cols = [c for c in ("domain", "type", "target") if c in df.columns]
    n_unique = max((df[c].nunique() for c in meta_cols), default=0)
    if n_unique < 3:
        raise AssertionError("T1.2: need at least one metadata column with >= 3 domain values")
    return df


def load_bengali_hs_30k(
    csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the 30k Bengali hate-speech dataset (Romim et al., 2020/2021).

    Columns: sentence, hate, category. Used as a secondary LOCO axis because
    every row carries a topic category (unlike BD-SHS where type/target are
    hate-only annotations).
    """
    if csv_path is None:
        candidates = [
            Path("data/raw/bengali_hs_30k/Bengali hate speech .csv"),
            Path("data/raw/bengali_hs_30k/Bengali_hate_speech.csv"),
        ]
        csv_path = next((p for p in candidates if p.exists()), None)
        if csv_path is None:
            raise FileNotFoundError(
                "30k Bengali HS CSV not found under data/raw/bengali_hs_30k/."
            )
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    _require_columns(df, ["sentence", "hate", "category"], "bengali_hs_30k")
    df = df.rename(columns={"sentence": "text", "hate": "label", "category": "domain"})
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    df["language"] = "bn"
    df["dataset"] = "bengali_hs_30k"
    df["source_split"] = "full"
    df["example_id"] = [f"bn30k_{i}" for i in range(len(df))]
    print(f"Bengali 30k loaded: {len(df)} samples; domains={df['domain'].nunique()}")
    print(df["domain"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# English control
# ---------------------------------------------------------------------------

def load_hatexplain(raw_dir: str | Path = "data/raw/hatexplain") -> pd.DataFrame:
    """Load HateXplain (20,148 posts) from the official JSON dump.

    Majority-vote label -> binary hate (hatespeech=1; normal/offensive=0).
    Domain axis: majority targeted community from annotators.
    """
    raw_dir = Path(raw_dir)
    path = raw_dir / "hatexplain.json"
    if not path.exists():
        raise FileNotFoundError(f"HateXplain JSON not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("HateXplain JSON must be a dict keyed by post_id")

    rows: list[dict[str, Any]] = []
    for pid, item in obj.items():
        annotators = item.get("annotators") or []
        if not annotators:
            continue
        label_votes = [a.get("label") for a in annotators if isinstance(a, dict)]
        label_votes = [v for v in label_votes if v]
        if not label_votes:
            continue
        majority = Counter(label_votes).most_common(1)[0][0]
        label = 1 if majority == "hatespeech" else 0

        target_votes: list[str] = []
        for a in annotators:
            if not isinstance(a, dict):
                continue
            t = a.get("target")
            if t is None:
                continue
            if isinstance(t, list):
                target_votes.extend([str(x) for x in t if x and str(x).lower() != "none"])
            else:
                tv = str(t)
                if tv.lower() != "none":
                    target_votes.append(tv)
        domain = Counter(target_votes).most_common(1)[0][0] if target_votes else "unspecified"

        tokens = item.get("post_tokens") or []
        text = " ".join(tokens) if tokens else ""
        if not text.strip():
            continue
        platform = "twitter" if str(pid).endswith("_twitter") else (
            "gab" if str(pid).endswith("_gab") else "unknown"
        )
        rows.append(
            {
                "example_id": f"hateexplain_{pid}",
                "text": text,
                "label": label,
                "majority_label_raw": majority,
                "domain": domain,
                "platform": platform,
                "language": "en",
                "dataset": "hatexplain",
                "source_split": "full",
            }
        )
    df = pd.DataFrame(rows)
    print(f"HateXplain loaded: {len(df)} samples")
    print(f"  label prior: {df['label'].mean():.4f}")
    print(f"  domains ({df['domain'].nunique()}): {df['domain'].value_counts().head(12).to_dict()}")
    return df


def load_toxigen(raw_dir: str | Path = "data/raw/toxigen") -> pd.DataFrame:
    """Load ToxiGen (large English identity-focused toxicity corpus).

    Prefers a local parquet/csv cache of the standardized frame; otherwise
    downloads HF `toxigen/toxigen-data` train split.
    Binary label: prompt_label / toxicity score > 0.
    Domain axis: identity `group`.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in ("toxigen.parquet", "toxigen.csv", "train.csv"):
        p = raw_dir / name
        if not p.exists():
            continue
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        if {"text", "label", "domain"}.issubset(df.columns):
            out = df.copy()
            if "example_id" not in out.columns or out["example_id"].isna().all():
                out["example_id"] = [f"toxigen_{i}" for i in range(len(out))]
            if "language" not in out.columns:
                out["language"] = "en"
            if "dataset" not in out.columns:
                out["dataset"] = "toxigen"
            print(f"ToxiGen cache loaded: {len(out)} samples; prior={out['label'].mean():.4f}")
            print(
                f"  domains ({out['domain'].nunique()}): "
                f"{out['domain'].value_counts().head(10).to_dict()}"
            )
            return out
        # Raw-ish cache without standardized columns
        return _standardize_toxigen(df)

    try:
        from datasets import load_dataset

        # Prefer the large train split (identity-group + model toxicity labels);
        # fall back to the smaller human-annotated subset.
        try:
            ds = load_dataset("toxigen/toxigen-data", "train", split="train")
            df = ds.to_pandas()
            print(f"ToxiGen HF train split: {len(df)} rows, cols={list(df.columns)}")
        except Exception as e_train:
            print(f"ToxiGen train split failed ({e_train}); trying annotated")
            ds = load_dataset("toxigen/toxigen-data", "annotated", split="train")
            df = ds.to_pandas()
    except ImportError as e:  # pragma: no cover
        raise ImportError("datasets is required to download ToxiGen") from e

    out = _standardize_toxigen(df)
    # Cache for offline re-runs
    try:
        out.to_parquet(raw_dir / "toxigen.parquet", index=False)
    except Exception as e:  # pragma: no cover - optional cache
        print(f"ToxiGen parquet cache skipped: {e}")
    return out


def _standardize_toxigen(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # HF train split uses `generation` (generated claim) as the text field.
    text_col = next(
        (cols[k] for k in ("text", "sentence", "content", "generation") if k in cols),
        None,
    )
    if text_col is None:
        raise ValueError(f"ToxiGen: no text column; cols={list(df.columns)}")

    label = None
    # Prefer human annotations when present; else prompt_label / toxicity scores.
    if "toxicity_human" in cols:
        toxic = pd.to_numeric(df[cols["toxicity_human"]], errors="coerce").fillna(0)
        label = (toxic > 0.5).astype(int)
    elif "prompt_label" in cols:
        raw = df[cols["prompt_label"]]
        if raw.dtype == object:
            mapping = {
                "hate": 1, "toxic": 1, "toxigen": 1, "1": 1, 1: 1,
                "benign": 0, "non-toxic": 0, "normal": 0, "0": 0, 0: 0,
            }
            label = raw.map(lambda v: mapping.get(str(v).lower(), 0)).astype(int)
        else:
            label = (pd.to_numeric(raw, errors="coerce").fillna(0) > 0).astype(int)
    elif "toxicity_ai" in cols:
        toxic = pd.to_numeric(df[cols["toxicity_ai"]], errors="coerce").fillna(0)
        label = (toxic > 0.5).astype(int)
    elif "toxicity" in cols:
        toxic = pd.to_numeric(df[cols["toxicity"]], errors="coerce").fillna(0)
        label = (toxic > 0.5).astype(int)
    elif "label" in cols:
        raw = df[cols["label"]]
        if raw.dtype == object:
            mapping = {
                "hate": 1, "toxic": 1, "toxigen": 1, "1": 1, 1: 1,
                "benign": 0, "non-toxic": 0, "normal": 0, "0": 0, 0: 0,
            }
            label = raw.map(lambda v: mapping.get(str(v).lower(), 0)).astype(int)
        else:
            label = (pd.to_numeric(raw, errors="coerce").fillna(0) > 0).astype(int)
    else:
        raise ValueError(f"ToxiGen: no label/toxicity column; cols={list(df.columns)}")

    domain_col = None
    for cand in ("group", "predicted_group", "target_group", "identity"):
        if cand in cols:
            domain_col = cols[cand]
            break
    domain = (
        df[domain_col].astype(str).fillna("unspecified")
        if domain_col
        else pd.Series(["unspecified"] * len(df))
    )
    domain = domain.replace({"nan": "unspecified", "None": "unspecified", "": "unspecified"})

    out = pd.DataFrame(
        {
            "example_id": [f"toxigen_{i}" for i in range(len(df))],
            "text": df[text_col].astype(str),
            "label": label.astype(int),
            "domain": domain.values,
            "language": "en",
            "dataset": "toxigen",
            "source_split": "full",
        }
    )
    print(f"ToxiGen loaded: {len(out)} samples; prior={out['label'].mean():.4f}")
    print(f"  domains ({out['domain'].nunique()}): {out['domain'].value_counts().head(10).to_dict()}")
    return out


def subsample_to_match(
    df: pd.DataFrame,
    *,
    target_n: int,
    target_prior: float,
    seed: int = 42,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """Stratified subsample so size is close to target_n and prior to target_prior.

    Within each label stratum we keep as many rows as the class-prior target
    allows, then top-up/down to hit target_n as closely as possible.
    """
    if len(df) == 0:
        raise ValueError("subsample_to_match: empty dataframe")
    rng = np.random.default_rng(seed)
    n0 = max(1, min(len(df), target_n))
    n_pos = int(round(n0 * target_prior))
    n_pos = max(0, min(n0, n_pos))
    n_neg = n0 - n_pos

    pos = df[df[label_col] == 1]
    neg = df[df[label_col] == 0]
    if len(pos) < n_pos or len(neg) < n_neg:
        # Fall back to proportional sampling of what we have.
        total = len(df)
        n_pos = min(n_pos, len(pos))
        n_neg = min(n_neg, len(neg))
        # If still short overall, scale up to available size
        if n_pos + n_neg < min(target_n, total):
            # use as much as possible while keeping prior as close as we can
            avail = len(pos) + len(neg)
            n_pos = min(len(pos), int(round(avail * target_prior)))
            n_neg = min(len(neg), avail - n_pos)

    def _take(frame: pd.DataFrame, k: int) -> pd.DataFrame:
        if k <= 0 or len(frame) == 0:
            return frame.iloc[0:0]
        if k >= len(frame):
            return frame
        idx = rng.choice(len(frame), size=k, replace=False)
        return frame.iloc[np.sort(idx)]

    sampled = pd.concat([_take(pos, n_pos), _take(neg, n_neg)], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"Subsampled English control: {len(sampled)} rows "
        f"(prior={sampled[label_col].mean():.4f}, target_n={target_n}, target_prior={target_prior})"
    )
    return sampled


def load_english_control(
    dataset_name: str = "auto",
    *,
    target_n: int | None = None,
    target_prior: float | None = None,
    seed: int = 42,
    prefer: str = "toxigen",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the English control dataset and match BD-SHS size/class prior.

    Decision (Phase 1):
      - HateXplain has rich target-community metadata but only ~20k rows,
        so it CANNOT satisfy T1.10 (size within 10% of BD-SHS ~50k).
      - ToxiGen is large enough and has identity-group metadata for LOCO.
      - Therefore the primary English control is ToxiGen (subsampled).
      - HateXplain is retained as a secondary English metadata-rich corpus.
    """
    bd_prior = 24156 / 50281
    target_n = target_n if target_n is not None else BD_SHS_EXPECTED_N
    target_prior = target_prior if target_prior is not None else bd_prior

    decision: dict[str, Any] = {
        "chosen": None,
        "candidates": {},
        "target_n": target_n,
        "target_prior": target_prior,
        "justification": "",
    }

    hatexplain_df: pd.DataFrame | None = None
    try:
        hatexplain_df = load_hatexplain()
        decision["candidates"]["hatexplain"] = {
            "n": int(len(hatexplain_df)),
            "prior": float(hatexplain_df["label"].mean()),
            "n_domains": int(hatexplain_df["domain"].nunique()),
            "role": "secondary_metadata_rich",
        }
    except Exception as e:
        decision["candidates"]["hatexplain"] = {"error": str(e)}

    if dataset_name in ("auto", "toxigen", prefer):
        try:
            tox = load_toxigen()
        except Exception as e:
            print(f"ToxiGen load failed: {e}")
            tox = None
        if tox is not None:
            decision["candidates"]["toxigen"] = {
                "n": int(len(tox)),
                "prior": float(tox["label"].mean()),
                "n_domains": int(tox["domain"].nunique()),
                "role": "primary_control",
            }
            matched = subsample_to_match(
                tox, target_n=target_n, target_prior=target_prior, seed=seed
            )
            # Size / prior acceptance
            size_ok = abs(len(matched) - target_n) / target_n <= 0.10
            prior_ok = abs(matched["label"].mean() - target_prior) <= 0.02
            decision["chosen"] = "toxigen"
            decision["matched_n"] = int(len(matched))
            decision["matched_prior"] = float(matched["label"].mean())
            decision["size_ok"] = bool(size_ok)
            decision["prior_ok"] = bool(prior_ok)
            decision["justification"] = (
                "ToxiGen selected as primary English control: large enough to "
                "subsample to BD-SHS size (T1.10), binary toxicity labels, and "
                "identity-group metadata for LOCO-style domain shift. HateXplain "
                "is kept as a secondary English corpus with richer target-community "
                "labels but is too small (~20k) to match BD-SHS size."
            )
            print("English control decision:")
            print(json.dumps(decision, indent=2))
            return matched, decision

    # Fallback: HateXplain matched as closely as possible (may fail T1.10 size).
    if hatexplain_df is not None:
        matched = subsample_to_match(
            hatexplain_df,
            target_n=min(target_n, len(hatexplain_df)),
            target_prior=target_prior,
            seed=seed,
        )
        decision["chosen"] = "hatexplain_fallback"
        decision["matched_n"] = int(len(matched))
        decision["matched_prior"] = float(matched["label"].mean())
        decision["size_ok"] = False
        decision["prior_ok"] = abs(matched["label"].mean() - target_prior) <= 0.02
        decision["justification"] = (
            "Fallback to HateXplain; size will not meet T1.10. Prefer ToxiGen."
        )
        return matched, decision

    raise RuntimeError("No English control dataset could be loaded")


def write_english_control_decision(
    decision: dict[str, Any], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(decision, indent=2, default=_default), encoding="utf-8")
    return path
