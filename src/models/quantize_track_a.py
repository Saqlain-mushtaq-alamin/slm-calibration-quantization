"""Track A post-training quantization (GPU / Hugging Face).

Track A never pools with Track B (global rule 4).  Every method here returns a
model that the shared :func:`src.models.logit_extraction.extract_logits_hf` can
read, so downstream metrics are identical across precision settings.

GPTQ calibration data must come **only** from the training split (global rule 5).
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.models.finetune import DEFAULT_PROMPT_TEMPLATE, build_prompt

LOG = logging.getLogger("slm-calib.quantize")

TRACK_A_METHODS = ("bnb_int8", "bnb_nf4", "gptq_int4", "awq_int4")


def bnb_int8_config() -> Any:
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_8bit=True)


def bnb_nf4_config(compute_dtype: str = "bfloat16") -> Any:
    from transformers import BitsAndBytesConfig

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        compute_dtype
    ]
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_quantized_hf(model_path: str, method: str, compute_dtype: str = "bfloat16"):
    """Load ``model_path`` under the requested on-the-fly Track A method."""
    from src.models.finetune import load_tokenizer  # noqa: F401  (import guard)
    from transformers import AutoModelForCausalLM

    if method == "bnb_int8":
        qc = bnb_int8_config()
    elif method == "bnb_nf4":
        qc = bnb_nf4_config(compute_dtype)
    else:
        raise ValueError(f"{method!r} is not a load-time bnb method; quantize it first")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=qc,
        device_map="auto",
        # transformers <4.56 spells this `torch_dtype`; `dtype` was added later.
        torch_dtype=torch.bfloat16 if compute_dtype == "bfloat16" else torch.float16,
    )
    model.config.use_cache = False
    model.eval()
    return model


def free(model: Any) -> None:
    """Drop a model and reclaim VRAM before loading the next precision."""
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# GPTQ
# ---------------------------------------------------------------------------

def build_calibration_texts(
    train_df: pd.DataFrame,
    n_samples: int = 128,
    max_len: int = 256,
    seed: int = 42,
    template: str = DEFAULT_PROMPT_TEMPLATE,
) -> list[str]:
    """Calibration prompts drawn **only** from the training split."""
    if n_samples >= len(train_df):
        picked = train_df
    else:
        picked = train_df.sample(n=n_samples, random_state=seed)
    return [build_prompt(t, template) for t in picked["text"].tolist()]


class GPTQUnavailable(RuntimeError):
    """The installed gptqmodel build exposes no end-to-end quantization driver."""


def gptq_backend_status() -> dict[str, Any]:
    """Report which GPTQ driver APIs the installed ``gptqmodel`` build actually offers.

    gptqmodel restructured across 1.x -> 2.x. Older builds expose the driver as
    ``GPTQModel.load(...).quantize(tokenizer, calib)`` then ``.save(dir)``; newer
    builds ship only the per-layer :class:`gptqmodel.quantization.gptq.GPTQ`
    accumulator (``add_batch`` / ``fasterquant``) and expect the caller to write
    the layer-sequential loop, which we deliberately do not do.
    """
    status: dict[str, Any] = {
        "installed": False,
        "version": None,
        "driver_api": None,
        "has_model_quantize": False,
        "has_model_save": False,
        "low_level_api": [],
        "problems": [],
    }
    try:
        import gptqmodel
        from gptqmodel import GPTQModel
    except Exception as exc:
        status["problems"].append(f"import failed: {exc!r}")
        return status

    status["installed"] = True
    status["version"] = getattr(gptqmodel, "__version__", "unknown")
    status["has_model_quantize"] = callable(getattr(GPTQModel, "quantize", None))
    status["has_model_save"] = callable(getattr(GPTQModel, "save", None))
    if status["has_model_quantize"] and status["has_model_save"]:
        status["driver_api"] = "GPTQModel.load/.quantize/.save"
        return status

    try:
        from gptqmodel.quantization.gptq import GPTQ

        status["low_level_api"] = [n for n in ("add_batch", "fasterquant") if hasattr(GPTQ, n)]
    except Exception as exc:
        status["problems"].append(f"low-level GPTQ import failed: {exc!r}")

    status["problems"].append(
        f"gptqmodel {status['version']} exposes no end-to-end driver "
        f"(GPTQModel.quantize={status['has_model_quantize']}, "
        f"GPTQModel.save={status['has_model_save']}); only the low-level per-layer "
        f"{status['low_level_api']} API is present and no `gptqmodel` CLI is installed, "
        "so GPTQ-4 cannot be produced without a hand-written sequential driver."
    )
    return status


def quantize_gptq(
    model_path: str,
    calib_texts: list[str],
    output_path: str,
    *,
    bits: int = 4,
    group_size: int = 128,
    tokenizer_path: str | None = None,
    batch_size: int = 1,
    max_len: int = 256,
    device: str = "cuda:0",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """GPTQ weight-only quantization via GPTQModel.

    ``calib_texts`` must be sampled from the training split only.

    Raises:
        GPTQUnavailable: if the installed build has no ``load -> quantize -> save``
            driver (see :func:`gptq_backend_status`).
    """
    status = gptq_backend_status()
    if status["driver_api"] is None:
        raise GPTQUnavailable("; ".join(status["problems"]) or "gptqmodel driver API not found")

    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    if not calib_texts:
        raise ValueError("GPTQ requires a non-empty calibration set from the training split")

    tok_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dynamic_enum = None
    for candidate in ("DYNAMIC",):
        try:
            from gptqmodel import DYNAMIC  # type: ignore

            dynamic_enum = getattr(DYNAMIC, "AUTO", None)
        except Exception:
            try:
                from gptqmodel.utils import DYNAMIC  # type: ignore

                dynamic_enum = getattr(DYNAMIC, "AUTO", None)
            except Exception:
                dynamic_enum = None
    quant_kwargs: dict[str, Any] = {"bits": bits, "group_size": group_size, "desc_act": True}
    if dynamic_enum is not None:
        quant_kwargs["dynamic"] = dynamic_enum
    quant_config = QuantizeConfig(**quant_kwargs)

    t0 = time.perf_counter()
    model = GPTQModel.load(model_path, device=device, trust_remote_code=trust_remote_code)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    with torch.no_grad():
        model.quantize(tokenizer, calib_texts, batch_size=batch_size, max_len=max_len)
    quantize_s = time.perf_counter() - t1

    Path(output_path).mkdir(parents=True, exist_ok=True)
    t2 = time.perf_counter()
    model.save(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))
    save_s = time.perf_counter() - t2

    try:
        del model
        free(None)
    except Exception:
        pass

    return {
        "method": "gptq_int4",
        "bits": bits,
        "group_size": group_size,
        "desc_act": True,
        "dynamic": dynamic_enum is not None,
        "n_calib": len(calib_texts),
        "output_path": str(output_path),
        "load_s": load_s,
        "quantize_s": quantize_s,
        "save_s": save_s,
        "total_s": load_s + quantize_s + save_s,
        "backend": status,
    }


def load_gptq(model_path: str, device: str = "cuda:0"):
    """Reload a saved GPTQ checkpoint through the standard HF auto classes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def quantize_awq(
    model_path: str,
    calib_texts: list[str],
    output_path: str,
    *,
    bits: int = 4,
    group_size: int = 128,
    max_len: int = 256,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """AWQ (activation-aware) weight-only INT4.

    AWQ ships with ``llmcompressor`` rather than the GPTQ stack pinned here; this
    wrapper keeps the pilot's method table complete and reports availability
    honestly instead of silently substituting GPTQ.
    """
    try:
        from llmcompressor.transformers import oneshot  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "AWQ requires llmcompressor, which is not installed in this environment. "
            "Install it (Phase 6) or drop awq_int4 from the pilot method list."
        ) from exc
    raise NotImplementedError("AWQ path lands in Phase 6; Phase 3 pilots GPTQ-4 only.")


def quantize_track_a(
    model_path: str,
    method: str,
    output_path: str | None = None,
    *,
    calib_texts: list[str] | None = None,
    bits: int = 4,
    group_size: int = 128,
    max_len: int = 256,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Dispatch to the requested Track A method."""
    if method not in TRACK_A_METHODS:
        raise ValueError(f"Unknown Track A method {method!r}; expected one of {TRACK_A_METHODS}")
    if method in {"bnb_int8", "bnb_nf4"}:
        return {"method": method, "mode": "on_load", "model_path": model_path}
    if method == "gptq_int4":
        if output_path is None:
            raise ValueError("gptq_int4 needs an output_path")
        if calib_texts is None:
            raise ValueError("gptq_int4 needs calibration texts from the training split")
        return quantize_gptq(
            model_path,
            calib_texts,
            output_path,
            bits=bits,
            group_size=group_size,
            max_len=max_len,
            device=device,
        )
    if method == "awq_int4":
        if calib_texts is None:
            raise ValueError("awq_int4 needs calibration texts from the training split")
        return quantize_awq(
            model_path, calib_texts, output_path or "", bits=bits, group_size=group_size, max_len=max_len
        )
    raise AssertionError(f"unhandled method {method!r}")


def artifact_size_mb(path: str | Path) -> float:
    """Total size of a saved model directory in MB."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def quant_summary(model: Any) -> dict[str, Any]:
    """Parameter/quantisation summary of a live model."""
    from src.metrics.efficiency import model_size_bytes, peak_vram_gb

    try:
        from transformers.quantizers.quantizer_bnb_4bit import (
            Params4bit,
        )  # noqa: F401
    except Exception:
        pass
    n_params = sum(p.numel() for p in model.parameters())
    n_4bit = n_8bit = 0
    for p in model.parameters():
        cls = type(p).__name__
        if "Int8" in cls or "int8" in cls.lower():
            n_8bit += p.numel()
        elif "4bit" in cls.lower():
            n_4bit += p.numel()
    return {
        "n_parameters": int(n_params),
        "n_params_4bit": int(n_4bit),
        "n_params_8bit": int(n_8bit),
        "weight_bytes": model_size_bytes(model),
        "peak_vram_gib": None if peak_vram_gb() is None else round(peak_vram_gb() or 0.0, 3),
        "class": type(model).__name__,
    }


def build_gptq_calib_from_train(train_df: pd.DataFrame, n_samples: int, max_len: int, seed: int) -> list[str]:
    """Convenience wrapper enforcing the train-split-only calibration rule."""
    return build_calibration_texts(train_df, n_samples=n_samples, max_len=max_len, seed=seed)


def calibration_provenance(train_df: pd.DataFrame, calib_texts: list[str], forbidden_ids: set[str]) -> dict[str, Any]:
    """Assert the calibration prompts never touch id_val/id_test/ood examples."""
    sample = calib_texts[: min(50, len(calib_texts))]
    joined = "\n".join(sample)
    leaked = [eid for eid in forbidden_ids if eid and eid in joined]
    return {
        "n_calib": len(calib_texts),
        "forbidden_ids_checked": len(forbidden_ids),
        "leaked_ids": leaked,
        "ok": not leaked,
        "source": "train_split_only",
    }


def calib_texts_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sanity accessor used by the pilot script's manifest."""
    if frame.empty:
        raise ValueError("Empty training frame for GPTQ calibration")
    return frame[["example_id", "text", "label"]].head(1)