"""Unified inference interface for Hugging Face and llama.cpp backends.

Provides a consolidated interface across both Track A (GPU / PyTorch / HF)
and Track B (GGUF / llama.cpp / CPU/GPU).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from src.models.finetune import DEFAULT_PROMPT_TEMPLATE, DEFAULT_VERBALIZER
from src.models.logit_extraction import extract_logits_hf
from src.models.quantize_track_b import extract_logits_llamacpp, free_gguf, load_gguf_model

LOG = logging.getLogger("slm-calib.inference")


class UnifiedInferenceModel:
    """Unified inference wrapper supporting both Hugging Face models and GGUF models."""

    def __init__(
        self,
        backend: str,
        model: Any = None,
        tokenizer: Any = None,
        model_path: str | Path | None = None,
        **kwargs: Any,
    ):
        self.backend = backend.lower().strip()
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = Path(model_path) if model_path else None
        self.kwargs = kwargs

        if self.backend in {"gguf", "llamacpp", "llama_cpp"}:
            if self.model is None and self.model_path:
                self.model = load_gguf_model(self.model_path, **kwargs)
        elif self.backend in {"hf", "transformers", "pytorch"}:
            if self.model is not None and hasattr(self.model, "eval"):
                self.model.eval()
        else:
            raise ValueError(f"Unsupported backend {backend!r}; choose 'hf' or 'gguf'")

    def extract_logits(
        self,
        dataset: pd.DataFrame,
        verbalizer: dict[str, str] | None = None,
        *,
        split: str = "eval",
        max_len: int = 256,
        template: str = DEFAULT_PROMPT_TEMPLATE,
        model_id: str = "",
        should_stop: Callable[[], bool] | None = None,
        **extra_kwargs: Any,
    ) -> pd.DataFrame:
        """Extract next-token logits for verbalizer tokens on dataset."""
        v = verbalizer or DEFAULT_VERBALIZER
        if self.backend in {"gguf", "llamacpp", "llama_cpp"}:
            return extract_logits_llamacpp(
                self.model,
                dataset=dataset,
                verbalizer=v,
                split=split,
                max_len=max_len,
                template=template,
                model_id=model_id,
                tokenizer=self.tokenizer,
                should_stop=should_stop,
                **extra_kwargs,
            )
        else:
            return extract_logits_hf(
                self.model,
                tokenizer=self.tokenizer,
                dataset=dataset,
                verbalizer=v,
                split=split,
                max_len=max_len,
                template=template,
                model_id=model_id,
                should_stop=should_stop,
                **extra_kwargs,
            )

    def close(self) -> None:
        """Free memory and release resources."""
        if self.backend in {"gguf", "llamacpp", "llama_cpp"}:
            free_gguf(self.model)
        else:
            if self.model is not None:
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.model = None


def extract_logits_unified(
    backend: str,
    model: Any,
    dataset: pd.DataFrame,
    tokenizer: Any = None,
    verbalizer: dict[str, str] | None = None,
    *,
    split: str = "eval",
    max_len: int = 256,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    model_id: str = "",
    should_stop: Callable[[], bool] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience function to extract logits across any backend."""
    wrapper = UnifiedInferenceModel(backend=backend, model=model, tokenizer=tokenizer, **kwargs)
    return wrapper.extract_logits(
        dataset=dataset,
        verbalizer=verbalizer,
        split=split,
        max_len=max_len,
        template=template,
        model_id=model_id,
        should_stop=should_stop,
    )

