"""Track B post-training quantization (GGUF / llama.cpp).

Global Rule 4: Track A and Track B are separate. Never pool them.

Provides:
- Conversion of merged BF16 Hugging Face models to GGUF format (F16)
- Quantization of GGUF models to Q8_0, Q4_K_M, Q3_K_M (and optional Q2_K)
- Inference and per-example logit extraction via llama-cpp-python
- Sanity checking for GGUF F16 vs HF BF16 reference (T7.5 critical gate)
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.models.finetune import (
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_VERBALIZER,
    build_prompt,
    encode_label_token,
)
from src.models.logit_extraction import probs_to_frame

LOG = logging.getLogger("slm-calib.quantize_b")

TRACK_B_METHODS: tuple[str, ...] = ("f16", "q8_0", "q4_k_m", "q3_k_m")
OPTIONAL_METHODS: tuple[str, ...] = ("q2_k",)
ALL_TRACK_B_METHODS: tuple[str, ...] = TRACK_B_METHODS + OPTIONAL_METHODS

# Map method names to llama.cpp quant type string and ftype enum
LLAMA_QUANT_TYPE_NAMES: dict[str, str] = {
    "f16": "f16",
    "q8_0": "q8_0",
    "q4_k_m": "q4_k_m",
    "q4_k_s": "q4_k_s",
    "q3_k_m": "q3_k_m",
    "q3_k_s": "q3_k_s",
    "q3_k_l": "q3_k_l",
    "q2_k": "q2_k",
    "q4_0": "q4_0",
    "q5_0": "q5_0",
    "q5_k_m": "q5_k_m",
    "q6_k": "q6_k",
}


class GGUFConversionUnavailable(RuntimeError):
    """The convert_hf_to_gguf.py script could not be located or failed."""


class GGUFQuantizeUnavailable(RuntimeError):
    """Neither llama-quantize binary nor llama_cpp.llama_model_quantize is available."""


def check_llama_gpu_support() -> bool:
    """Check if the installed llama_cpp build supports CUDA / GPU offload."""
    try:
        import llama_cpp
        if hasattr(llama_cpp, "llama_supports_gpu_offload"):
            return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        pass
    return False


def is_valid_gguf_file(path: str | Path, min_size_bytes: int = 1024 * 1024) -> bool:
    """Verify that a GGUF file exists, has a valid header ('GGUF'), and non-trivial size."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        size = p.stat().st_size
        if size < min_size_bytes:
            return False
        with open(p, "rb") as f:
            magic = f.read(4)
            return magic == b"GGUF"
    except Exception:
        return False


def gguf_file_size_mb(path: str | Path) -> float:
    """Return size of file in Megabytes (MB)."""
    p = Path(path)
    if not p.is_file():
        return 0.0
    return round(p.stat().st_size / (1024 * 1024), 2)


def find_convert_script(custom_path: str | Path | None = None) -> Path:
    """Locate convert_hf_to_gguf.py script across repository and system paths."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p

    env_path = os.environ.get("LLAMACPP_CONVERT_SCRIPT")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "outputs" / "quantized" / "llama.cpp" / "convert_hf_to_gguf.py",
        root / "llama.cpp" / "convert_hf_to_gguf.py",
        root / "scripts" / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.is_file():
            return c

    sys_which = shutil.which("convert_hf_to_gguf.py")
    if sys_which:
        return Path(sys_which).resolve()

    raise GGUFConversionUnavailable(
        "Could not find convert_hf_to_gguf.py. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def find_llama_quantize_binary(custom_path: str | Path | None = None) -> Path | None:
    """Locate standalone llama-quantize executable if compiled."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file():
            return p

    env_path = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return p

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "outputs" / "quantized" / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        root / "outputs" / "quantized" / "llama.cpp" / "build" / "bin" / "llama-quantize",
        root / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        root / "llama.cpp" / "build" / "bin" / "llama-quantize",
    ]
    for c in candidates:
        if c.is_file():
            return c

    for name in ("llama-quantize", "llama-quantize.exe"):
        sys_which = shutil.which(name)
        if sys_which:
            return Path(sys_which).resolve()

    return None


def get_llama_ftype_enum(quant_type: str) -> int:
    """Map string quant_type (e.g. 'q8_0', 'q4_k_m') to llama_cpp LLAMA_FTYPE enum value."""
    import llama_cpp

    norm = quant_type.lower().strip()
    mapping: dict[str, int] = {
        "f32": getattr(llama_cpp, "LLAMA_FTYPE_ALL_F32", 0),
        "f16": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_F16", 1),
        "q4_0": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q4_0", 2),
        "q4_1": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q4_1", 3),
        "q8_0": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q8_0", 7),
        "q5_0": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q5_0", 8),
        "q5_1": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q5_1", 9),
        "q2_k": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q2_K", 10),
        "q3_k_s": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q3_K_S", 11),
        "q3_k_m": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q3_K_M", 12),
        "q3_k_l": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q3_K_L", 13),
        "q4_k_s": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q4_K_S", 14),
        "q4_k_m": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q4_K_M", 15),
        "q5_k_s": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q5_K_S", 16),
        "q5_k_m": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q5_K_M", 17),
        "q6_k": getattr(llama_cpp, "LLAMA_FTYPE_MOSTLY_Q6_K", 18),
    }
    if norm in mapping:
        return mapping[norm]

    # Try matching by attribute name
    attr_name = f"LLAMA_FTYPE_MOSTLY_{norm.upper()}"
    if hasattr(llama_cpp, attr_name):
        return getattr(llama_cpp, attr_name)

    raise ValueError(f"Unknown GGUF quantization type: {quant_type!r}. Available: {list(mapping.keys())}")


# ---------------------------------------------------------------------------
# Task 7.1 — GGUF Conversion & Quantization
# ---------------------------------------------------------------------------

def convert_to_gguf(
    model_path: str | Path,
    output_path: str | Path,
    outtype: str = "f16",
    convert_script: str | Path | None = None,
    extra_args: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Convert Hugging Face merged checkpoint to GGUF format using convert_hf_to_gguf.py.

    Writes to a temporary .part file and atomically renames on success.

    Args:
        model_path: Path to directory containing HF merged checkpoint.
        output_path: Target .gguf file path.
        outtype: Output floating-point precision ('f16', 'bf16', 'f32').
        convert_script: Path to convert_hf_to_gguf.py.
        extra_args: Additional command-line flags.

    Returns:
        dict with conversion metadata (duration, sizes, status).
    """
    model_p = Path(model_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    tmp_p = out_p.with_name(out_p.name + ".part")
    if tmp_p.exists():
        tmp_p.unlink()

    script_path = find_convert_script(convert_script)
    python_exe = sys.executable

    cmd = [
        str(python_exe),
        str(script_path),
        str(model_p),
        "--outfile",
        str(tmp_p),
        "--outtype",
        outtype,
    ]
    if extra_args:
        cmd.extend(extra_args)

    LOG.info("Converting HF model %s -> GGUF %s (%s)", model_p.name, out_p.name, outtype)
    t0 = time.perf_counter()

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.perf_counter() - t0

        if res.returncode != 0:
            LOG.error("GGUF conversion failed (code %d):\nSTDOUT:\n%s\nSTDERR:\n%s", res.returncode, res.stdout, res.stderr)
            raise RuntimeError(
                f"convert_hf_to_gguf.py failed with return code {res.returncode}:\n{res.stderr or res.stdout}"
            )

        if not is_valid_gguf_file(tmp_p):
            raise RuntimeError(f"GGUF conversion completed but temporary file {tmp_p} is not a valid GGUF file")

        os.replace(tmp_p, out_p)
    finally:
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass

    size_mb = gguf_file_size_mb(out_p)
    LOG.info("Converted %s successfully: %.2f MB in %.2fs", out_p.name, size_mb, duration)
    return {
        "status": "success",
        "method": outtype,
        "output_path": str(out_p),
        "file_size_mb": size_mb,
        "convert_s": round(duration, 2),
    }


def quantize_gguf(
    input_gguf: str | Path,
    output_gguf: str | Path,
    quant_type: str,
    quantize_bin: str | Path | None = None,
    n_threads: int | None = None,
) -> dict[str, Any]:
    """Quantize GGUF model to specified type (Q8_0, Q4_K_M, Q3_K_M, Q2_K).

    Uses standalone llama-quantize binary if available, or falls back to
    llama_cpp.llama_model_quantize C-API. Writes to a temporary .part file
    and atomically renames on success.

    Args:
        input_gguf: Path to source F16 GGUF model.
        output_gguf: Target quantized GGUF path.
        quant_type: Target quantization scheme (e.g. 'q8_0', 'q4_k_m', 'q3_k_m').
        quantize_bin: Optional explicit path to llama-quantize binary.
        n_threads: Number of worker threads.

    Returns:
        dict with quantization metadata (duration, sizes, ratio).
    """
    inp_p = Path(input_gguf).resolve()
    out_p = Path(output_gguf).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not is_valid_gguf_file(inp_p):
        raise FileNotFoundError(f"Source GGUF file {inp_p} not found or invalid")

    tmp_p = out_p.with_name(out_p.name + ".part")
    if tmp_p.exists():
        tmp_p.unlink()

    norm_type = quant_type.lower().strip()
    quant_type_str = LLAMA_QUANT_TYPE_NAMES.get(norm_type, norm_type).upper()

    t0 = time.perf_counter()
    bin_path = find_llama_quantize_binary(quantize_bin)

    try:
        if bin_path is not None:
            LOG.info("Quantizing via binary %s: %s -> %s (%s)", bin_path.name, inp_p.name, out_p.name, quant_type_str)
            # llama-quantize positional syntax: llama-quantize [flags] input output quant_type [nthreads]
            cmd = [str(bin_path), str(inp_p), str(tmp_p), quant_type_str]
            if n_threads:
                cmd.append(str(int(n_threads)))
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                LOG.error("llama-quantize binary failed (code %d):\n%s", res.returncode, res.stderr or res.stdout)
                raise RuntimeError(f"llama-quantize failed: {res.stderr or res.stdout}")
            backend_used = f"binary ({bin_path.name})"
        else:
            # Use llama_cpp Python C-API directly
            try:
                import llama_cpp
            except ImportError as exc:
                raise GGUFQuantizeUnavailable("llama_cpp module is required for in-process GGUF quantization") from exc

            LOG.info("Quantizing via llama_cpp C-API: %s -> %s (%s)", inp_p.name, out_p.name, quant_type_str)
            ftype_val = get_llama_ftype_enum(norm_type)
            params = llama_cpp.llama_model_quantize_default_params()
            params.ftype = ftype_val
            if n_threads:
                params.nthread = int(n_threads)

            inp_bytes = str(inp_p).encode("utf-8")
            tmp_bytes = str(tmp_p).encode("utf-8")

            ret = llama_cpp.llama_model_quantize(inp_bytes, tmp_bytes, ctypes.byref(params))
            if ret != 0:
                raise RuntimeError(f"llama_model_quantize returned non-zero code {ret} for {out_p}")
            backend_used = "llama_cpp C-API"

        duration = time.perf_counter() - t0
        if not is_valid_gguf_file(tmp_p):
            raise RuntimeError(f"Quantization finished but temporary file {tmp_p} is missing or invalid")

        os.replace(tmp_p, out_p)
    finally:
        if tmp_p.exists():
            try:
                tmp_p.unlink()
            except Exception:
                pass

    inp_size_mb = gguf_file_size_mb(inp_p)
    out_size_mb = gguf_file_size_mb(out_p)
    ratio = round(out_size_mb / inp_size_mb, 4) if inp_size_mb > 0 else 0.0

    LOG.info(
        "Quantized %s (%s): %.2f MB -> %.2f MB (ratio: %.1f%%) in %.2fs [%s]",
        out_p.name,
        quant_type_str,
        inp_size_mb,
        out_size_mb,
        ratio * 100,
        duration,
        backend_used,
    )
    return {
        "status": "success",
        "method": norm_type,
        "backend": backend_used,
        "input_path": str(inp_p),
        "output_path": str(out_p),
        "input_size_mb": inp_size_mb,
        "output_size_mb": out_size_mb,
        "compression_ratio": ratio,
        "quantize_s": round(duration, 2),
    }


# ---------------------------------------------------------------------------
# Task 7.2 — Inference & Logit Extraction via llama-cpp-python
# ---------------------------------------------------------------------------

def load_gguf_model(
    model_path: str | Path,
    n_ctx: int = 512,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
    verbose: bool = False,
    logits_all: bool = True,
    **kwargs: Any,
) -> Any:
    """Load GGUF model into a llama_cpp.Llama instance."""
    from llama_cpp import Llama

    model_p = Path(model_path).resolve()
    if not is_valid_gguf_file(model_p):
        raise FileNotFoundError(f"GGUF model file not found or invalid: {model_p}")

    init_kwargs: dict[str, Any] = {
        "model_path": str(model_p),
        "n_ctx": n_ctx,
        "logits_all": logits_all,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
    }
    if n_threads:
        init_kwargs["n_threads"] = n_threads
    init_kwargs.update(kwargs)

    return Llama(**init_kwargs)


def free_gguf(llm: Any) -> None:
    """Safely close context and free memory used by llama_cpp.Llama instance."""
    if llm is not None:
        try:
            if hasattr(llm, "close"):
                llm.close()
        except Exception:
            pass
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_logits_llamacpp(
    model_or_path: Any,
    dataset: pd.DataFrame,
    verbalizer: dict[str, str] | None = None,
    *,
    split: str = "eval",
    max_len: int = 256,
    template: str = DEFAULT_PROMPT_TEMPLATE,
    model_id: str = "",
    n_ctx: int = 512,
    n_gpu_layers: int = -1,
    n_threads: int | None = None,
    progress_every: int = 50,
    should_stop: Callable[[], bool] | None = None,
    tokenizer: Any = None,
    hate_token_id: int | None = None,
    nonhate_token_id: int | None = None,
) -> pd.DataFrame:
    """Extract next-token label logits from GGUF model using llama-cpp-python.

    Label-token probability = softmax over the two verbalizer token logits.
    Returns the canonical logits dataframe with columns:
    [example_id, split, y_true, logit_hate, logit_nonhate, prob_hate,
     prob_nonhate, prediction, confidence, correct, margin, entropy].

    Args:
        model_or_path: Either a path to .gguf file or an existing Llama instance.
        dataset: DataFrame containing columns 'text', 'label', 'example_id'.
        verbalizer: Dict mapping 'hate' and 'nonhate' to tokens (e.g. {'hate': ' A', 'nonhate': ' B'}).
        split: Split name for provenance ('id_val', 'id_test', 'ood_test').
        max_len: Max sequence length for prompt. Over-long prompts are left-truncated.
        template: Prompt template string with {text} placeholder.
        model_id: Optional identifier string to record in dataframe.
        n_ctx: Context window size for Llama.
        n_gpu_layers: Number of layers to offload to GPU (-1 for all).
        n_threads: CPU thread count.
        progress_every: Log progress every N examples.
        should_stop: Graceful cancellation predicate.
        tokenizer: Optional HF tokenizer used to resolve verbalizer token IDs.
        hate_token_id: Optional explicit token ID for hate.
        nonhate_token_id: Optional explicit token ID for non-hate.
    """
    from llama_cpp import Llama

    v = verbalizer or DEFAULT_VERBALIZER
    hate_str = v["hate"]
    nonhate_str = v["nonhate"]

    owns_model = False
    if isinstance(model_or_path, (str, Path)):
        llm = load_gguf_model(
            model_or_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
            logits_all=True,
        )
        owns_model = True
    else:
        llm = model_or_path

    try:
        # Resolve verbalizer token IDs
        if hate_token_id is None or nonhate_token_id is None:
            if tokenizer is not None:
                hate_token_id = encode_label_token(tokenizer, hate_str)
                nonhate_token_id = encode_label_token(tokenizer, nonhate_str)
            else:
                tok_hate = llm.tokenize(hate_str.encode("utf-8"), add_bos=False)
                tok_nonhate = llm.tokenize(nonhate_str.encode("utf-8"), add_bos=False)
                if not tok_hate or not tok_nonhate:
                    raise ValueError(f"Could not tokenize verbalizers: hate={hate_str!r}, nonhate={nonhate_str!r}")
                hate_token_id = tok_hate[-1]
                nonhate_token_id = tok_nonhate[-1]

        if hate_token_id == nonhate_token_id:
            raise ValueError(f"Verbalizer tokens collide on id {hate_token_id}")

        n_examples = len(dataset)
        example_ids: list[str] = []
        y_true_list: list[int] = []
        logit_hate_list: list[float] = []
        logit_nonhate_list: list[float] = []

        t_start = time.perf_counter()
        for idx, row in enumerate(dataset.itertuples(index=False)):
            if should_stop and should_stop():
                LOG.warning("Logit extraction stopped early at index %d / %d", idx, n_examples)
                break

            ex_id = str(getattr(row, "example_id", idx))
            raw_text = str(getattr(row, "text", ""))
            y_label = int(getattr(row, "label", 0))

            prompt = build_prompt(raw_text, template)
            tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
            if len(tokens) > max_len:
                tokens = tokens[-max_len:]

            llm.reset()
            llm.eval(tokens)

            # Answer position is the last token evaluated: n_tokens - 1
            ans_pos = llm.n_tokens - 1
            z_hate = float(llm.scores[ans_pos, hate_token_id])
            z_nonhate = float(llm.scores[ans_pos, nonhate_token_id])

            example_ids.append(ex_id)
            y_true_list.append(y_label)
            logit_hate_list.append(z_hate)
            logit_nonhate_list.append(z_nonhate)

            if progress_every > 0 and (idx + 1) % progress_every == 0:
                elapsed = time.perf_counter() - t_start
                rate = (idx + 1) / elapsed
                LOG.info(
                    "[%s] Evaluated %d / %d examples (%.1f ex/s)",
                    split,
                    idx + 1,
                    n_examples,
                    rate,
                )

        frame = probs_to_frame(
            example_ids=example_ids,
            split=split,
            y_true=y_true_list,
            logit_hate=np.asarray(logit_hate_list, dtype=np.float64),
            logit_nonhate=np.asarray(logit_nonhate_list, dtype=np.float64),
            extra={"model_id": model_id} if model_id else None,
        )
        return frame

    finally:
        if owns_model:
            free_gguf(llm)


# ---------------------------------------------------------------------------
# Validation Gate T7.5: GGUF F16 vs HF BF16 Reference Check
# ---------------------------------------------------------------------------

def sanity_check_gguf_f16(
    fp_logits: pd.DataFrame,
    gguf_f16_logits: pd.DataFrame,
    tolerance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Validation Gate T7.5: GGUF F16 approximately matches HF BF16 reference.

    Pass condition: Agreement >= 90% on common subset. Numerics differ slightly
    due to FP16 vs BF16 precision and backend kernel implementations.

    Args:
        fp_logits: DataFrame of baseline BF16 logits (Phase 4/5).
        gguf_f16_logits: DataFrame of GGUF F16 logits.
        tolerance: Tolerance thresholds dict (default: min_agreement=0.90, max_f1_drop=0.05).

    Returns:
        dict with validation results and metrics.
    """
    if tolerance is None:
        tolerance = {"min_agreement": 0.90, "max_f1_drop": 0.05, "max_mean_delta_p": 0.10}

    if "example_id" not in fp_logits.columns or "example_id" not in gguf_f16_logits.columns:
        raise ValueError("Both fp_logits and gguf_f16_logits must contain 'example_id' column.")

    fp_df = fp_logits.copy()
    gguf_df = gguf_f16_logits.copy()
    fp_df["example_id"] = fp_df["example_id"].astype(str)
    gguf_df["example_id"] = gguf_df["example_id"].astype(str)

    aligned = pd.merge(fp_df, gguf_df, on="example_id", suffixes=("_fp", "_gguf"))
    if aligned.empty:
        raise ValueError("No overlapping example_ids between FP reference and GGUF logits.")

    y_true_col = "y_true_fp" if "y_true_fp" in aligned.columns else "y_true"
    y_true = aligned[y_true_col].to_numpy(dtype=np.int64)

    fp_pred = aligned["prediction_fp"].to_numpy(dtype=np.int64)
    gguf_pred = aligned["prediction_gguf"].to_numpy(dtype=np.int64)

    prob_fp = aligned["prob_hate_fp"].to_numpy(dtype=np.float64)
    prob_gguf = aligned["prob_hate_gguf"].to_numpy(dtype=np.float64)

    single_class = bool(np.unique(y_true).size < 2)
    fp_f1 = float(f1_score(y_true, fp_pred, average="macro", zero_division=0)) if not single_class else None
    gguf_f1 = float(f1_score(y_true, gguf_pred, average="macro", zero_division=0)) if not single_class else None

    agreement = float((fp_pred == gguf_pred).mean())
    mean_delta_p = float(np.abs(prob_fp - prob_gguf).mean())
    max_delta_p = float(np.abs(prob_fp - prob_gguf).max())
    f1_drop = (fp_f1 - gguf_f1) if (fp_f1 is not None and gguf_f1 is not None) else None

    checks = {
        "agreement_ok": bool(agreement >= tolerance.get("min_agreement", 0.90)),
        "f1_drop_ok": None if single_class else bool(f1_drop <= tolerance.get("max_f1_drop", 0.05)),
        "delta_p_ok": bool(mean_delta_p <= tolerance.get("max_mean_delta_p", 0.10)),
    }

    passed = all(v for v in checks.values() if v is not None)
    return {
        "passed": passed,
        "n_samples": len(aligned),
        "single_class": single_class,
        "fp_macro_f1": fp_f1,
        "gguf_macro_f1": gguf_f1,
        "f1_drop": f1_drop,
        "agreement": agreement,
        "mean_delta_p": mean_delta_p,
        "max_delta_p": max_delta_p,
        "tolerance": tolerance,
        "checks": checks,
    }

