"""LoRA fine-tuning for instruction-format binary hate-speech classification.

The task is cast as single-token verbalizer completion: the prompt ends with
``"Answer:"`` and the model must emit ``" A"`` (hate) or ``" B"`` (non-hate).
Training masks every prompt token and supervises the single label token, so the
next-token distribution at the answer position is a calibrated two-way
classifier (see :mod:`src.models.logit_extraction`).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LOG = logging.getLogger("slm-calib.finetune")

DEFAULT_PROMPT_TEMPLATE = (
    "Classify the following text as hate speech or not.\n"
    "Text: {text}\n"
    "Answer:"
)

# The answer token must be a *single* token for the two-way softmax read-out.
# " A" / " B" are single Qwen-family BPE tokens (verified by assert_single_token).
DEFAULT_VERBALIZER: dict[str, str] = {"hate": " A", "nonhate": " B"}


@dataclass
class SFTExample:
    example_id: str
    prompt: str
    label: int
    domain: str = ""
    language: str = "bn"


def build_prompt(text: str, template: str = DEFAULT_PROMPT_TEMPLATE) -> str:
    return template.format(text=" ".join(str(text).split()))


def frame_to_examples(df: pd.DataFrame, template: str = DEFAULT_PROMPT_TEMPLATE) -> list[SFTExample]:
    """Convert a processed dataframe with ``text``/``label``/``example_id``."""
    out: list[SFTExample] = []
    for row in df.itertuples(index=False):
        out.append(
            SFTExample(
                example_id=str(getattr(row, "example_id")),
                prompt=build_prompt(row.text, template),
                label=int(row.label),
                domain=str(getattr(row, "domain", "")),
                language=str(getattr(row, "language", "bn")),
            )
        )
    return out


class VerbalizedSFTDataset(Dataset):
    """Prompt + single label token, with prompt tokens masked out of the loss."""

    def __init__(
        self,
        examples: Iterable[SFTExample],
        tokenizer: Any,
        max_len: int = 256,
        template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.template = template
        self.label_token_ids = label_token_ids(tokenizer, DEFAULT_VERBALIZER)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        answer_ids = [self.label_token_ids[int(ex.label)]]
        # Truncate the prompt from the LEFT so the answer slot is always kept.
        budget = self.max_len - len(answer_ids)
        if len(prompt_ids) > budget:
            prompt_ids = prompt_ids[:1] + prompt_ids[-(budget - 1) :]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
            "example_id": ex.example_id,
            "label": int(ex.label),
        }


class SFTCollator:
    """Right-pad a batch and pad labels with the model's ignore index."""

    def __init__(self, pad_token_id: int, label_pad_token_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "example_id": [f["example_id"] for f in features],
            "label": [f["label"] for f in features],
        }
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [self.label_pad_token_id] * pad)
        out = {
            k: (torch.tensor(v, dtype=torch.long) if k != "example_id" else v)
            for k, v in batch.items()
        }
        out["labels"][out["attention_mask"] == 0] = self.label_pad_token_id
        return out


# ---------------------------------------------------------------------------
# Verbalizer helpers
# ---------------------------------------------------------------------------

def encode_label_token(tokenizer: Any, token: str) -> int:
    """Return the id of the *first* token of ``token``, asserting it is single."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Verbalizer token {token!r} maps to {len(ids)} tokens {ids}; "
            "the two-way read-out requires single-token labels."
        )
    return int(ids[0])


def label_token_ids(tokenizer: Any, verbalizer: dict[str, str] | None = None) -> dict[int, int]:
    """Map class index (0 = non-hate, 1 = hate) to token id."""
    v = verbalizer or DEFAULT_VERBALIZER
    return {0: encode_label_token(tokenizer, v["nonhate"]), 1: encode_label_token(tokenizer, v["hate"])}


def assert_single_token_verbalizer(tokenizer: Any, verbalizer: dict[str, str] | None = None) -> dict[str, int]:
    """T-style guard: verbalizer tokens must be distinct and single-token."""
    v = verbalizer or DEFAULT_VERBALIZER
    hate = encode_label_token(tokenizer, v["hate"])
    nonhate = encode_label_token(tokenizer, v["nonhate"])
    if hate == nonhate:
        raise ValueError(f"Verbalizer tokens collide: {v['hate']!r} and {v['nonhate']!r} -> {hate}")
    return {"hate": hate, "nonhate": nonhate}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class FinetuneResult:
    adapter_path: str
    merged_path: str
    history: list[dict[str, float]] = field(default_factory=list)
    best_eval_loss: float | None = None
    best_checkpoint: str | None = None
    n_train: int = 0
    n_id_val: int = 0
    loss_decreased: bool = False
    final_train_loss: float | None = None
    peak_vram_gb: float | None = None
    wall_s: float | None = None
    trainable_params: int = 0
    total_params: int = 0
    label_token_ids: dict[str, int] = field(default_factory=dict)
    max_seq_len: int = 0


def load_base_model(
    hf_id: str,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
    attn_implementation: str | None = None,
):
    """Load a causal LM in the requested dtype.

    ``attn_implementation`` must be ``"eager"`` for Gemma-2: its sliding-window
    attention and logit soft-capping are not handled correctly by the fused sdpa /
    flash kernels, and transformers warns about this at load time.
    """
    from transformers import AutoModelForCausalLM

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[dtype.lower()]

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if device_map:
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.config.use_cache = False
    return model


def load_tokenizer(hf_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def finetune_lora(config: dict[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame) -> FinetuneResult:
    """LoRA fine-tune a causal LM with completion-only loss.

    ``config`` keys (mirrors ``configs/pilot/default.yaml``)::

        model.hf_id, model.dtype, model.verbalizer.{hate,nonhate}
        train.{rank,alpha,dropout,target_modules,lr,epochs,batch_size,grad_accum,
               warmup_ratio,weight_decay,grad_checkpointing,max_grad_norm}
        data.max_len, data.prompt_template
        experiment.{name,seed,output_dir}

    Returns a :class:`FinetuneResult`; the merged model is written to
    ``<output_dir>/merged``.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from src.metrics.efficiency import Stopwatch
    from transformers import Trainer, TrainingArguments, set_seed as hf_set_seed

    model_cfg = config["model"]
    train_cfg = config["train"]
    data_cfg = config.get("data", {})
    exp_cfg = config["experiment"]

    seed = int(exp_cfg["seed"])
    hf_set_seed(seed)

    output_dir = Path(exp_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_len = int(data_cfg.get("max_len", 256))
    template = str(data_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE))
    verbalizer = dict(model_cfg.get("verbalizer", DEFAULT_VERBALIZER))

    sw = Stopwatch("finetune")
    with sw:
        tokenizer = load_tokenizer(model_cfg["hf_id"])
        token_ids = assert_single_token_verbalizer(tokenizer, verbalizer)
        LOG.info("Verbalizer token ids: %s", token_ids)

        model = load_base_model(
            model_cfg["hf_id"],
            str(model_cfg.get("dtype", "bfloat16")),
            attn_implementation=model_cfg.get("attn_implementation") or None,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        lora_config = LoraConfig(
            r=int(train_cfg.get("rank", 16)),
            lora_alpha=int(train_cfg.get("alpha", 32)),
            lora_dropout=float(train_cfg.get("dropout", 0.05)),
            target_modules=list(train_cfg["target_modules"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        model.print_trainable_parameters()
        if bool(train_cfg.get("grad_checkpointing", True)):
            # Without this the checkpointed blocks receive inputs that do not require
            # grad, no graph is built, and backward() raises "element 0 of tensors does
            # not require grad and does not have a grad_fn".
            model.enable_input_require_grads()

        train_ds = VerbalizedSFTDataset(
            frame_to_examples(train_df, template), tokenizer, max_len=max_len, template=template
        )
        val_ds = VerbalizedSFTDataset(
            frame_to_examples(val_df, template), tokenizer, max_len=max_len, template=template
        )
        collator = SFTCollator(pad_token_id=tokenizer.pad_token_id)

        epochs = float(train_cfg.get("epochs", 1))
        batch_size = int(train_cfg.get("batch_size", 2))
        grad_accum = int(train_cfg.get("grad_accum", 8))
        # A short run must still emit loss points: with logging_steps larger than the
        # number of optimiser updates, log_history contains no "loss" entry at all.
        steps_per_epoch = max(1, math.ceil(len(train_ds) / (batch_size * grad_accum)))
        total_updates = max(1, int(steps_per_epoch * epochs))
        configured_logging = int(train_cfg.get("logging_steps", 10))
        logging_steps = max(1, min(configured_logging, total_updates // 5 or 1))
        args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 4)),
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=bool(train_cfg.get("grad_checkpointing", True)),
            learning_rate=float(train_cfg["lr"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            lr_scheduler_type=str(train_cfg.get("lr_scheduler", "cosine")),
            logging_steps=logging_steps,
            save_strategy=str(train_cfg.get("save_strategy", "epoch")),
            save_total_limit=int(train_cfg.get("save_total_limit", 2)),
            eval_strategy=str(train_cfg.get("eval_strategy", "epoch")),
            load_best_model_at_end=True,
            metric_for_best_model=str(train_cfg.get("metric_for_best_model", "eval_loss")),
            greater_is_better=False,
            bf16=bool(train_cfg.get("bf16", True)),
            seed=seed,
            data_seed=seed,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=int(train_cfg.get("dataloader_workers", 0)),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collator,
        )
        train_out = trainer.train()
        metrics = trainer.evaluate()
        trainer.save_model(str(output_dir / "adapter"))
        tokenizer.save_pretrained(str(output_dir / "adapter"))

        history = [
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in step.items()}
            for step in trainer.state.log_history
        ]
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)

        # Merge adapter into the base weights -> standalone model for PTQ.
        merged_dir = output_dir / "merged"
        merged = trainer.model.merge_and_unload()
        merged.config.use_cache = True
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

    train_losses = [h["loss"] for h in history if "loss" in h]
    loss_decreased = bool(
        len(train_losses) >= 2 and train_losses[-1] < train_losses[0] and not any(
            not math.isfinite(v) for v in train_losses
        )
    )
    (output_dir / "train_metrics.json").write_text(
        json.dumps(
            {
                "train_runtime_s": float(getattr(train_out.metrics, "train_runtime", sw.wall_s)),
                "train_samples_per_second": float(
                    getattr(train_out.metrics, "train_samples_per_second", 0.0) or 0.0
                ),
                "eval": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                "log_history": history,
                "label_token_ids": token_ids,
                "best_checkpoint": best_ckpt,
                "total_updates": total_updates,
                "logging_steps": logging_steps,
                "peak_vram_gib": sw.peak_vram_gb,
                "wall_s": sw.wall_s,
                "energy_wh": sw.energy_wh,
                "trainable_params": trainable,
                "total_params": total,
                "n_train": len(train_ds),
                "n_id_val": len(val_ds),
                "max_seq_len": max_len,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return FinetuneResult(
        adapter_path=str(output_dir / "adapter"),
        merged_path=str(merged_dir),
        history=history,
        best_eval_loss=float(metrics.get("eval_loss")) if metrics.get("eval_loss") is not None else None,
        n_train=len(train_ds),
        n_id_val=len(val_ds),
        loss_decreased=loss_decreased,
        best_checkpoint=str(best_ckpt) if best_ckpt else None,
        final_train_loss=train_losses[-1] if train_losses else None,
        peak_vram_gb=sw.peak_vram_gb,
        wall_s=sw.wall_s,
        trainable_params=int(trainable),
        total_params=int(total),
        label_token_ids=token_ids,
        max_seq_len=max_len,
    )


def default_target_modules(hf_id: str) -> list[str]:
    """Standard LoRA target set for Qwen/Llama/Gemma decoder blocks."""
    name = str(hf_id).lower()
    if "qwen" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "llama" in name or "smol" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "gemma" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj"]


def subsample(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """Deterministic stratified subsample (stratified on ``label``) used to cap pilot cost.

    Allocation uses the largest-remainder method so the class ratio is preserved
    and the row count sums *exactly* to ``n``.  Written without ``groupby.apply``
    because recent pandas releases drop the grouping column from the result.
    """
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    n = int(n)
    if "label" not in df.columns:
        raise ValueError("subsample() requires a 'label' column for stratification")
    if n < int(df["label"].nunique()):
        # Too few rows to keep every class; stratification is impossible.
        return df.sample(n=n, random_state=seed).reset_index(drop=True)

    counts = df["label"].value_counts().sort_index()
    exact = counts.to_numpy(dtype=np.float64) * (n / len(df))
    frac = exact - np.floor(exact)
    shares = np.clip(np.floor(exact).astype(np.int64), 1, counts.to_numpy())

    order = np.argsort(-frac, kind="stable")
    guard = 0
    while shares.sum() < n and guard < 10 * n:
        for i in order:
            if shares[i] < counts.to_numpy()[i]:
                shares[i] += 1
                break
        guard += 1
    guard = 0
    while shares.sum() > n and guard < 10 * n:
        for i in order[::-1]:
            if shares[i] > 1:
                shares[i] -= 1
                break
        guard += 1

    picked = pd.concat(
        [
            df[df["label"] == label].sample(int(k), random_state=seed)
            for label, k in zip(counts.index, shares)
            if k > 0
        ],
        ignore_index=True,
    )
    return picked.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def set_hf_env(gradient_checkpointing: bool = True) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def frame_hash(df: pd.DataFrame, cols: list[str] | None = None) -> str:
    """Content hash of a dataframe slice (recorded in run manifests)."""
    import hashlib

    use = cols or [c for c in ("example_id", "label") if c in df.columns]
    payload = df[use].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def numpy_labels(df: pd.DataFrame) -> np.ndarray:
    return df["label"].to_numpy(dtype=np.int64)