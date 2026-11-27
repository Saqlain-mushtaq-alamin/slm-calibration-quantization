"""Phase 16 focused tests using frozen splits and a local tiny encoder fixture."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from src.models.encoder import (
    EncodedDataset,
    checkpoint_valid,
    evaluate_and_save,
    latest_checkpoint,
    load_config,
    load_frozen,
    logits_frame,
    prepare_text,
    quantize_dynamic_int8,
)


class TinyTokenizer:
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": [2, 3, 4], "attention_mask": [1, 1, 1]}
        ids = torch.tensor([[2, 3, 4] for _ in text])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def tiny_model():
    return BertForSequenceClassification(BertConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_labels=2,
    ))


@pytest.mark.parametrize("name", ["banglabert", "xlmr"])
def test_configs_and_frozen_integrity(name):
    cfg = load_config(f"configs/encoder_{name}.yaml")
    frames, sha = load_frozen(cfg)
    assert len(sha) == 64
    assert all(set(f.label.astype(int)) <= {0, 1} for f in frames.values())
    ids = [set(f.example_id.astype(str)) for f in frames.values()]
    assert all(not a & b for i, a in enumerate(ids) for b in ids[i + 1:])
    assert cfg["model_id"] == ("csebuetnlp/banglabert" if name == "banglabert" else "FacebookAI/xlm-roberta-base")


def test_tokenization_labels_and_preprocessing():
    frame = pd.DataFrame({"example_id": ["a"], "text": ["  HELLO   World  "], "label": [1]})
    ds = EncodedDataset(frame, TinyTokenizer(), "en", 32)
    assert ds[0]["labels"] == 1
    assert prepare_text(frame.text.iloc[0], "en") == "hello world"
    with pytest.raises(ValueError):
        prepare_text("x", "invalid")


def test_forward_backward_checkpoint_resume_int8_and_output(tmp_path):
    torch.manual_seed(42)
    model = tiny_model()
    rows = pd.DataFrame({"example_id": [f"e{i}" for i in range(8)],
                         "text": ["a"] * 8, "label": [i % 2 for i in range(8)]})
    tokenizer = TinyTokenizer()
    ds = EncodedDataset(rows, tokenizer, "en", 32)

    def collate(items):
        return {key: torch.tensor([item[key] for item in items]) for key in items[0]}

    def trainer(max_steps, model):
        args = TrainingArguments(output_dir=str(tmp_path / "checkpoints"),
                                 per_device_train_batch_size=2, max_steps=max_steps,
                                 save_strategy="steps", save_steps=1,
                                 save_total_limit=4, report_to="none", seed=42)
        return Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)

    first = trainer(2, model)
    first.train()
    ckpt = latest_checkpoint(tmp_path / "checkpoints")
    assert ckpt and checkpoint_valid(ckpt)
    step = json.loads((ckpt / "trainer_state.json").read_text())["global_step"]
    assert step == 2
    broken = tmp_path / "checkpoints" / "checkpoint-99"
    broken.mkdir()
    (broken / "trainer_state.json").write_text("{")
    assert latest_checkpoint(tmp_path / "checkpoints") == ckpt
    second = trainer(3, tiny_model())
    second.train(resume_from_checkpoint=str(ckpt))
    assert second.state.global_step == 3
    assert latest_checkpoint(tmp_path / "checkpoints") != ckpt

    cfg = {"eval_batch_size": 2, "max_length": 32, "language": "en", "fold": "fixture"}
    fp, _ = logits_frame(second.model.cpu(), tokenizer, rows, cfg)
    qmodel, modules = quantize_dynamic_int8(second.model)
    int8, _ = logits_frame(qmodel, tokenizer, rows, cfg)
    assert modules and any("classifier" in x for x in modules)
    for pred in (fp, int8):
        assert pred.example_id.tolist() == rows.example_id.tolist()
        assert pred.y_true.tolist() == rows.label.tolist()
        assert pred[["logit_nonhate", "logit_hate", "prob_nonhate", "prob_hate"]].shape == (8, 4)
        assert np.isfinite(pred[["logit_nonhate", "logit_hate", "prob_nonhate", "prob_hate"]]).all().all()
        assert set(pred.prediction) <= {0, 1}
        assert np.allclose(pred.prob_nonhate + pred.prob_hate, 1)

    frames = {part: rows for part in ("id_val", "id_test", "ood_test")}
    metrics = evaluate_and_save(qmodel, tokenizer, frames, cfg, "fixture_sha", "int8", tmp_path)
    assert all(math.isfinite(metrics[part][key]) for part in metrics for key in
               ("accuracy", "macro_f1", "macro_precision", "macro_recall", "ece", "brier", "nll", "aurc"))
    assert (tmp_path / "metrics" / "int8_id_test.json").exists()
    assert (tmp_path / "predictions" / "int8_id_test.csv").exists()
