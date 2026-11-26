"""Interrupted Phase 12 cells reuse verified quantization, training and splits."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.models.logit_extraction import probs_to_frame


SPEC = importlib.util.spec_from_file_location("phase12_runner", Path(__file__).resolve().parents[1] / "scripts/08_ablations.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _frames(prefix="x"):
    train = pd.DataFrame({"example_id": [f"{prefix}t{i}" for i in range(8)],
                          "text": [f"train {i}" for i in range(8)], "label": [0, 1] * 4})
    test = pd.DataFrame({"example_id": [f"{prefix}e{i}" for i in range(2)],
                         "text": ["test 0", "test 1"], "label": [0, 1]})
    return {"train": train, "id_val": test.copy(), "id_test": test.copy(), "ood_test": test.copy()}


def test_evaluation_resumes_only_missing_or_corrupt_split(tmp_path, monkeypatch):
    frames = _frames()
    cfg = {"eval_sets": ["id_test", "ood_test"], "verbalizer": {"hate": " A", "nonhate": " B"},
           "eval_batch_size": 2, "max_len": 32}
    calls = []

    def fake_extract(model, tokenizer, expected, **kwargs):
        calls.append(kwargs["split"])
        return probs_to_frame(expected.example_id.tolist(), kwargs["split"], expected.label.tolist(),
                              np.array([-2., 2.]), np.zeros(2))

    monkeypatch.setattr(runner, "extract_logits_hf", fake_extract)
    params = dict(cfg=cfg, template="{text}", ablation="A2", base="bn_m_lora_bf16_s42",
                  method="label_token", variant="direct", split_sha="sha",
                  logits_root=tmp_path / "logits", smoke=False)
    runner._evaluate(object(), object(), frames, **params)
    runner._evaluate(object(), object(), frames, **params)
    assert calls == ["id_test", "ood_test"]
    eval_dir = params["logits_root"] / "A2" / params["base"] / "label_token/direct"
    bad = pd.read_parquet(eval_dir / "id_test.parquet")
    bad.loc[0, "prob_hate"] = 0.7
    bad.to_parquet(eval_dir / "id_test.parquet", index=False)
    (eval_dir / "ood_test.parquet").unlink()
    runner._evaluate(object(), object(), frames, **params)
    assert calls == ["id_test", "ood_test", "id_test", "ood_test"]


def test_completed_training_is_reused_with_matching_plan(tmp_path, monkeypatch):
    from src.models import finetune

    calls = []

    def fake_train(cfg, train, val):
        calls.append(1)
        root = Path(cfg["experiment"]["output_dir"])
        adapter = root / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"valid")
        merged = root / "merged"
        merged.mkdir()
        (merged / ".completed").write_text("ok")
        (merged / "config.json").write_text("{}")
        (merged / "model.safetensors").write_bytes(b"valid")
        (root / "train_metrics.json").write_text(json.dumps({
            "n_train": len(train), "n_id_val": len(val),
            "log_history": [{"loss": 2.}, {"loss": 1.}], "wall_s": 7.0}))
        return SimpleNamespace(adapter_path=str(adapter), merged_path=str(merged),
                               loss_decreased=True, wall_s=7.0)

    monkeypatch.setattr(finetune, "finetune_lora", fake_train)
    cfg = {"model": {"hf_id": "model"}, "train": {"lr": .001},
           "data": {"max_len": 32}, "experiment": {"seed": 42}}
    root = tmp_path / "work"
    dest = root / "models" / "cell"
    first = runner._train_or_resume(cfg, _frames()["train"], _frames()["id_val"],
                                    dest, root, qlora=False, force=False)
    second = runner._train_or_resume(cfg, _frames()["train"], _frames()["id_val"],
                                     dest, root, qlora=False, force=False)
    assert len(calls) == 1 and first.merged_path == second.merged_path
    assert cfg["train"]["save_steps"] == 10


def test_quantized_checkpoint_is_reused_after_missing_cell_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_quant(model_path, prompts, output_path, **kwargs):
        calls.append(1)
        out = Path(output_path)
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"valid")
        return {"quantize_s": 3.0}

    monkeypatch.setattr(runner, "quantize_gptq", fake_quant)
    monkeypatch.setattr(runner, "load_gptq", lambda path: (object(), object()))
    monkeypatch.setattr(runner, "_evaluate", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "_release", lambda model: None)
    cfg = {"calibration_samples": 8, "max_len": 32,
           "prompt_templates": {"bn": {"baseline": "{text}"}}}
    cell = ("A1", "bn", "m", 42, "bn_m_lora_bf16_s42", "gptq_int4", "bn", tmp_path / "merged")
    cache = {"bn": (_frames("b"), "sha-b"), "en": (_frames("e"), "sha-e")}
    root = tmp_path / "work"
    runner._run_cell(cell, cfg, cache=cache, output_root=root, smoke=True)
    runner._run_cell(cell, cfg, cache=cache, output_root=root, smoke=True)
    assert len(calls) == 1
