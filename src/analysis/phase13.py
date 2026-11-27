"""Phase 13 hypothesis contrasts over frozen Phase 8–12 artifacts."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.analysis.statistical_tests import (
    apply_correction, cliffs_delta, cohens_d, paired_bootstrap_delta,
    tost_equivalence,
)

TEST_SETS = ("id_test", "ood_test")
QUANT_METHODS = ("bnb_int8", "bnb_nf4", "gptq_int4", "q8_0", "q4_k_m", "q3_k_m")
ECE_MARGIN = .02
F1_MARGIN = .01


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"Phase 13 input missing: {path}")
    return pd.read_parquet(path)


def prepare_pairs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pair every quantized row to the same examples of its BF16 run."""
    needed = {"artifact", "base_run", "phase", "method", "model", "seed", "lang", "loco",
              "eval_set", "sample_fingerprint", "n_samples", "ece_equal_mass_15", "brier", "nll", "macro_f1"}
    if needed - set(metrics):
        raise ValueError(f"Phase 8 metrics missing {sorted(needed - set(metrics))}")
    data = metrics[(metrics.phase != "pilot") & metrics.eval_set.isin(TEST_SETS)].copy()
    if data.duplicated(["artifact", "eval_set"]).any():
        raise ValueError("Duplicate artifact/eval-set metrics")
    keys = ["base_run", "model", "seed", "lang", "loco", "eval_set"]
    fp = data[data.method == "bf16"]
    if fp.duplicated(keys).any():
        raise ValueError("Ambiguous BF16 references")
    variants = data[data.method.isin(QUANT_METHODS)].copy()
    cols = keys + ["artifact", "sample_fingerprint", "n_samples", "ece_equal_mass_15", "brier", "nll", "macro_f1"]
    joined = variants.merge(fp[cols], on=keys, how="left", suffixes=("", "_fp"),
                            validate="many_to_one", indicator=True)
    if (joined._merge != "both").any():
        raise ValueError("Quantized artifact missing its BF16 reference")
    for col in ("sample_fingerprint", "n_samples"):
        if (joined[col] != joined[f"{col}_fp"]).any():
            raise ValueError(f"BF16 pairing mismatch: {col}")
    for metric in ("ece_equal_mass_15", "brier", "nll", "macro_f1"):
        joined[f"delta_{metric}"] = joined[metric] - joined[f"{metric}_fp"]
    return joined.drop(columns="_merge")


def _model_values(frame: pd.DataFrame, value: str) -> np.ndarray:
    """One inference unit per model family, averaged over paired seeds."""
    if frame.empty:
        raise ValueError("No paired model rows")
    if frame.duplicated(["model", "seed"]).any():
        raise ValueError("Duplicate model/seed within contrast")
    means = frame.groupby("model", sort=True)[value].mean()
    if not np.isfinite(means.to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite model-family values for {value}")
    return means.to_numpy(dtype=float)


def _contrast(hypothesis: str, contrast: str, values: np.ndarray, *, method: str = "",
              language: str = "", domain: str = "", metric: str = "",
              n_runs: int = 0, margin: float | None = None, status: str = "tested",
              note: str = "", n_boot: int = 10000, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=float)
    zeros = np.zeros_like(values)
    boot = paired_bootstrap_delta(zeros, values, n_resamples=n_boot, seed=seed)
    tost = tost_equivalence(zeros, values, margin) if margin is not None else None
    return {"hypothesis": hypothesis, "contrast": contrast, "method": method,
            "language": language, "domain": domain, "metric": metric,
            "n_models": len(values), "n_model_seed_runs": n_runs,
            "mean_delta": boot["observed_delta"], "ci_lower": boot["ci_lower"],
            "ci_upper": boot["ci_upper"], "p_value": boot["p_value"],
            "cohens_d_paired": cohens_d(zeros, values),
            "cliffs_delta_descriptive": cliffs_delta(zeros, values),
            "tost_margin": margin, "p_tost": None if tost is None else tost["p_tost"],
            "status": status, "note": note}


def _paired_did(pairs: pd.DataFrame, factor: str, left: str, right: str, metric: str) -> pd.DataFrame:
    """Return right−left within identical model, seed, method and other axis."""
    keys = ["model", "seed", "method", "lang" if factor == "eval_set" else "eval_set"]
    source = pairs[pairs[factor].isin([left, right])]
    if source.duplicated(keys + [factor]).any():
        raise ValueError(f"Duplicate {factor} interaction cells")
    wide = source.pivot(index=keys, columns=factor, values=metric)
    if wide[[left, right]].isna().any().any():
        raise ValueError(f"Missing paired {factor} interaction cells")
    out = wide.reset_index()
    out["did"] = out[right] - out[left]
    return out


def _mixed_interaction(metrics: pd.DataFrame, *, factor: str) -> dict:
    """Supplementary mixed model for GPTQ vs BF16; primary DID is above."""
    data = metrics[(metrics.phase != "pilot") & metrics.method.isin(["bf16", "gptq_int4"])
                   & metrics.eval_set.isin(TEST_SETS)].copy()
    data["domain"] = data.eval_set.map({"id_test": "id", "ood_test": "ood"})
    data["precision"] = data.method
    if factor == "domain":
        formula = "ece_equal_mass_15 ~ C(precision, Treatment('bf16')) * C(domain, Treatment('id')) + C(lang)"
        term_parts = ("gptq_int4", "ood")
    else:
        data = data[data.domain == "id"]
        formula = "ece_equal_mass_15 ~ C(precision, Treatment('bf16')) * C(lang, Treatment('en'))"
        term_parts = ("gptq_int4", "bn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = smf.mixedlm(formula, data=data, groups=data["model"],
                              re_formula="1", vc_formula={"seed": "0+C(seed)"}).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            terms = [name for name in fit.params.index if ":" in name and all(part in name for part in term_parts)]
            if len(terms) != 1:
                raise RuntimeError(f"Interaction term not uniquely found: {terms}")
            name = terms[0]
            ci = fit.conf_int().loc[name]
            return {"factor": factor, "method": "gptq_int4", "converged": bool(fit.converged),
                    "coefficient": float(fit.params[name]), "ci_lower": float(ci.iloc[0]),
                    "ci_upper": float(ci.iloc[1]), "p_value": float(fit.pvalues[name]),
                    "n_observations": len(data), "n_models": int(data.model.nunique()),
                    "warning": "; ".join(sorted({str(w.message) for w in caught})), "error": None}
        except Exception as exc:
            return {"factor": factor, "method": "gptq_int4", "converged": False,
                    "coefficient": None, "ci_lower": None, "ci_upper": None, "p_value": None,
                    "n_observations": len(data), "n_models": int(data.model.nunique()),
                    "warning": "; ".join(sorted({str(w.message) for w in caught})),
                    "error": f"{type(exc).__name__}: {exc}"}


def run_phase13(results_dir: Path, *, n_boot: int = 10000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metrics = _read(results_dir, "all_metrics.parquet")
    selective = _read(results_dir, "selective_metrics.parquet")
    recal = _read(results_dir, "recalibration_results.parquet")
    efficiency = _read(results_dir, "efficiency_metrics.parquet")
    ablation = _read(results_dir, "ablation_results.parquet")
    pairs = prepare_pairs(metrics)
    rows = []

    # H1: separate INT8 equivalence, INT4 calibration changes, and the
    # relative calibration-versus-F1 degradation on ID tests.
    id_pairs = pairs[pairs.eval_set == "id_test"]
    for (method, lang), group in id_pairs.groupby(["method", "lang"]):
        for metric in ("ece_equal_mass_15", "brier", "nll"):
            values = _model_values(group, f"delta_{metric}")
            rows.append(_contrast("H1", "quantized minus BF16", values, method=method,
                                  language=lang, domain="id", metric=metric, n_runs=len(group),
                                  margin=ECE_MARGIN if metric == "ece_equal_mass_15" else None,
                                  n_boot=n_boot, seed=seed))
        f1 = _model_values(group, "delta_macro_f1")
        rows.append(_contrast("H1", "quantized minus BF16", f1, method=method,
                              language=lang, domain="id", metric="macro_f1", n_runs=len(group),
                              margin=F1_MARGIN, n_boot=n_boot, seed=seed))
        if method in {"bnb_nf4", "gptq_int4", "q4_k_m", "q3_k_m"}:
            rel = group["delta_ece_equal_mass_15"] / group["ece_equal_mass_15_fp"].clip(lower=1e-6)
            rel += group["delta_macro_f1"] / group["macro_f1_fp"].clip(lower=1e-6)
            derived = group[["model", "seed"]].copy()
            derived["relative_gap"] = rel
            rows.append(_contrast("H1", "relative ECE change minus relative F1 loss",
                                  _model_values(derived, "relative_gap"), method=method,
                                  language=lang, domain="id", metric="relative_gap", n_runs=len(group),
                                  n_boot=n_boot, seed=seed))

    # H2: quantization-induced ECE difference under OOD vs ID, paired by run.
    domain_did = _paired_did(pairs, "eval_set", "id_test", "ood_test", "delta_ece_equal_mass_15")
    for (method, lang), group in domain_did.groupby(["method", "lang"]):
        rows.append(_contrast("H2", "OOD minus ID quantization effect",
                              _model_values(group, "did"), method=method, language=lang,
                              domain="ood_minus_id", metric="ece_equal_mass_15",
                              n_runs=len(group), n_boot=n_boot, seed=seed))

    # H3: Bengali minus English degradation on paired model/seed/precision.
    language_did = _paired_did(id_pairs, "lang", "en", "bn", "delta_ece_equal_mass_15")
    for method, group in language_did.groupby("method"):
        rows.append(_contrast("H3", "Bengali minus English quantization effect",
                              _model_values(group, "did"), method=method, domain="id",
                              metric="ece_equal_mass_15", n_runs=len(group), n_boot=n_boot, seed=seed))

    # H4: E-AURC subtracts the optimal risk ranking for the same error count;
    # it is a more direct confidence-ranking contrast than raw AURC.
    sel = selective[(selective.phase != "pilot") & selective.eval_set.isin(TEST_SETS)].copy()
    sel_keys = ["base_run", "model", "seed", "lang", "loco", "eval_set"]
    fp_sel = sel[sel.method == "bf16"]
    if fp_sel.duplicated(sel_keys).any():
        raise ValueError("Ambiguous BF16 selective references")
    sel = sel[sel.method.isin(QUANT_METHODS)].merge(
        fp_sel[sel_keys + ["sample_fingerprint", "e_aurc", "aurc"]], on=sel_keys,
        suffixes=("", "_fp"), how="left", validate="many_to_one", indicator=True)
    if (sel._merge != "both").any() or (sel.sample_fingerprint != sel.sample_fingerprint_fp).any():
        raise ValueError("Selective examples are not paired to BF16")
    sel["delta_e_aurc"] = sel.e_aurc - sel.e_aurc_fp
    for (method, lang, eval_set), group in sel.groupby(["method", "lang", "eval_set"]):
        rows.append(_contrast("H4", "quantized minus BF16 E-AURC",
                              _model_values(group, "delta_e_aurc"), method=method, language=lang,
                              domain=eval_set, metric="e_aurc", n_runs=len(group), n_boot=n_boot, seed=seed))

    # H5: own-fit vs raw BF16 and transfer vs own-fit. Oracle comparison is
    # explicitly unavailable in the current frozen data.
    rec = recal[(recal.phase != "pilot") & recal.eval_set.isin(TEST_SETS)].copy()
    if rec.duplicated(["artifact", "eval_set", "arm"]).any():
        raise ValueError("Duplicate recalibration arm")
    fp_raw = rec[(rec.method == "bf16") & (rec.arm == "none")]
    key = ["base_run", "model", "seed", "lang", "loco", "eval_set"]
    for arm in ("T_own", "T_fp"):
        block = rec[(rec.arm == arm) & rec.method.isin(QUANT_METHODS)]
        if (block.fit_split != "id_val").any():
            raise ValueError("Non-oracle temperature fitted outside ID validation")
        joined = block.merge(fp_raw[key + ["ece_equal_mass_15", "sample_fingerprint"]],
                             on=key, suffixes=("", "_fp"), validate="many_to_one", how="left", indicator=True)
        if (joined._merge != "both").any() or (joined.sample_fingerprint != joined.sample_fingerprint_fp).any():
            raise ValueError("Recalibration and BF16 samples do not align")
        joined["residual_ece"] = joined.ece_equal_mass_15 - joined.ece_equal_mass_15_fp
        if arm == "T_own":
            for (method, lang), group in joined[joined.eval_set == "id_test"].groupby(["method", "lang"]):
                rows.append(_contrast("H5a", "own-fit ECE minus raw BF16 ECE",
                                      _model_values(group, "residual_ece"), method=method, language=lang,
                                      domain="id", metric="ece_equal_mass_15", n_runs=len(group),
                                      margin=ECE_MARGIN, n_boot=n_boot, seed=seed))
    own = rec[(rec.arm == "T_own") & rec.method.isin(QUANT_METHODS)]
    transferred = rec[(rec.arm == "T_fp") & rec.method.isin(QUANT_METHODS)]
    arm_keys = ["artifact", "base_run", "model", "seed", "lang", "loco", "eval_set", "sample_fingerprint"]
    transfer = transferred.merge(own[arm_keys + ["ece_equal_mass_15"]], on=arm_keys,
                                 suffixes=("", "_own"), how="inner", validate="one_to_one")
    if len(transfer) != len(own):
        raise ValueError("Missing own/FP-transfer arm pairing")
    transfer["transfer_gap"] = transfer.ece_equal_mass_15 - transfer.ece_equal_mass_15_own
    for (method, lang, eval_set), group in transfer.groupby(["method", "lang", "eval_set"]):
        rows.append(_contrast("H5b", "FP-transfer minus own-fit ECE",
                              _model_values(group, "transfer_gap"), method=method, language=lang,
                              domain=eval_set, metric="ece_equal_mass_15", n_runs=len(group),
                              margin=ECE_MARGIN, status="partial", note="Oracle OOD-validation arm unavailable",
                              n_boot=n_boot, seed=seed))

    # H6: measured peak-VRAM compression and calibration equivalence are
    # necessary evidence; Phase 14 Pareto frontiers remain a separate gate.
    eff = efficiency[efficiency.method.isin(["bf16", "bnb_int8"])].copy()
    eff["lang"] = eff.language
    eff_keys = ["base_run", "model", "seed", "lang"]
    fp_eff = eff[eff.method == "bf16"]
    int8_eff = eff[eff.method == "bnb_int8"]
    eff_join = int8_eff.merge(fp_eff[eff_keys + ["peak_vram_mb"]], on=eff_keys,
                             suffixes=("", "_fp"), how="left", validate="one_to_one", indicator=True)
    if (eff_join._merge != "both").any() or (eff_join.peak_vram_mb <= 0).any():
        raise ValueError("Missing or invalid INT8/BF16 efficiency pairing")
    eff_join["memory_ratio"] = eff_join.peak_vram_mb_fp / eff_join.peak_vram_mb
    for lang, group in eff_join.groupby("lang"):
        rows.append(_contrast("H6", "BF16 divided by INT8 peak VRAM minus 2",
                              _model_values(group.assign(ratio_minus_two=group.memory_ratio - 2), "ratio_minus_two"),
                              method="bnb_int8", language=lang, metric="memory_ratio_minus_two",
                              n_runs=len(group), status="partial", note="Pareto frontier awaits Phase 14",
                              n_boot=n_boot, seed=seed))
    # Reuse H1 INT8 ECE TOST as calibration component; do not duplicate tests.

    # H7: paired Bengali vs English quantizer calibration text for GPTQ only.
    a1 = ablation[(ablation.ablation_id == "A1") & (ablation.method == "gptq_int4")].copy()
    if a1.duplicated(["artifact", "eval_set", "variant"]).any():
        raise ValueError("Duplicate A1 calibration variant")
    for eval_set, block in a1.groupby("eval_set"):
        wide = block.pivot(index=["model", "seed", "artifact"], columns="variant", values="ece_equal_mass_15")
        if not {"bn", "en", "mixed", "random"} <= set(wide) or wide[["bn", "en", "mixed", "random"]].isna().any().any():
            raise ValueError("A1 lacks one of four GPTQ calibration variants")
        group = wide.reset_index()
        group["bn_minus_en"] = group["bn"] - group["en"]
        rows.append(_contrast("H7", "Bengali minus English GPTQ calibration ECE",
                              _model_values(group, "bn_minus_en"), method="gptq_int4", language="bn",
                              domain=eval_set, metric="ece_equal_mass_15", n_runs=len(group),
                              status="partial", note="AWQ calibration variants unavailable; only two model families",
                              n_boot=n_boot, seed=seed))

    table = pd.DataFrame(rows)
    if not {"H1", "H2", "H3", "H4", "H5a", "H5b", "H6", "H7"} <= set(table.hypothesis):
        raise ValueError("One or more hypotheses have no statistical row")
    table["p_holm"] = apply_correction(table.p_value.tolist())
    table["significant_holm_05"] = table.p_holm < .05
    table["p_tost_holm"] = np.nan
    mask = table.p_tost.notna()
    table.loc[mask, "p_tost_holm"] = apply_correction(table.loc[mask, "p_tost"].tolist())
    table["equivalent_holm_05"] = table.p_tost_holm < .05
    interactions = pd.DataFrame([_mixed_interaction(metrics, factor="domain"),
                                 _mixed_interaction(metrics, factor="language")])
    summary = {"n_tests": len(table), "n_models_per_language": int(pairs.model.nunique()),
               "hypotheses": sorted(set(table.hypothesis)),
               "mixed_models_converged": int(interactions.converged.sum()),
               "mixed_models_total": len(interactions),
               "limitations": ["AWQ unavailable in Phase 12", "No OOD-validation oracle temperature",
                               "Phase 14 Pareto frontier not yet computed",
                               "OOD targets can be single-class and shift class prior"]}
    return table, interactions, summary
