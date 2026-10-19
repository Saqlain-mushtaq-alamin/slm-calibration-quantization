"""Leave-one-context-out (LOCO) splits, hashing, and integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def _class_priors(series: pd.Series) -> dict[str, float]:
    vc = series.value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def create_loco_splits(
    df: pd.DataFrame,
    domain_col: str = "domain",
    *,
    seed: int = 42,
    min_domain_size: int = 200,
    id_col: str = "example_id",
    train_frac: float = 0.7,
) -> dict[str, Any]:
    """Create LOCO split definitions.

    For each domain D with enough samples:
      - source = all rows where domain != D
      - target = rows where domain == D  (ood_test)
      - source is stratified into train / id_val / id_test

    Returns dict keyed by ``loco_<D>`` with example-ID lists and class priors.
    """
    if domain_col not in df.columns:
        raise ValueError(f"domain_col {domain_col!r} not in dataframe")
    if id_col not in df.columns:
        df = df.copy()
        df[id_col] = [f"row_{i}" for i in range(len(df))]

    domain_sizes = df[domain_col].astype(str).value_counts()
    domains = sorted([d for d, n in domain_sizes.items() if n >= min_domain_size])
    if len(domains) < 2:
        raise ValueError(
            f"Need >=2 domains with size>={min_domain_size}; got {dict(domain_sizes)}"
        )

    splits: dict[str, Any] = {}
    for held_out in domains:
        source = df[df[domain_col].astype(str) != held_out].copy()
        target = df[df[domain_col].astype(str) == held_out].copy()
        if len(source) < 10 or len(target) < 5:
            continue

        strat = source["label"] if source["label"].nunique() > 1 else None
        try:
            train, temp = train_test_split(
                source,
                test_size=1.0 - train_frac,
                stratify=strat,
                random_state=seed,
            )
        except ValueError:
            train, temp = train_test_split(
                source, test_size=1.0 - train_frac, random_state=seed
            )
        strat_temp = temp["label"] if temp["label"].nunique() > 1 else None
        val_frac_of_temp = 1.0 / 3.0  # temp is 30% → val 10%, test 20%
        try:
            id_val, id_test = train_test_split(
                temp,
                test_size=1.0 - val_frac_of_temp,
                stratify=strat_temp,
                random_state=seed,
            )
        except ValueError:
            id_val, id_test = train_test_split(
                temp, test_size=1.0 - val_frac_of_temp, random_state=seed
            )

        train_ids = train[id_col].astype(str).tolist()
        id_val_ids = id_val[id_col].astype(str).tolist()
        id_test_ids = id_test[id_col].astype(str).tolist()
        ood_test_ids = target[id_col].astype(str).tolist()

        splits[f"loco_{held_out}"] = {
            "target_domain": held_out,
            "source_domains": [d for d in domains if d != held_out],
            "train_ids": train_ids,
            "id_val_ids": id_val_ids,
            "id_test_ids": id_test_ids,
            "ood_test_ids": ood_test_ids,
            "class_priors": {
                "train": _class_priors(train["label"]),
                "id_val": _class_priors(id_val["label"]),
                "id_test": _class_priors(id_test["label"]),
                "ood_test": _class_priors(target["label"]),
            },
            "sizes": {
                "train": len(train_ids),
                "id_val": len(id_val_ids),
                "id_test": len(id_test_ids),
                "ood_test": len(ood_test_ids),
            },
            "seed": seed,
            "domain_col": domain_col,
        }
    if not splits:
        raise ValueError("No LOCO splits were produced")
    return splits


def freeze_and_hash_splits(splits: dict[str, Any], output_path: str | Path) -> str:
    """Write splits JSON deterministically and return SHA-256 of content."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys for byte-stable hashing regardless of insertion order
    content = json.dumps(splits, sort_keys=True, indent=2, ensure_ascii=False)
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    output_path.write_text(content, encoding="utf-8")
    sha_path = Path(str(output_path) + ".sha256")
    sha_path.write_text(sha + "\n", encoding="utf-8")
    print(f"Splits saved to {output_path} | SHA-256: {sha}")
    return sha


def load_splits(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return json.loads(content), sha


def verify_split_integrity(splits: dict[str, Any]) -> dict[str, Any]:
    """T1.5 / T1.7 checks: disjoint IDs, coverage, no target leakage."""
    report: dict[str, Any] = {"ok": True, "splits": {}}
    for name, spec in splits.items():
        train = set(spec["train_ids"])
        id_val = set(spec["id_val_ids"])
        id_test = set(spec["id_test_ids"])
        ood = set(spec["ood_test_ids"])

        issues: list[str] = []
        if train & id_val:
            issues.append("train∩id_val")
        if train & id_test:
            issues.append("train∩id_test")
        if train & ood:
            issues.append("train∩ood_test")
        if id_val & id_test:
            issues.append("id_val∩id_test")
        if id_val & ood:
            issues.append("id_val∩ood_test")
        if id_test & ood:
            issues.append("id_test∩ood_test")
        source_union = train | id_val | id_test
        # source union should be disjoint from ood (already checked pairwise)
        entry = {
            "issues": issues,
            "n_train": len(train),
            "n_id_val": len(id_val),
            "n_id_test": len(id_test),
            "n_ood_test": len(ood),
            "source_covers_only_source": not (source_union & ood),
        }
        if issues:
            report["ok"] = False
        report["splits"][name] = entry
    return report
