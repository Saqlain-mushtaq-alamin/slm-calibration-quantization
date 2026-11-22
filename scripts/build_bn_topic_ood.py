# scripts/build_bn_topic_ood.py
import re, glob
import pandas as pd
from datasketch import MinHash, MinHashLSH

norm = lambda s: re.sub(r"\s+", " ", str(s).lower().strip())
KEEP = ["crime", "religion", "sports"]

def mh(t, num_perm=128):
    m = MinHash(num_perm=num_perm)
    grams = [t[i:i+3] for i in range(len(t)-2)] if len(t) >= 3 else [t]
    for g in grams:
        m.update(g.encode("utf-8"))
    return m

a = pd.read_csv(glob.glob("data/raw/bengali_hs_30k/*.csv")[0])
b = pd.read_csv("data/processed/bd_shs_processed.csv")
a["k"] = a["sentence"].map(norm); b["k"] = b["text"].map(norm)
a = a.drop_duplicates("k")
a = a[~a["k"].isin(set(b["k"]))]                      # exact overlap
a = a[a["category"].isin(KEEP)].reset_index(drop=True)

lsh = MinHashLSH(threshold=0.8, num_perm=128)         # index BD-SHS
for i, t in enumerate(b["k"].drop_duplicates()):
    lsh.insert(str(i), mh(t))
keep = [not lsh.query(mh(t)) for t in a["k"]]         # near overlap
print("removed near-dups:", len(a) - sum(keep))
a = a[keep].reset_index(drop=True)

out = pd.DataFrame({
    "example_id": [f"bn30k_ood_{i}" for i in range(len(a))],
    "text": a["sentence"].astype(str),
    "label": a["hate"].astype(int),
    "domain": a["category"],
    "language": "bn",
})
out.to_csv("data/processed/bn_topic_ood.csv", index=False)
print(len(out), "rows; hate rate", round(out["label"].mean(), 3))
print(pd.crosstab(out["domain"], out["label"]))