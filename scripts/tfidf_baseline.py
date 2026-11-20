import json, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def run(name, csv, splits_json, split):
    d = pd.read_csv(csv); d["example_id"] = d["example_id"].astype(str)
    s = json.load(open(splits_json, encoding="utf-8"))[split]
    m = d.set_index("example_id")
    tr, te, ood = (m.loc[s[k]] for k in ("train_ids", "id_test_ids", "ood_test_ids"))
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
    clf = LogisticRegression(max_iter=2000, C=4).fit(vec.fit_transform(tr["text"]), tr["label"])
    for tag, df in (("id_test", te), ("ood_test", ood)):
        p = clf.predict(vec.transform(df["text"]))
        print(name, tag, "n =", len(df), "macroF1 =", round(f1_score(df["label"], p, average="macro"), 4), "acc =", round((p == df["label"]).mean(), 4))

run("bn", "data/processed/bd_shs_processed.csv", "data/splits/bd_shs_loco_splits.json", "loco_slander")
run("en", "data/processed/english_control_matched.csv", "data/splits/english_control_loco_splits.json", "loco_lgbtq")