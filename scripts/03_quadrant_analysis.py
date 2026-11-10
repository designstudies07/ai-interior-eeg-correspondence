"""How often does the EEG change land in the intended Russell quadrant? Compared with chance by within-person permutation.

absolute: sign of change from baseline (valence: FAA difference; arousal: relative change) matches the target quadrant
relative: sign of the deviation from the person's own mean over the four images matches the target quadrant
          (removes the common eye-opening shift; equivalent to asking whether the image was relatively more pleasant/arousing)
Chance: category labels permuted within each participant (10,000 permutations); one-sided p = share of permutations with at least
as many hits as observed. Only participants with all four images valid on both indices are used.
Input: data/participant_indices.csv. Output: results/quadrant_correspondence.json (Table 2 of the manuscript).
"""
import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, load_indices

d = load_indices()
d["group"] = d["label"].map({"E": "A", "G": "B", "I": "C", "K": "D"})
CAT = ["LVLA", "HVLA", "LVHA", "HVHA"]
TV = np.array([-1, 1, -1, 1])
TA = np.array([-1, -1, 1, 1])
ok_b = d.valid_Previous >= 15
V = np.full((len(d), 4), np.nan); A = np.full((len(d), 4), np.nan)
for j, k in enumerate("1234"):
    ok = ok_b & (d[f"valid_{k}"] >= 15)
    V[:, j] = np.where(ok, d[f"val_{k}"] - d.val_Previous, np.nan)
    A[:, j] = np.where(ok, (d[f"aro_{k}"] - d.aro_Previous) / d.aro_Previous, np.nan)
keep = ~np.isnan(V).any(1) & ~np.isnan(A).any(1)
V, A, grp = V[keep], A[keep], d.group.to_numpy()[keep]
n = len(V)
rng = np.random.default_rng(20261110)
NPERM = 10000
perms = np.array([rng.permutation(4) for _ in range(NPERM * n)]).reshape(NPERM, n, 4)
out = {"n_participants": int(n), "by_group_n": pd.Series(grp).value_counts().sort_index().to_dict()}


def hits_matrix(v, a):
    return (np.sign(v) == TV) & (np.sign(a) == TA)


for mode in ("absolute", "relative"):
    v = V if mode == "absolute" else V - V.mean(1, keepdims=True)
    a = A if mode == "absolute" else A - A.mean(1, keepdims=True)
    H = hits_matrix(v, a)  # n x 4
    idx = np.arange(n)[None, :, None]
    Hp = hits_matrix(v[idx, perms], a[idx, perms])  # NPERM x n x 4 (values shuffled across categories)
    res = {"overall": {}, "by_category": {}, "by_group": {}, "by_group_category": {}, "per_person": {}}
    obs, null = H.sum(), Hp.sum((1, 2))
    res["overall"] = {"hits": int(obs), "of": int(H.size), "rate": round(float(obs / H.size), 3),
                      "chance_rate": round(float(null.mean() / H.size), 3), "p": float((null >= obs).mean())}
    for j, c in enumerate(CAT):
        o, nl = H[:, j].sum(), Hp[:, :, j].sum(1)
        res["by_category"][c] = {"hits": int(o), "of": int(n), "rate": round(float(o / n), 3),
                                 "chance_rate": round(float(nl.mean() / n), 3), "p": float((nl >= o).mean())}
    for g in "ABCD":
        m = grp == g
        o, nl = H[m].sum(), Hp[:, m].sum((1, 2))
        res["by_group"][g] = {"hits": int(o), "of": int(H[m].size), "rate": round(float(o / H[m].size), 3),
                              "chance_rate": round(float(nl.mean() / H[m].size), 3), "p": float((nl >= o).mean())}
        for j, c in enumerate(CAT):
            o2 = H[m, j].sum()
            res["by_group_category"][f"{g}_{c}"] = {"hits": int(o2), "of": int(m.sum())}
    pp = H.sum(1)
    res["per_person"] = {"distribution": pd.Series(pp).value_counts().sort_index().to_dict(),
                         "chance_distribution": (pd.Series(Hp.sum(2).ravel()).value_counts(normalize=True).sort_index() * n).round(1).to_dict()}
    out[mode] = res
    print(f"\n===== {mode} =====")
    print("overall:", res["overall"])
    print("by category:"); [print("  ", c, r) for c, r in res["by_category"].items()]
    print("by group:"); [print("  ", g, r) for g, r in res["by_group"].items()]
    gc = pd.DataFrame({k: v["hits"] for k, v in res["by_group_category"].items()}, index=[0]).T
    print("group x category hits:"); print(pd.DataFrame([[res["by_group_category"][f"{g}_{c}"]["hits"] for c in CAT] for g in "ABCD"],
                                                    index=list("ABCD"), columns=CAT).to_string())
    print("hits per person (0-4):", res["per_person"]["distribution"], "| expected by chance:", res["per_person"]["chance_distribution"])
json.dump(out, open(ROOT / "results" / "quadrant_correspondence.json", "w"), indent=1, default=int)

