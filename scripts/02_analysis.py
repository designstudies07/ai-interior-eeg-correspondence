"""Main analyses: platform comparisons (Table 3), within-participant tests of the intended categories (2 x 2 x 4 contrasts),
category means, squared Euclidean distance (Fig. 4), Figures 2-4 and Supplementary Figure S1.
Input: data/participant_indices.csv. Output: results/.

Change scores: valence = FAA(stimulus) - FAA(baseline); arousal = (A(stimulus) - A(baseline)) / A(baseline).
A segment with < 15 valid rows is treated as missing for that participant/category.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import ROOT, load_indices
from scipy import stats

rec = load_indices()
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
ORDER = ["LVLA", "HVLA", "LVHA", "HVHA"]
KEY = {"LVLA": "1", "HVLA": "2", "LVHA": "3", "HVHA": "4"}
GRP = {"E": "A", "G": "B", "I": "C", "K": "D"}
MINROWS = 15
rec["group"] = rec["label"].map(GRP)
rec["platform"] = np.where(rec["group"].isin(["A", "B"]), "MidJourney", "Kling AI")
okb = rec["valid_Previous"] >= MINROWS
for cat, k in KEY.items():
    ok = okb & (rec[f"valid_{k}"] >= MINROWS)
    rec[f"v_{cat}"] = np.where(ok, rec[f"val_{k}"] - rec["val_Previous"], np.nan)
    rec[f"a_{cat}"] = np.where(ok, (rec[f"aro_{k}"] - rec["aro_Previous"]) / rec["aro_Previous"], np.nan)
R = {"n_recordings": int(len(rec)), "groups": rec["group"].value_counts().sort_index().to_dict(),
     "platform_n": rec["platform"].value_counts().to_dict(), "sex": rec.groupby(["group", "sex"]).size().unstack().to_dict(),
     "order": rec.groupby(["group", "order"]).size().unstack().to_dict(), "baseline_short": int((~okb).sum()),
     "missing_cells": {f"{d}_{c}": int(rec[f"{d}_{c}"].isna().sum()) for d in ("v", "a") for c in ORDER},
     "text_cells": int(rec["text_cells_decoded"].sum())}
long = rec.melt(id_vars=["group", "platform", "order"], value_vars=[f"{d}_{c}" for d in ("v", "a") for c in ORDER])
R["share_abs_gt1"] = {d: float((rec[[f"{d}_{c}" for c in ORDER]].abs() > 1).stack().mean()) for d in ("v", "a")}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1, run)
    return adj


# ---------- Table 2: platform comparisons ----------
t2 = []
for cat in ORDER:
    for d, name in (("v", "valence"), ("a", "arousal")):
        mj = rec.loc[rec.platform == "MidJourney", f"{d}_{cat}"].dropna().to_numpy()
        kl = rec.loc[rec.platform == "Kling AI", f"{d}_{cat}"].dropna().to_numpy()
        f, p = stats.f_oneway(mj, kl)
        allv = np.concatenate([mj, kl])
        eta = (len(mj) * (mj.mean() - allv.mean()) ** 2 + len(kl) * (kl.mean() - allv.mean()) ** 2) / ((allv - allv.mean()) ** 2).sum()
        h, ph = stats.kruskal(mj, kl)
        v1, v2 = mj.var(ddof=1) / len(mj), kl.var(ddof=1) / len(kl)
        dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(mj) - 1) + v2 ** 2 / (len(kl) - 1))
        diff = mj.mean() - kl.mean(); tc = stats.t.ppf(.975, dfw); se = math.sqrt(v1 + v2)
        t2.append(dict(category=cat, index=name, n_mj=len(mj), mean_mj=mj.mean(), sd_mj=mj.std(ddof=1), n_kl=len(kl),
                       mean_kl=kl.mean(), sd_kl=kl.std(ddof=1), df2=len(mj) + len(kl) - 2, F=f, p=p, eta2=eta, H=h, p_kw=ph,
                       diff=diff, ci_lo=diff - tc * se, ci_hi=diff + tc * se))
for r, ph in zip(t2, holm([r["p"] for r in t2])):
    r["p_holm"] = ph
R["table2"] = t2


# ---------- direct test of target categories: 2 x 2 (within) x 4 (group) via 1-df contrasts ----------
def ols_tests(y, X_cols, df):
    """y ~ intercept + sum-coded group (+ optional order); returns intercept F and group F."""
    d = df.dropna(subset=[y]).copy()
    G = pd.get_dummies(d["group"]).astype(float)
    Gs = G.iloc[:, :-1].sub(G.iloc[:, -1], axis=0)  # sum coding
    Xs = [np.ones(len(d)), *[Gs[c].to_numpy() for c in Gs]]
    if "order" in X_cols:
        Xs.append(np.where(d["order"].astype(str) == "4321", 0.5, -0.5))
    X = np.column_stack(Xs)
    yv = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    res = yv - X @ beta
    dfe = len(yv) - X.shape[1]
    sse = float(res @ res)
    out = {"n": int(len(yv)), "df_error": int(dfe)}

    def ftest(cols):
        keep = [i for i in range(X.shape[1]) if i not in cols]
        b2, *_ = np.linalg.lstsq(X[:, keep], yv, rcond=None)
        sse_r = float(((yv - X[:, keep] @ b2) ** 2).sum())
        df1 = len(cols)
        F = ((sse_r - sse) / df1) / (sse / dfe)
        return F, df1, float(stats.f.sf(F, df1, dfe)), (sse_r - sse) / (sse_r - sse + sse)
    out["intercept"] = ftest([0])
    out["group"] = ftest(list(range(1, 4)))
    if "order" in X_cols:
        out["order"] = ftest([4])
    out["mean"] = float(np.mean([d.loc[d.group == g, y].mean() for g in "ABCD"]))
    return out


anova = {}
for d, name in (("v", "valence"), ("a", "arousal")):
    sub = rec.copy()
    sub["c_val"] = (sub[f"{d}_HVLA"] + sub[f"{d}_HVHA"] - sub[f"{d}_LVLA"] - sub[f"{d}_LVHA"]) / 2
    sub["c_aro"] = (sub[f"{d}_LVHA"] + sub[f"{d}_HVHA"] - sub[f"{d}_LVLA"] - sub[f"{d}_HVLA"]) / 2
    sub["c_int"] = (sub[f"{d}_LVLA"] + sub[f"{d}_HVHA"] - sub[f"{d}_HVLA"] - sub[f"{d}_LVHA"]) / 2
    sub["grand"] = sub[[f"{d}_{c}" for c in ORDER]].mean(axis=1, skipna=False)
    res = {}
    for c in ("c_val", "c_aro", "c_int", "grand"):
        res[c] = ols_tests(c, ["group"], sub)
        res[c + "_order"] = ols_tests(c, ["group", "order"], sub)
    # between-subject group effect on the participant mean: one-way ANOVA
    gm = [sub.loc[sub.group == g, "grand"].dropna() for g in "ABCD"]
    Fg, pg = stats.f_oneway(*gm)
    res["group_main"] = (float(Fg), 3, float(pg), int(sum(len(x) for x in gm) - 4))
    res["complete_n"] = int(sub[[f"{d}_{c}" for c in ORDER]].notna().all(axis=1).sum())
    anova[name] = res
R["anova"] = anova
# category means (all participants)
R["cat_means"] = {name: {c: [float(rec[f"{d}_{c}"].mean()), float(rec[f"{d}_{c}"].std()), int(rec[f"{d}_{c}"].notna().sum())] for c in ORDER}
                  for d, name in (("v", "valence"), ("a", "arousal"))}

# ---------- Fig. 4 distances (group means vs target coordinates) ----------
TGT = {"LVLA": (-1, -1), "HVLA": (1, -1), "LVHA": (-1, 1), "HVHA": (1, 1)}
f4 = []
for g in "ABCD":
    for cat in ORDER:
        s = rec[rec.group == g]
        v, a = s[f"v_{cat}"].mean(), s[f"a_{cat}"].mean()
        f4.append(dict(group=g, category=cat, v=v, a=a, d2=(v - TGT[cat][0]) ** 2 + (a - TGT[cat][1]) ** 2))
R["fig4"] = f4
json.dump(R, open(OUT / "results.json", "w"), indent=1, default=float)
pd.DataFrame(t2).to_csv(OUT / "table3_platform_comparisons.csv", index=False)


# ---------- figures ----------
SIZE = {"image18.png": (1442, 842), "image19.png": (1442, 842), "image20.png": (1385, 826)}  # pixel sizes of the manuscript figures
NAME = {"image18.png": "fig2_valence_by_platform.png", "image19.png": "fig3_arousal_by_platform.png", "image20.png": "fig4_squared_distance.png"}


def size_of(n):
    return SIZE[n]


MJ_C, KL_C = "#22608f", "#d07a35"
T = pd.DataFrame(t2)
for d, fname, title, ylab in (("valence", "image18.png", "Participant-level valence-related change by platform", "Change in frontal alpha asymmetry"),
                              ("arousal", "image19.png", "Participant-level arousal-related change by platform", "Relative change in β/(α+θ)")):
    w, h = size_of(fname)
    fig, ax = plt.subplots(figsize=(w / 200, h / 200), dpi=200)
    x = np.arange(4)
    rows = T[T["index"] == d].set_index("category").loc[ORDER]
    for off, col, k, lab in ((-0.17, MJ_C, "mj", "MidJourney"), (0.17, KL_C, "kl", "Kling AI")):
        ax.errorbar(x + off, rows[f"mean_{k}"], yerr=rows[f"sd_{k}"], fmt="o", color=col, ecolor=col, elinewidth=2.2,
                    capsize=6, capthick=2.2, markersize=8, label=f"{lab} (mean ± SD)")
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xticks(x, ORDER); ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel("Target affective category"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=13)
    ax.grid(axis="y", color="#eeeeee"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / NAME[fname], dpi=200); plt.close(fig)

w, h = size_of("image20.png")
fig, ax = plt.subplots(figsize=(w / 200, h / 200), dpi=200)
gcol = {"A": "#d99a1e", "B": "#d8673a", "C": "#d33f5c", "D": "#df6bbd"}
lab = {"A": "A: MidJourney–ChatGPT", "B": "B: MidJourney–DeepSeek", "C": "C: Kling AI–ChatGPT", "D": "D: Kling AI–DeepSeek"}
F4 = pd.DataFrame(f4)
for i, g in enumerate("ABCD"):
    vals = F4[F4.group == g].set_index("category").loc[ORDER, "d2"].to_numpy()
    bars = ax.bar(np.arange(4) + (i - 1.5) * 0.2, vals, 0.2, color=gcol[g], label=lab[g])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(np.arange(4), ORDER); ax.set_xlabel("Target affective category")
ax.set_ylabel("Squared Euclidean distance from target"); ax.set_title("Squared distance to target coordinates (lower = closer)", fontsize=13, pad=12)
ax.set_ylim(0, max(3.5, F4["d2"].max() + 0.4)); ax.grid(axis="y", color="#e6e6e6", ls="--"); ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False); ax.legend(title="Group", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(OUT / NAME["image20.png"], dpi=200); plt.close(fig)

# S1: participant-level distributions
fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.4), dpi=300, sharex=True)
rng = np.random.default_rng(20261109)
for ax, d, ttl, yl in zip(axes, ("v", "a"), ("A   Valence-related change (FAA)", "B   Arousal-related relative change"),
                          ("Change from baseline", "Relative change from baseline")):
    for i, cat in enumerate(ORDER):
        for off, plat, col in ((-0.18, "MidJourney", MJ_C), (0.18, "Kling AI", KL_C)):
            v = rec.loc[rec.platform == plat, f"{d}_{cat}"].dropna().to_numpy()
            ax.scatter(i + off + rng.uniform(-0.07, 0.07, len(v)), v, s=10, color=col, alpha=0.45, linewidths=0)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i + off - 0.1, i + off + 0.1], [med, med], color="black", lw=1.8)
            ax.add_patch(plt.Rectangle((i + off - 0.1, q1), 0.2, q3 - q1, fill=False, ec="black", lw=0.9))
            ax.plot(i + off, v.mean(), marker="D", color="white", mec="black", ms=4.5, zorder=5)
            ax.text(i + off, 1.02, f"n={len(v)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="grey", ls="--", lw=0.8)
    ax.set_ylabel(yl); ax.set_title(ttl, loc="left", fontsize=11, fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
axes[1].set_xticks(range(4), ORDER); axes[1].set_xlabel("Target affective category")
hd = [plt.Line2D([], [], marker="o", ls="", color=MJ_C, alpha=.7, label="MidJourney"), plt.Line2D([], [], marker="o", ls="", color=KL_C, alpha=.7, label="Kling AI"),
      plt.Line2D([], [], color="black", lw=1.8, label="Median (box: IQR)"), plt.Line2D([], [], marker="D", ls="", color="white", mec="black", label="Mean")]
fig.legend(handles=hd, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "figS1_participant_distributions.png", dpi=300); plt.close(fig)

# ---------- print summary ----------
print(json.dumps({k: R[k] for k in ("n_recordings", "groups", "platform_n", "baseline_short", "missing_cells", "share_abs_gt1")}, indent=0))
print(pd.DataFrame(t2)[["category", "index", "n_mj", "mean_mj", "sd_mj", "n_kl", "mean_kl", "sd_kl", "F", "p", "eta2", "H", "p_kw", "diff", "ci_lo", "ci_hi", "p_holm"]].round(3).to_string(index=False))
for name, res in anova.items():
    print(f"\n{name}: complete n={res['complete_n']}")
    for c in ("c_val", "c_aro", "c_int", "grand"):
        o = res[c]
        print(f"  {c:6s} mean={o['mean']:+.3f} | within F(1,{o['df_error']})={o['intercept'][0]:.2f} p={o['intercept'][2]:.3f} pe2={o['intercept'][3]:.3f}"
              f" | x group F(3,{o['df_error']})={o['group'][0]:.2f} p={o['group'][2]:.3f}"
              f" | +order: within p={res[c + '_order']['intercept'][2]:.3f}, order F={res[c + '_order']['order'][0]:.2f} p={res[c + '_order']['order'][2]:.3f}")
    print(f"  group main (participant mean): F(3,{res['group_main'][3]})={res['group_main'][0]:.2f} p={res['group_main'][2]:.3f}")
print("\ncategory means:", json.dumps(R["cat_means"], indent=0))
print("fig4:\n", F4.pivot(index="category", columns="group", values="d2").loc[ORDER].round(2).to_string())
print("order by group:", R["order"])

