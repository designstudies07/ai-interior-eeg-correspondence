"""Sensitivity analysis: smallest effects detectable with 80% power, given the analysed sample sizes (manuscript Section 2.12).

* Within-participant contrasts (one-sample t test on participant-level contrast scores, two-sided alpha = .05): dz.
* Between-platform comparisons (independent groups, two-sided alpha = .05): Cohen's d and the corresponding eta-squared,
  d^2 / (d^2 + 4), for the smallest and largest per-comparison sample sizes.
* Intended-quadrant analysis (one-sided alpha = .05, binomial approximation to the permutation test): the hit rate that would be
  detected with 80% power against the chance rate.

Power for the t tests uses the normal approximation to the noncentral t distribution (adequate for df > 40).
Input: results/results.json (script 02) and results/quadrant_correspondence.json (script 03).
Output: results/sensitivity.json.
"""
import json

from scipy.optimize import brentq
from scipy.stats import binom, norm, t

from common import ROOT


def power_t(d, df, ncp_scale, alpha=.05):
    tc = t.ppf(1 - alpha / 2, df)
    ncp = d * ncp_scale
    return norm.sf(tc - ncp) + norm.cdf(-tc - ncp)


def mde_paired(n):
    return brentq(lambda d: power_t(d, n - 1, n ** .5) - .8, .05, 1.5)


def mde_independent(n1, n2):
    return brentq(lambda d: power_t(d, n1 + n2 - 2, (n1 * n2 / (n1 + n2)) ** .5) - .8, .05, 1.5)


R = json.load(open(ROOT / "results" / "results.json", encoding="utf-8"))
Q = json.load(open(ROOT / "results" / "quadrant_correspondence.json", encoding="utf-8"))

out = {}
nv, na = R["anova"]["valence"]["complete_n"], R["anova"]["arousal"]["complete_n"]
out["within"] = {"n_valence": nv, "dz_valence": mde_paired(nv), "n_arousal": na, "dz_arousal": mde_paired(na)}

n1s, n2s = [r["n_mj"] for r in R["table2"]], [r["n_kl"] for r in R["table2"]]
d_lo, d_hi = mde_independent(max(n1s), max(n2s)), mde_independent(min(n1s), min(n2s))
out["between"] = {"n_mj": [min(n1s), max(n1s)], "n_kl": [min(n2s), max(n2s)], "d_range": [d_lo, d_hi],
                  "eta2_range": [d_lo ** 2 / (d_lo ** 2 + 4), d_hi ** 2 / (d_hi ** 2 + 4)]}

o = Q["relative"]["overall"]
n, p0 = o["of"], o["chance_rate"]
crit = binom.ppf(.95, n, p0)
rate = next(p / 1000 for p in range(int(p0 * 1000), 1000) if binom.sf(crit, n, p / 1000) >= .8)
out["quadrant"] = {"N": n, "chance": p0, "detectable_rate": rate}

(ROOT / "results" / "sensitivity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
