"""Significance <-> attention in the goal-shelving diary (Mayer & Freund data in the archive). Written by Claude.
Claims checked (stated before running this script; related archive estimates had been seen, so this is an
independent re-estimation with stricter controls, not a blind preregistration):
 C1a  attention follows significance: shelved goal's value at t-1 predicts thoughts about it at t (+)
 C1b  significance follows attention: thoughts at t-1 predict the goal's value at t (+)
 C1c  pursuits compete for finite attention: the PRIORITY goal's value at t-1 predicts FEWER thoughts
      about the shelved goal at t (-)
Model: within-person (person fixed effects) lag-1 regressions with the outcome's own lag and time,
person-clustered CR1 SEs; half-panel jackknife (first vs second half of each person's series) as a
Nickell-bias check; effects also in within-person SD units.
Usage: python3 -I attention_test.py <diarydt.csv> <out.json>
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv(sys.argv[1]).sort_values(["id", "time"])
cols = ["s_value", "s_thoughts", "p_value"]
for c in cols:
    d["l_" + c] = d.groupby("id")[c].shift(1)
d["l_time"] = d.groupby("id")["time"].shift(1)
pairs = d.dropna(subset=cols + ["l_" + c for c in cols]).copy()
pairs["gap"] = pairs.time - pairs.l_time
pairs = pairs[pairs.gap > 0]

def cluster_fe(df, y, xs):
    Xd = df[xs] - df.groupby("id")[xs].transform("mean")
    yd = df[y] - df.groupby("id")[y].transform("mean")
    X = Xd.to_numpy(float); yv = yd.to_numpy(float); g = pd.factorize(df.id)[0]; G = g.max() + 1
    n, k = X.shape
    XtX = np.linalg.pinv(X.T @ X); b = XtX @ X.T @ yv; e = yv - X @ b
    S = np.zeros((G, k)); np.add.at(S, g, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX @ (S.T @ S) @ XtX
    se = np.sqrt(np.diag(V))
    return {x: {"b": float(b[i]), "lo": float(b[i] - 1.96 * se[i]), "hi": float(b[i] + 1.96 * se[i]),
                "p": float(2 * stats.norm.sf(abs(b[i] / se[i])))} for i, x in enumerate(xs)}

def wsd(c):  # within-person SD
    return float((pairs[c] - pairs.groupby("id")[c].transform("mean")).std())

res = {"pairs": len(pairs), "people": int(pairs.id.nunique()), "median_gap": float(pairs.gap.median())}
specs = {
    "C1a_thoughts_from_value": ("s_thoughts", ["l_s_value", "l_s_thoughts", "l_p_value", "time"], "l_s_value"),
    "C1b_value_from_thoughts": ("s_value", ["l_s_thoughts", "l_s_value", "l_p_value", "time"], "l_s_thoughts"),
    "C1c_thoughts_from_priority_value": ("s_thoughts", ["l_s_value", "l_s_thoughts", "l_p_value", "time"], "l_p_value"),
}
# halves for jackknife: split each person's pairs at their median time
pairs["half"] = pairs.groupby("id")["time"].transform(lambda t: (t > t.median()).astype(int))
for name, (y, xs, key) in specs.items():
    full = cluster_fe(pairs, y, xs)
    h0 = cluster_fe(pairs[pairs.half == 0], y, xs); h1 = cluster_fe(pairs[pairs.half == 1], y, xs)
    jk = 2 * full[key]["b"] - 0.5 * (h0[key]["b"] + h1[key]["b"])
    scale = wsd(key[2:]) / wsd(y)
    res[name] = {"estimate": full[key], "jackknife_bias_corrected": jk,
                 "standardized_within": full[key]["b"] * scale, "all_terms": full}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or "estimate" not in v else
                      {"b": round(v["estimate"]["b"], 4), "ci": [round(v["estimate"]["lo"], 4), round(v["estimate"]["hi"], 4)],
                       "p": round(v["estimate"]["p"], 5), "jk": round(v["jackknife_bias_corrected"], 4),
                       "std": round(v["standardized_within"], 3)}) for k, v in res.items()}, indent=1))
