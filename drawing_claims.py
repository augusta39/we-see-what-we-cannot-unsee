"""Claims read off the author's sketch, tested on archive data. Written by Claude.
D1 (z vs dot size)  Tangible accomplishment does not build significance: earlier 'getting done' predicts later
                    meaning far less than earlier positive feeling does (Gainey EMA, within-person, lag 1).
D2 (x, lifespan)    Revision slows over the lifespan: older participants revise the meaning of their important
                    events less over four years (MainLife, random-intercept ML model, published Model 6 terms).
D3 ('me')           Who you are shapes revision: more open people revise more (same model).
D4 (coming back)    People keep intending to return to set-aside pursuits even as those pursuits matter less
                    (goal diary: within-person time trends of readoption certainty vs value).
Usage: python3 -I drawing_claims.py <archive dir> <out.json>
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats, optimize

A = sys.argv[1]; out = {}

def fe(df, y, xs, g):
    Xd = df[xs] - df.groupby(g)[xs].transform("mean"); yd = df[y] - df.groupby(g)[y].transform("mean")
    X = Xd.to_numpy(float); yv = yd.to_numpy(float); gi = pd.factorize(df[g])[0]; G = gi.max() + 1; n, k = X.shape
    XtX = np.linalg.pinv(X.T @ X); b = XtX @ X.T @ yv; e = yv - X @ b
    S = np.zeros((G, k)); np.add.at(S, gi, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX @ (S.T @ S) @ XtX; se = np.sqrt(np.diag(V))
    return {x: {"b": float(b[i]), "lo": float(b[i] - 1.96 * se[i]), "hi": float(b[i] + 1.96 * se[i]),
                "p": float(2 * stats.norm.sf(abs(b[i] / se[i])))} for i, x in enumerate(xs)}

# D1 Gainey
g = pd.read_csv(A + "/0058_gainey_ts.tsv", sep="\t").sort_values(["id", "day", "beep"])
pos = ["active", "interested", "excited", "strong"]; neg = ["irritable", "upset", "afraid_anxious", "sad"]
g["pa"] = g[pos].mean(axis=1).where(g[pos].notna().all(axis=1)); g["na"] = g[neg].mean(axis=1).where(g[neg].notna().all(axis=1))
for c in ["meaning", "getting_done", "pa", "na", "day", "beep", "time_since_start"]:
    g["l_" + c] = g.groupby("id")[c].shift(1)
g["gap"] = g.time_since_start - g.l_time_since_start
p = g[(g.day == g.l_day) & (g.beep == g.l_beep + 1) & (g.gap > 0) & (g.gap < 12)].dropna(
    subset=["meaning", "l_meaning", "l_getting_done", "l_pa", "l_na", "getting_done", "pa", "na"])
lag = fe(p, "meaning", ["l_getting_done", "l_pa", "l_na", "l_meaning", "gap", "beep"], "id")
now = fe(p, "meaning", ["getting_done", "pa", "na"], "id")
out["D1"] = {"pairs": len(p), "people": int(p.id.nunique()),
             "lagged_getting_done": lag["l_getting_done"], "lagged_positive_feeling": lag["l_pa"],
             "same_moment_getting_done": now["getting_done"], "same_moment_positive_feeling": now["pa"]}

# D2, D3 MainLife random-intercept ML (published Model 6 terms)
m = pd.read_excel(A + "/mainlife_data/Data for MLMs of change in AM.xlsx").dropna(
    subset=["Id", "AM", "Age1_cent", "nNFFI_openness1_cent", "Timesinceevent1_cent", "Negemo1_cent"])
y = m.AM.to_numpy(float); cols = ["Age1_cent", "nNFFI_openness1_cent", "Timesinceevent1_cent", "Negemo1_cent"]
X = np.column_stack([np.ones(len(m))] + [m[c].to_numpy(float) for c in cols])
idx = pd.factorize(m.Id)[0]; G = idx.max() + 1; nj = np.bincount(idx); n = len(y)
def gls(gam):
    w = gam / (1 + gam * nj); Sx = np.zeros((G, X.shape[1])); np.add.at(Sx, idx, X); Sy = np.bincount(idx, weights=y)
    A_ = X.T @ X - (Sx * w[:, None]).T @ Sx; c_ = X.T @ y - (Sx * w[:, None]).T @ Sy; b = np.linalg.solve(A_, c_)
    r = y - X @ b; Sr = np.bincount(idx, weights=r); s2 = (r @ r - np.sum(w * Sr ** 2)) / n
    ll = -0.5 * (n * np.log(2 * np.pi * s2) + np.sum(np.log(1 + gam * nj)) + n); return ll, b, s2, A_
lg = optimize.minimize_scalar(lambda v: -gls(np.exp(v))[0], bounds=(-12, 8), method="bounded").x
ll, b, s2, A_ = gls(np.exp(lg)); se = np.sqrt(np.diag(np.linalg.inv(A_)) * s2)
sd = {c: float(m[c].std()) for c in cols}
out["D2_D3_mainlife"] = {"events": len(m), "people": int(m.Id.nunique()), "outcome_sd": float(m.AM.std()),
    **{c: {"b": float(b[i + 1]), "lo": float(b[i + 1] - 1.96 * se[i + 1]), "hi": float(b[i + 1] + 1.96 * se[i + 1]),
           "p": float(2 * stats.norm.sf(abs(b[i + 1] / se[i + 1]))), "predictor_sd": sd[c]} for i, c in enumerate(cols)}}

# D4 goal diary trends (per 30 time units)
d = pd.read_csv(A + "/diarydt.csv")
d["month"] = d.time / 30.0
res4 = {}
for y_ in ["s_readoption_certainty", "s_value", "s_thoughts"]:
    sub = d.dropna(subset=[y_, "month"])
    r = fe(sub, y_, ["month"], "id")["month"]
    r["mean_level"] = float(sub[y_].mean()); r["share_people_mean_above_50"] = float((sub.groupby("id")[y_].mean() > 50).mean())
    res4[y_] = r
out["D4_goal_diary_per_month"] = res4
out["D4_time_range"] = [float(d.time.min()), float(d.time.max())]
json.dump(out, open(sys.argv[2], "w"), indent=1)
def s(r): return f"{r['b']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] p={r['p']:.3g}"
print("D1 lagged getting_done -> meaning", s(out["D1"]["lagged_getting_done"]), "| lagged PA", s(out["D1"]["lagged_positive_feeling"]))
print("D1 same-moment getting_done", s(out["D1"]["same_moment_getting_done"]), "| same-moment PA", s(out["D1"]["same_moment_positive_feeling"]))
for c in cols: print("ML", c, s(out["D2_D3_mainlife"][c]), "sd", round(sd[c], 3))
print("outcome sd", round(out["D2_D3_mainlife"]["outcome_sd"], 3))
for k, v in res4.items(): print("D4", k, s(v), "mean", round(v["mean_level"], 1), "share>50", round(v["share_people_mean_above_50"], 2))
print("time range", out["D4_time_range"])
