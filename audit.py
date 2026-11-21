"""Independent re-computation of headline numbers in the Pursuits and Perception archive.
Written from scratch (not reusing the archive's scripts). Usage: python3 -I audit.py <archive_dir>
"""
import sys, json
import numpy as np, pandas as pd
from scipy import optimize, stats

A = sys.argv[1]
out = {}

def cluster_ols(y, X, g, names):
    """OLS with CR1 cluster-robust SEs (same small-sample factor as statsmodels)."""
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(g)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    groups = pd.factorize(g)[0]; G = groups.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, groups, X * e[:, None])
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))
    zn = 1.959963984540054
    return {nm: {"b": float(b[i]), "lo": float(b[i] - zn * se[i]), "hi": float(b[i] + zn * se[i]),
                 "p": float(2 * stats.norm.sf(abs(b[i] / se[i])))} for i, nm in enumerate(names)}

def within(df, cols, gcol):
    return df[cols] - df.groupby(gcol)[cols].transform("mean")

# ---------------- Hippocorpus ----------------
d = pd.read_csv(A + "/hcV3-stories.csv")
rec = d[d.memType == "recalled"][["AssignmentId", "WorkerId", "importance"]]
ret = d[d.memType == "retold"][["recAgnPairId", "WorkerId", "importance", "timeSinceEvent"]]
p = ret.merge(rec, left_on="recAgnPairId", right_on="AssignmentId", suffixes=("_t2", "_t1"))
p = p[p.WorkerId_t2.astype(str) == p.WorkerId_t1.astype(str)].dropna(subset=["importance_t1", "importance_t2"])
p["delta"] = p.importance_t2 - p.importance_t1
h = {"retold_rows": int((d.memType == "retold").sum()), "pairs": len(p), "people": int(p.WorkerId_t1.nunique()),
     "pct_increase": 100 * (p.delta > 0).mean(), "pct_same": 100 * (p.delta == 0).mean(),
     "pct_decrease": 100 * (p.delta < 0).mean(), "at_ceiling_t1": int((p.importance_t1 == 5).sum()),
     "mean_change": p.delta.mean()}
# Among those NOT at ceiling, how do changes look? (ceiling-free view)
nc = p[p.importance_t1 < 5]
h["not_at_ceiling"] = {"n": len(nc), "pct_increase": 100 * (nc.delta > 0).mean(),
                       "pct_same": 100 * (nc.delta == 0).mean(), "pct_decrease": 100 * (nc.delta < 0).mean(),
                       "mean_change": nc.delta.mean()}
# Floor-ceiling-symmetric check: among t1==1..4 how much is regression to the mean? correlation t1,t2
h["corr_t1_t2"] = float(np.corrcoef(p.importance_t1, p.importance_t2)[0, 1])
out["hippocorpus"] = h

# ---------------- MainLife ----------------
m = pd.read_excel(A + "/mainlife_data/Data for MLMs of change in AM.xlsx")
m = m.dropna(subset=["Id", "AM", "Timesinceevent1_cent", "Negemo1_cent"])
X = np.column_stack([np.ones(len(m)), m.Timesinceevent1_cent])
ml = {"events": len(m), "participants": int(m.Id.nunique()),
      "cluster_ols_event_age_only": cluster_ols(m.AM, X, m.Id, ["Intercept", "Timesinceevent1_cent"])["Timesinceevent1_cent"]}

def ri_ml(y, X, g):
    """Random-intercept LMM by maximum likelihood (profile over gamma = tau2/sigma2)."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    idx = pd.factorize(np.asarray(g))[0]; G = idx.max() + 1; n = len(y)
    nj = np.bincount(idx)
    def gls(gam):
        w = gam / (1 + gam * nj)  # V^-1 = (I - w J)/sigma2 per group
        Sx = np.zeros((G, X.shape[1])); np.add.at(Sx, idx, X)
        Sy = np.bincount(idx, weights=y)
        XtVX = X.T @ X - (Sx * w[:, None]).T @ Sx
        XtVy = X.T @ y - (Sx * w[:, None]).T @ Sy
        b = np.linalg.solve(XtVX, XtVy)
        r = y - X @ b; Sr = np.bincount(idx, weights=r)
        q = r @ r - np.sum(w * Sr ** 2)
        s2 = q / n
        logdet = np.sum(np.log(1 + gam * nj))
        ll = -0.5 * (n * np.log(2 * np.pi * s2) + logdet + n)
        return ll, b, s2, XtVX
    res = optimize.minimize_scalar(lambda lg: -gls(np.exp(lg))[0], bounds=(-12, 8), method="bounded")
    gam = np.exp(res.x); ll, b, s2, XtVX = gls(gam)
    se = np.sqrt(np.diag(np.linalg.inv(XtVX)) * s2)
    return {"loglik": float(ll), "AIC": float(-2 * ll + 2 * (X.shape[1] + 2)), "tau2": float(gam * s2), "sigma2": float(s2),
            "icc": float(gam / (1 + gam)), "b": b.tolist(), "se": se.tolist()}

ri = {}
ri["null"] = ri_ml(m.AM, np.ones((len(m), 1)), m.Id)
specs = {"model5_published": ["Age1_cent", "nNFFI_openness1_cent", "Timesinceevent1_cent"],
         "model6_published": ["Age1_cent", "nNFFI_openness1_cent", "Timesinceevent1_cent", "Negemo1_cent"],
         "event_age_only": ["Timesinceevent1_cent"]}
for k, cols in specs.items():
    r = ri_ml(m.AM, np.column_stack([np.ones(len(m))] + [m[c] for c in cols]), m.Id)
    r["coef"] = {c: {"b": r["b"][i + 1], "se": r["se"][i + 1]} for i, c in enumerate(cols)}
    r["pseudoR2_within"] = 1 - r["sigma2"] / ri["null"]["sigma2"]
    r["pseudoR2_between"] = 1 - r["tau2"] / ri["null"]["tau2"]
    del r["b"], r["se"]
    ri[k] = r
ml["random_intercept_ML"] = ri
out["mainlife"] = ml

# ---------------- Gainey ----------------
g = pd.read_csv(A + "/0058_gainey_ts.tsv", sep="\t").sort_values(["id", "day", "beep"]).copy()
pos = ["active", "interested", "excited", "strong"]; neg = ["irritable", "upset", "afraid_anxious", "sad"]
g["pa"] = g[pos].mean(axis=1).where(g[pos].notna().all(axis=1))
g["na"] = g[neg].mean(axis=1).where(g[neg].notna().all(axis=1))
for c in ["meaning", "pa", "na", "getting_done", "day", "beep", "time_since_start"]:
    g["l_" + c] = g.groupby("id")[c].shift(1)
g["gap"] = g.time_since_start - g.l_time_since_start
pairs = g[(g.day == g.l_day) & (g.beep == g.l_beep + 1) & (g.gap > 0) & (g.gap < 12)]
b3 = pairs.dropna(subset=["meaning", "pa", "na", "l_meaning", "l_pa", "l_na"]).copy()
cols = ["l_meaning", "l_pa", "l_na", "day", "beep", "gap"]

def fe_fit(df, ycol, cols):
    Xw = within(df, cols, "id"); yw = df[ycol] - df.groupby("id")[ycol].transform("mean")
    return cluster_ols(yw, Xw, df.id, cols)

gy = {"pairs": len(b3), "people": int(b3.id.nunique()), "fe_model": fe_fit(b3, "meaning", cols)}
# Nickell-bias check: half-panel jackknife (Dhaene & Jochmans 2015), split each person's series by day 1-3 vs 4-7
b3["half"] = np.where(b3.day <= 3, 0, 1)
full = gy["fe_model"]
h0 = fe_fit(b3[b3.half == 0], "meaning", cols); h1 = fe_fit(b3[b3.half == 1], "meaning", cols)
gy["half_panel_jackknife"] = {c: 2 * full[c]["b"] - 0.5 * (h0[c]["b"] + h1[c]["b"]) for c in ["l_meaning", "l_pa", "l_na"]}
# Reverse direction: does earlier meaning predict later PA?
gy["reverse_meaning_to_pa"] = fe_fit(b3, "pa", ["l_pa", "l_meaning", "l_na", "day", "beep", "gap"])["l_meaning"]
gy["avg_obs_per_person"] = len(b3) / b3.id.nunique()
out["gainey_phase3"] = gy

# ---------------- Phase 5 ----------------
b5 = pairs.dropna(subset=["meaning", "getting_done", "pa", "na", "l_meaning", "l_getting_done", "l_pa", "l_na"]).copy()
b5["daybeep"] = b5.day.astype(int).astype(str) + "_" + b5.beep.astype(int).astype(str)
D = pd.get_dummies(b5.daybeep, dtype=float, drop_first=True)
def fe_dummies(ycol, cols):
    X = pd.concat([b5[cols], D], axis=1)
    Xw = X - X.groupby(b5.id).transform("mean"); yw = b5[ycol] - b5.groupby("id")[ycol].transform("mean")
    allnames = list(X.columns)
    r = cluster_ols(yw, Xw, b5.id, allnames)
    return {c: r[c] for c in cols}
out["phase5"] = {"pairs": len(b5), "people": int(b5.id.nunique()),
                 "meaning_to_getting_done": fe_dummies("getting_done", ["l_getting_done", "l_meaning", "l_pa", "l_na", "gap"])["l_meaning"],
                 "pa_to_getting_done": fe_dummies("getting_done", ["l_getting_done", "l_meaning", "l_pa", "l_na", "gap"])["l_pa"],
                 "getting_done_to_meaning": fe_dummies("meaning", ["l_meaning", "l_getting_done", "l_pa", "l_na", "gap"])["l_getting_done"]}

print(json.dumps(out, indent=1, default=float))
