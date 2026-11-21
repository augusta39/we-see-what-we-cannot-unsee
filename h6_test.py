"""H6 test in Hippocorpus, implementing protocol_H6_hippocorpus_v1.md (sha256 03b69a52...).
Written by Claude. Usage: python3 -I h6_test.py <path to hcV3-stories.csv> <output json>
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

CSV, OUT = sys.argv[1], sys.argv[2]
SEED = 20261121
log = []

def cluster_ols(y, X, g, names):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    idx = pd.factorize(np.asarray(g))[0]; G = idx.max() + 1
    S = np.zeros((G, k)); np.add.at(S, idx, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None)); z = 1.959963984540054
    return {nm: {"b": float(b[i]), "se": float(se[i]), "lo": float(b[i] - z * se[i]), "hi": float(b[i] + z * se[i]),
                 "p": float(2 * stats.norm.sf(abs(b[i] / se[i]))) if se[i] > 0 else None} for i, nm in enumerate(names)}

def zs(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std(ddof=0)

def ols_coef(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]

# ---------------- data ----------------
d = pd.read_csv(CSV)
rec = d[d.memType == "recalled"].reset_index(drop=True)
im = d[d.memType == "imagined"].reset_index(drop=True)
pos = {a: i for i, a in enumerate(rec.AssignmentId)}
rec_imp = pd.to_numeric(rec.importance, errors="coerce").to_numpy()
rec_worker = rec.WorkerId.astype(str).to_numpy()

def vectors(kind):
    texts = rec.summary.fillna("").astype(str)
    tf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=2)
    X = tf.fit_transform(texts)
    if kind == "svd":
        Z = TruncatedSVD(n_components=100, random_state=0).fit_transform(X)
        return normalize(Z)
    return normalize(X).toarray()

# own recalled stories per worker (with non-missing importance)
own_by_worker = {}
for i, (w, c) in enumerate(zip(rec_worker, rec_imp)):
    if not np.isnan(c):
        own_by_worker.setdefault(w, []).append(i)

def build(V):
    colsum = V.sum(axis=0); nrec = V.shape[0]
    rows = []
    for _, r in im.iterrows():
        w = str(r.WorkerId); src = pos.get(r.recImgPairId)
        if src is None or w not in own_by_worker:
            continue
        if rec_worker[src] == w:
            continue  # scenario is the worker's own story
        own = [j for j in own_by_worker[w] if j != src]
        if not own:
            continue
        sims = V[own] @ V[src]
        k = int(np.argmax(sims))
        allsum = colsum @ V[src] - V[src] @ V[src] - sims.sum()
        typ = allsum / (nrec - 1 - len(own))
        rows.append({"worker": w, "aid": r.AssignmentId, "src": src, "y": pd.to_numeric(r.importance, errors="coerce"),
                     "simrate": pd.to_numeric(r.similarity, errors="coerce"), "s": float(sims[k]),
                     "C_own": float(rec_imp[own[k]]), "src_imp": float(rec_imp[src]), "typ": float(typ),
                     "n_own": len(own)})
    return pd.DataFrame(rows)

V = vectors("svd")
df_all = build(V)
n_before = len(df_all)
df = df_all.dropna(subset=["y", "s", "C_own", "src_imp", "typ"]).reset_index(drop=True)
log.append(f"eligible imagined stories {n_before}; complete cases {len(df)} (dropped {n_before-len(df)} with a missing model variable)")

def design(frame, with_h6=True):
    Z = {k: zs(frame[k]) for k in ["s", "C_own", "src_imp", "typ"]}
    cols = [np.ones(len(frame)), Z["src_imp"], Z["typ"], Z["C_own"]]; names = ["const", "src_imp", "typ", "C_own"]
    if with_h6:
        cols += [Z["s"], Z["s"] * Z["C_own"]]; names += ["s", "s_x_C_own"]
    return np.column_stack(cols), names

res = {"protocol_sha256": "03b69a52395bd3451dbc428a65443c14367b8ff8c6963fbbfeda37846b2c9391", "log": log}
res["sample"] = {"stories": len(df), "workers": int(df.worker.nunique()),
                 "stories_per_worker_median": float(df.groupby("worker").size().median()),
                 "stories_per_worker_max": int(df.groupby("worker").size().max()),
                 "workers_with_2plus": int((df.groupby("worker").size() >= 2).sum()),
                 "y_mean": float(df.y.mean()), "y_sd": float(df.y.std()),
                 "s_mean": float(df.s.mean()), "s_sd": float(df.s.std())}

# ---------------- primary ----------------
X1, n1 = design(df)
yz = zs(df.y)
res["M1_standardized"] = cluster_ols(yz, X1, df.worker, n1)
res["M1_raw_y"] = cluster_ols(df.y, X1, df.worker, n1)
X0, n0 = design(df, with_h6=False)
res["M0_standardized"] = cluster_ols(yz, X0, df.worker, n0)
gamma_obs = res["M1_standardized"]["s_x_C_own"]["b"]; beta_obs = res["M1_standardized"]["s"]["b"]

# ---------------- manipulation check ----------------
mc = df.dropna(subset=["simrate"]).reset_index(drop=True)
Xm = np.column_stack([np.ones(len(mc)), zs(mc.s), zs(mc.src_imp), zs(mc.typ)])
res["manipulation_check"] = cluster_ols(zs(mc.simrate), Xm, mc.worker, ["const", "s", "src_imp", "typ"])
res["manipulation_check"]["n"] = len(mc)

# ---------------- within-worker ----------------
cnt = df.groupby("worker").worker.transform("size")
ww = df[cnt >= 2].reset_index(drop=True)
Xw, nw = design(ww)
Xw = pd.DataFrame(Xw[:, 1:], columns=nw[1:])
Xw = Xw - Xw.groupby(ww.worker).transform("mean")
keep = [c for c in Xw.columns if Xw[c].std() > 1e-9]
yw = zs(ww.y); yw = yw - pd.Series(yw).groupby(ww.worker).transform("mean").to_numpy()
res["within_worker"] = cluster_ols(yw, Xw[keep].to_numpy(), ww.worker, keep)
res["within_worker"]["n_stories"] = len(ww); res["within_worker"]["n_workers"] = int(ww.worker.nunique())
res["within_worker"]["dropped_constant_terms"] = [c for c in Xw.columns if c not in keep]

# ---------------- placebo ----------------
rng = np.random.default_rng(SEED)
workers = df.worker.unique()
all_rec = np.arange(len(rec))
valid_rec = all_rec[~np.isnan(rec_imp)]
n_own_w = df.groupby("worker").n_own.first().to_dict()
src_arr = df.src.to_numpy(); w_arr = df.worker.to_numpy()
Vsrc = V[src_arr]
yz_arr = yz
pg, pb = [], []
for it in range(1000):
    draw = {}
    for w in workers:
        pool_mask = rec_worker[valid_rec] != w
        draw[w] = rng.choice(valid_rec[pool_mask], size=n_own_w[w], replace=False)
    s_p = np.empty(len(df)); c_p = np.empty(len(df))
    for i in range(len(df)):
        cand = draw[w_arr[i]]
        cand = cand[cand != src_arr[i]]
        if len(cand) == 0:
            cand = np.array([rng.choice(valid_rec[(rec_worker[valid_rec] != w_arr[i]) & (valid_rec != src_arr[i])])])
        sims = V[cand] @ Vsrc[i]; k = int(np.argmax(sims))
        s_p[i] = sims[k]; c_p[i] = rec_imp[cand[k]]
    Zs, Zc = zs(s_p), zs(c_p)
    Xp = np.column_stack([np.ones(len(df)), zs(df.src_imp), zs(df.typ), Zc, Zs, Zs * Zc])
    b = ols_coef(yz_arr, Xp)
    pb.append(b[4]); pg.append(b[5])
pg, pb = np.array(pg), np.array(pb)
res["placebo"] = {"n_perm": 1000, "gamma_obs": gamma_obs, "gamma_placebo_mean": float(pg.mean()),
                  "gamma_placebo_95pct": float(np.quantile(pg, .95)), "p_gamma": float((pg >= gamma_obs).mean()),
                  "beta_s_obs": beta_obs, "beta_s_placebo_mean": float(pb.mean()),
                  "beta_s_placebo_95pct": float(np.quantile(pb, .95)), "p_beta_s": float((pb >= beta_obs).mean())}

# ---------------- held-out ----------------
rng2 = np.random.default_rng(SEED)
uw = df.worker.unique(); rm0, rm1 = [], []
X0r, _ = design(df, with_h6=False); X1r, _ = design(df)
yraw = df.y.to_numpy()
for rep in range(10):
    perm = rng2.permutation(uw); folds = np.array_split(perm, 5)
    for f in folds:
        te = df.worker.isin(f).to_numpy(); tr = ~te
        p0 = X0r[te] @ ols_coef(yraw[tr], X0r[tr]); p1 = X1r[te] @ ols_coef(yraw[tr], X1r[tr])
        rm0.append(np.sqrt(np.mean((yraw[te] - p0) ** 2))); rm1.append(np.sqrt(np.mean((yraw[te] - p1) ** 2)))
rm0, rm1 = np.array(rm0), np.array(rm1)
res["heldout"] = {"splits": len(rm0), "rmse_M0": float(rm0.mean()), "rmse_M1": float(rm1.mean()),
                  "mean_pct_improvement": float(np.mean((rm0 - rm1) / rm0) * 100),
                  "share_M1_better": float(np.mean(rm1 < rm0))}

# ---------------- robustness ----------------
rob = {}
rng3 = np.random.default_rng(SEED)
keep_idx = []
for w, idxs in df.groupby("worker").groups.items():
    idxs = np.array(sorted(idxs))
    keep_idx.extend(idxs if len(idxs) <= 10 else rng3.choice(idxs, 10, replace=False))
capped = df.loc[sorted(keep_idx)].reset_index(drop=True)
log.append("2026-11-21: bug fix, cap-10 subsetting rewritten without groupby.apply (pandas dropped the grouping column); no change to the specified procedure")
Xc, nc = design(capped)
rob["cap10"] = {"n": len(capped), **{k: v for k, v in cluster_ols(zs(capped.y), Xc, capped.worker, nc).items() if k in ("s", "s_x_C_own")}}
Vr = vectors("raw")
dr = build(Vr).dropna(subset=["y", "s", "C_own", "src_imp", "typ"]).reset_index(drop=True)
Xr, nr = design(dr)
rob["raw_tfidf"] = {"n": len(dr), **{k: v for k, v in cluster_ols(zs(dr.y), Xr, dr.worker, nr).items() if k in ("s", "s_x_C_own")}}
yb = (df.y >= 4).astype(float).to_numpy()
rob["binary_y_ge4_LPM"] = {k: v for k, v in cluster_ols(yb, X1, df.worker, n1).items() if k in ("s", "s_x_C_own")}
res["robustness"] = rob

# ---------------- decision ----------------
g = res["M1_standardized"]["s_x_C_own"]; h = res["heldout"]
if g["b"] > 0 and g["lo"] > 0 and res["placebo"]["p_gamma"] < .05:
    dec = "supported" if (h["mean_pct_improvement"] > 0 and h["share_M1_better"] >= .8) else "coefficient supported, held-out criterion not met"
elif g["hi"] < 0.05:
    dec = "refuted"
else:
    dec = "inconclusive"
res["decision"] = dec
mcs = res["manipulation_check"]["s"]
res["manipulation_check_passed"] = bool(mcs["lo"] > 0 or mcs["hi"] < 0)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: res[k] for k in ["sample", "decision", "manipulation_check_passed", "placebo", "heldout"]}, indent=1))
print("M1", {k: (round(v["b"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["M1_standardized"].items()})
print("WW", {k: (round(v["b"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["within_worker"].items() if isinstance(v, dict)})
print("MC", {k: (round(v["b"], 4), round(v["lo"], 4), round(v["hi"], 4)) for k, v in res["manipulation_check"].items() if isinstance(v, dict)})
print("ROB", json.dumps(rob, indent=0)[:1500])
print(log)
