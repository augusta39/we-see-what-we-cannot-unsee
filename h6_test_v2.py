"""H6 test in Hippocorpus, protocol v2 (cross-fitted measure selection), sha256 9d97ebee...
Written by Claude. Usage: python3 -I h6_test_v2.py <hcV3-stories.csv> <output json>
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import vstack

CSV, OUT = sys.argv[1], sys.argv[2]
SEED = 20261121
Z95 = 1.959963984540054
log = []

def cluster_ols(y, X, g, names):
    y = np.asarray(y, float); X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; e = y - X @ b
    idx = pd.factorize(np.asarray(g))[0]; G = idx.max() + 1
    S = np.zeros((G, k)); np.add.at(S, idx, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ (S.T @ S) @ XtX_inv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    return {nm: {"b": float(b[i]), "se": float(se[i]), "lo": float(b[i] - Z95 * se[i]), "hi": float(b[i] + Z95 * se[i]),
                 "t": float(b[i] / se[i]) if se[i] > 0 else 0.0} for i, nm in enumerate(names)}

def zs(x):
    x = np.asarray(x, float); return (x - x.mean()) / x.std(ddof=0)

def lstsq(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]

d = pd.read_csv(CSV)
rec = d[d.memType == "recalled"].reset_index(drop=True)
im = d[d.memType == "imagined"].reset_index(drop=True)
pos = {a: i for i, a in enumerate(rec.AssignmentId)}
rec_imp = pd.to_numeric(rec.importance, errors="coerce").to_numpy()
rec_worker = rec.WorkerId.astype(str).to_numpy()
own_by_worker = {}
for i, (w, c) in enumerate(zip(rec_worker, rec_imp)):
    if not np.isnan(c):
        own_by_worker.setdefault(w, []).append(i)
valid_rec = np.where(~np.isnan(rec_imp))[0]

CANDS = [("summary", "summary"), ("story", "summary"), ("story", "story"), ("mainEvent", "mainEvent")]

def make_vectors(own_f, scen_f, rep):
    t_own = rec[own_f].fillna("").astype(str); t_scen = rec[scen_f].fillna("").astype(str)
    fit_texts = pd.concat([t_own, t_scen]) if own_f != scen_f else t_own
    tf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=2).fit(fit_texts)
    Xo, Xs = tf.transform(t_own), tf.transform(t_scen)
    if rep == "svd":
        svd = TruncatedSVD(n_components=100, random_state=0).fit(vstack([Xo, Xs]) if own_f != scen_f else Xo)
        return normalize(svd.transform(Xo)), normalize(svd.transform(Xs))
    return normalize(Xo).toarray(), normalize(Xs).toarray()

# v1 sample definition (complete cases), measure-independent parts
base = []
for _, r in im.iterrows():
    w = str(r.WorkerId); src = pos.get(r.recImgPairId)
    if src is None or w not in own_by_worker or rec_worker[src] == w:
        continue
    own = [j for j in own_by_worker[w] if j != src]
    if not own:
        continue
    base.append({"worker": w, "src": src, "own": own, "y": pd.to_numeric(r.importance, errors="coerce"),
                 "simrate": pd.to_numeric(r.similarity, errors="coerce"), "src_imp": rec_imp[src]})
base = pd.DataFrame(base).dropna(subset=["y", "src_imp"]).reset_index(drop=True)
log.append(f"sample {len(base)} imagined stories, {base.worker.nunique()} workers (v1 definition)")

def measure(Vo, Vs):
    colsum = Vo.sum(axis=0); n = Vo.shape[0]
    s, c, typ = [], [], []
    for _, r in base.iterrows():
        sims = Vo[r.own] @ Vs[r.src]; k = int(np.argmax(sims))
        s.append(sims[k]); c.append(rec_imp[r.own[k]])
        typ.append((colsum @ Vs[r.src] - Vo[r.src] @ Vs[r.src] - sims.sum()) / (n - 1 - len(r.own)))
    return np.array(s), np.array(c), np.array(typ)

M = {}
for own_f, scen_f in CANDS:
    for rep in ("svd", "raw"):
        Vo, Vs = make_vectors(own_f, scen_f, rep)
        s, c, t = measure(Vo, Vs)
        M[f"{own_f}|{scen_f}|{rep}"] = {"Vo": Vo, "Vs": Vs, "s": s, "C": c, "typ": t}

rng = np.random.default_rng(SEED)
uw = base.worker.unique(); perm = rng.permutation(uw)
foldA = set(perm[: len(perm) // 2])
base["fold"] = np.where(base.worker.isin(foldA), "A", "B")

def mc_fit(mask, key):
    m = mask & base.simrate.notna().to_numpy()
    sub = base[m]; mm = M[key]
    X = np.column_stack([np.ones(m.sum()), zs(mm["s"][m]), zs(sub.src_imp), zs(mm["typ"][m])])
    return cluster_ols(zs(sub.simrate), X, sub.worker, ["const", "s", "src_imp", "typ"])["s"]

def m1_design(sub_idx, s, C, typ):
    sub = base.iloc[sub_idx]
    Zs_, Zc = zs(s[sub_idx]), zs(C[sub_idx])
    X1 = np.column_stack([np.ones(len(sub)), zs(sub.src_imp), zs(typ[sub_idx]), Zc, Zs_, Zs_ * Zc])
    X0 = X1[:, :4]
    return sub, X0, X1

res = {"protocol_sha256": "9d97ebee0166951dcc6703ebc6378a462c7d49ddce4b484224bc638f0a00ff5d", "log": log, "folds": {}}
fold_out = {}
for sel, tst in (("A", "B"), ("B", "A")):
    smask = (base.fold == sel).to_numpy(); tmask = (base.fold == tst).to_numpy()
    sel_table = {k: mc_fit(smask, k) for k in M}
    ok = {k: v for k, v in sel_table.items() if v["lo"] > 0}
    entry = {"selection_fold": sel, "test_fold": tst,
             "selection_mc": {k: {"b": round(v["b"], 4), "lo": round(v["lo"], 4), "hi": round(v["hi"], 4), "t": round(v["t"], 2)} for k, v in sel_table.items()}}
    if not ok:
        entry["status"] = "no valid measure in selection fold"; res["folds"][tst] = entry; continue
    key = max(ok, key=lambda k: ok[k]["t"]); entry["chosen"] = key
    val = mc_fit(tmask, key); entry["test_fold_validity"] = val
    if not val["lo"] > 0:
        entry["status"] = "chosen measure failed validity in test fold"; res["folds"][tst] = entry; continue
    mm = M[key]; tidx = np.where(tmask)[0]
    sub, X0, X1 = m1_design(tidx, mm["s"], mm["C"], mm["typ"])
    fit = cluster_ols(zs(sub.y), X1, sub.worker, ["const", "src_imp", "typ", "C_own", "s", "s_x_C_own"])
    entry["M1"] = fit
    # placebo
    prng = np.random.default_rng(SEED + (1 if tst == "B" else 2))
    Vo, Vs = mm["Vo"], mm["Vs"]; pg = []
    yz = zs(sub.y)
    for it in range(500):
        draw = {}
        for w in sub.worker.unique():
            n_own = len(sub[sub.worker == w].iloc[0].own)
            draw[w] = prng.choice(valid_rec[rec_worker[valid_rec] != w], size=n_own, replace=False)
        sp, cp = np.empty(len(sub)), np.empty(len(sub))
        for i, (_, r) in enumerate(sub.iterrows()):
            cand = draw[r.worker]; cand = cand[cand != r.src]
            if len(cand) == 0:
                cand = np.array([prng.choice(valid_rec[(rec_worker[valid_rec] != r.worker) & (valid_rec != r.src)])])
            sims = Vo[cand] @ Vs[r.src]; k = int(np.argmax(sims)); sp[i] = sims[k]; cp[i] = rec_imp[cand[k]]
        Zs_, Zc = zs(sp), zs(cp)
        Xp = np.column_stack([np.ones(len(sub)), zs(sub.src_imp), zs(mm["typ"][tidx]), Zc, Zs_, Zs_ * Zc])
        pg.append(lstsq(yz, Xp)[5])
    pg = np.array(pg)
    entry["placebo"] = {"gamma_placebo_mean": float(pg.mean()), "gamma_placebo_95pct": float(np.quantile(pg, .95)),
                        "p_gamma": float((pg >= fit["s_x_C_own"]["b"]).mean())}
    # held-out within fold
    hrng = np.random.default_rng(SEED); yraw = sub.y.to_numpy(); r0, r1 = [], []
    for rep in range(10):
        ws = hrng.permutation(sub.worker.unique())
        for f in np.array_split(ws, 5):
            te = sub.worker.isin(f).to_numpy(); tr = ~te
            p0 = X0[te] @ lstsq(yraw[tr], X0[tr]); p1 = X1[te] @ lstsq(yraw[tr], X1[tr])
            r0.append(np.sqrt(np.mean((yraw[te] - p0) ** 2))); r1.append(np.sqrt(np.mean((yraw[te] - p1) ** 2)))
    r0, r1 = np.array(r0), np.array(r1)
    entry["heldout"] = {"rmse_M0": float(r0.mean()), "rmse_M1": float(r1.mean()),
                        "mean_pct_improvement": float(np.mean((r0 - r1) / r0) * 100), "share_M1_better": float(np.mean(r1 < r0))}
    entry["status"] = "tested"; entry["n_stories"] = len(sub); entry["n_workers"] = int(sub.worker.nunique())
    res["folds"][tst] = entry
    fold_out[tst] = (fit["s_x_C_own"], pg, r0, r1)

if fold_out:
    gs = [v[0] for v in fold_out.values()]
    w = np.array([1 / g["se"] ** 2 for g in gs]); b = np.array([g["b"] for g in gs])
    gp = float((w * b).sum() / w.sum()); sep = float(np.sqrt(1 / w.sum()))
    pls = np.vstack([v[1] for v in fold_out.values()]); pooled_pl = (w[:, None] * pls).sum(0) / w.sum()
    r0 = np.concatenate([v[2] for v in fold_out.values()]); r1 = np.concatenate([v[3] for v in fold_out.values()])
    pooled = {"gamma": gp, "se": sep, "lo": gp - Z95 * sep, "hi": gp + Z95 * sep, "folds_tested": list(fold_out),
              "placebo_p": float((pooled_pl >= gp).mean()),
              "heldout_mean_pct_improvement": float(np.mean((r0 - r1) / r0) * 100), "heldout_share_M1_better": float(np.mean(r1 < r0))}
    if pooled["lo"] > 0 and pooled["placebo_p"] < .05:
        dec = "supported" if pooled["heldout_mean_pct_improvement"] > 0 and pooled["heldout_share_M1_better"] >= .8 else "coefficient supported, held-out criterion not met"
    elif pooled["hi"] < 0.05:
        dec = "refuted"
    else:
        dec = "inconclusive"
    res["pooled"] = pooled
else:
    dec = "Hippocorpus cannot test H6 with available measures"
res["decision"] = dec
json.dump(res, open(OUT, "w"), indent=1, default=float)
for f, e in res["folds"].items():
    print("TEST FOLD", f, e["status"], "| chosen:", e.get("chosen"))
    print("  selection MC:", e["selection_mc"])
    if "test_fold_validity" in e: print("  test validity:", {k: round(v, 4) for k, v in e["test_fold_validity"].items()})
    if "M1" in e: print("  gamma:", {k: round(v, 4) for k, v in e["M1"]["s_x_C_own"].items()}, "beta_s:", {k: round(v, 4) for k, v in e["M1"]["s"].items()})
    if "placebo" in e: print("  placebo:", e["placebo"], "heldout:", e["heldout"])
print("POOLED", res.get("pooled")); print("DECISION", dec); print(log)
