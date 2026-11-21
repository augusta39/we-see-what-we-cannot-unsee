"""Independent checks of the H6 v1/v2 results (Claude). Usage: python3 -I h6_checks.py <hcV3-stories.csv>"""
import sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import LinearRegression

d = pd.read_csv(sys.argv[1])
rec = d[d.memType == "recalled"].reset_index(drop=True)
ret = d[d.memType == "retold"].reset_index(drop=True)
im = d[d.memType == "imagined"].reset_index(drop=True)
tf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True, min_df=2).fit(rec.summary.fillna(""))
svd = TruncatedSVD(100, random_state=0).fit(tf.transform(rec.summary.fillna("")))
vec = lambda t: normalize(svd.transform(tf.transform(pd.Series(t).fillna("").astype(str))))
Vr = vec(rec.summary); pos = {a: i for i, a in enumerate(rec.AssignmentId)}
rng = np.random.default_rng(1)

# Positive control 1: same event, same person, retold months later
m = ret[ret.recAgnPairId.isin(pos)]
Vt = vec(m.summary); idx = m.recAgnPairId.map(pos).to_numpy()
same = (Vt * Vr[idx]).sum(1); rand = (Vt * Vr[rng.integers(0, len(rec), len(idx))]).sum(1)
print(f"retold vs own recalled (same event): mean cos {same.mean():.3f}; vs random recalled: {rand.mean():.3f}; AUC {np.mean(same[:,None] > rand[None,:]):.3f}")

# Positive control 2: imagined story vs the summary it was written from
Vi = vec(im.story); sidx = im.recImgPairId.map(pos).to_numpy()
same2 = (Vi * Vr[sidx]).sum(1); rand2 = (Vi * Vr[rng.integers(0, len(rec), len(sidx))]).sum(1)
print(f"imagined story vs its prompt: mean cos {same2.mean():.3f}; vs random prompt: {rand2.mean():.3f}; AUC {np.mean(same2[:,None] > rand2[None,:]):.3f}")

# Distribution of s in the H6 sample and the raw validity correlation
own = {}
for i, (w, c) in enumerate(zip(rec.WorkerId.astype(str), pd.to_numeric(rec.importance, errors="coerce"))):
    if not np.isnan(c): own.setdefault(w, []).append(i)
rows = []
for _, r in im.iterrows():
    w = str(r.WorkerId); s = pos.get(r.recImgPairId)
    if s is None or w not in own or rec.WorkerId.astype(str)[s] == w: continue
    o = [j for j in own[w] if j != s]
    if not o: continue
    sims = Vr[o] @ Vr[s]; k = int(np.argmax(sims))
    rows.append((w, sims[k], float(rec.importance[o[k]]), float(rec.importance[s]), r.importance, r.similarity))
h = pd.DataFrame(rows, columns=["w", "s", "C", "src", "y", "simrate"]).dropna(subset=["y", "src"])
print(f"n={len(h)}; s quantiles 10/50/90%: {np.quantile(h.s,[.1,.5,.9]).round(3)}; share s<0.02: {(h.s<0.02).mean():.2f}")
print(f"raw corr(s, self-rated similarity) = {h.s.corr(h.simrate):.3f}; corr(self-rated similarity, imagined importance) = {h.simrate.corr(h.y):.3f}")

# Point-estimate cross-check of v1 gamma with sklearn (typ omitted here, so expect close but not identical)
Z = StandardScaler().fit_transform(h[["src", "C", "s"]])
X = np.column_stack([Z, Z[:, 2] * Z[:, 1]]); yz = (h.y - h.y.mean()) / h.y.std(ddof=0)
print("sklearn gamma (no typ control):", round(LinearRegression().fit(X, yz).coef_[3], 4))
