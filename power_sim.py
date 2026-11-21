"""Power simulation for the refined claim (allocation link). Written by Claude.
Design: N people x 4 pursuits x W weeks. Each week, people re-rate how significant earlier experiences
tied to each pursuit now feel; the change (dC) is measured with reliability 0.7. Next week's hours on that
pursuit depend on last week's hours (habit), weekly mood, goal importance, and beta * true dC.
beta is in SD units of the usual week-to-week fluctuation in hours. 20% of person-weeks missing at random.
Estimator: OLS of hours(w+1) on hours(w), observed dC, mood, goal, person-clustered SEs. Power = share of
replications whose 95% CI excludes 0 in the predicted direction.
Usage: python3 -I power_sim.py <out.json>
"""
import sys, json
import numpy as np

rng = np.random.default_rng(20261121)
K, REL, PHI, MISS = 4, 0.7, 0.5, 0.2
Z = 1.959963984540054

def one(N, W, beta):
    P = N * K
    person = np.repeat(np.arange(N), K)
    mu = rng.normal(0, 1, P)                       # stable pursuit level (person x pursuit)
    goal = mu * 0.6 + rng.normal(0, 0.8, P)        # goal importance tracks stable level
    slope = beta + rng.normal(0, abs(beta) * 0.5, N)[person]  # people differ in sensitivity
    h = mu + rng.normal(0, 1, P)
    rows = []
    for w in range(W - 1):
        dC = rng.normal(0, 1, P)
        dC_obs = np.sqrt(REL) * dC + np.sqrt(1 - REL) * rng.normal(0, 1, P)
        mood = np.repeat(rng.normal(0, 1, N), K)
        h_next = mu + PHI * (h - mu) + slope * dC + 0.1 * mood + rng.normal(0, 1, P)
        keep = rng.random(P) > MISS
        rows.append(np.column_stack([h_next, h, dC_obs, mood, goal, person])[keep])
        h = h_next
    D = np.vstack(rows)
    y = D[:, 0]; X = np.column_stack([np.ones(len(D)), D[:, 1:5]]); g = D[:, 5].astype(int)
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    S = np.zeros((N, X.shape[1])); np.add.at(S, g, X * e[:, None])
    n, k = X.shape
    V = (N / (N - 1)) * ((n - 1) / (n - k)) * XtX @ (S.T @ S) @ XtX
    se = np.sqrt(V[2, 2])
    return b[2], se

out = {"assumptions": {"pursuits": K, "reliability_of_significance_change": REL, "habit_phi": PHI,
                       "missing_share": MISS, "slope_heterogeneity_sd": "0.5 x beta", "reps_per_cell": 300}, "cells": []}
for beta in (0.05, 0.10, 0.15, 0.20):
    for W in (6, 8, 12):
        for N in (40, 60, 80, 100, 150, 200, 300, 400):
            hits, ests = 0, []
            for r in range(300):
                b, se = one(N, W, beta)
                hits += (b - Z * se) > 0
                ests.append(b)
            out["cells"].append({"beta": beta, "weeks": W, "people": N, "power": hits / 300,
                                 "mean_estimate": float(np.mean(ests))})
        row = [c for c in out["cells"] if c["beta"] == beta and c["weeks"] == W]
        print(beta, W, [(c["people"], round(c["power"], 2)) for c in row], "est~", round(row[-1]["mean_estimate"], 3))
json.dump(out, open(sys.argv[1], "w"), indent=1)
