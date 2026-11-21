# Protocol: H6 test in Hippocorpus (v2, cross-fitted measure selection)

Frozen: 2026-11-21, after v1 (sha256 03b69a52...) returned "inconclusive" with a failed manipulation check: the v1 text-similarity measure did not predict workers' own similarity ratings. v2 changes only how similarity is measured. Everything else (sample, outcome, covariates, M0/M1, decision rule, smallest effect of interest) is v1's. Author: Claude.

## Why a v2 is legitimate
v1's estimand is uninterpretable without a valid similarity measure. Choosing a measure by its validity against the self-rated similarity item could leak outcome information, because the same person gives both ratings at the same time. v2 prevents this with cross-fitting: a measure is chosen in one half of workers and the H6 test is run only in the other half.

## Folds
Workers split 50/50 at random (seed 20261121) into folds A and B.

## Candidate measures (fixed list of 8)
Text pairs (own side, scenario side):
1. own summary, scenario summary
2. own story, scenario summary
3. own story, source story
4. own mainEvent, source mainEvent
Representations: (a) TF-IDF as in v1 + truncated SVD to 100 dims; (b) raw TF-IDF. The vectorizer is fitted on all recalled texts of the fields involved. s = max cosine over the worker's own recalled stories; C_own = importance of the most similar one; typ = mean cosine of the scenario-side text to all recalled own-side texts except the source and the worker's own.

## Selection (uses only the manipulation check, never importance)
In the selection fold, for each candidate, fit simrate ~ s + src_imp + typ (z-scored, worker-clustered). Choose the candidate with the largest t for s, provided its 95% CI excludes 0. If none qualifies, that direction of the cross-fit stops: "no valid measure".

## Test (in the other fold only)
1. Validity must replicate: the chosen measure's manipulation-check CI excludes 0 (positive). Else that fold's test is uninterpretable.
2. Fit v1's M1; record gamma (s x C_own, standardized) with clustered SE.
3. Placebo: 500 permutations of own stories as in v1; record placebo gammas.
4. Held-out: v1's procedure within the fold.

## Pooling and decision
Pool fold gammas by inverse-variance weights (fixed effect); pooled placebo gammas use the same weights per permutation index. Apply v1's decision rule to the pooled gamma: supported (CI > 0, placebo p < .05, and held-out criterion: mean RMSE improvement > 0 and M1 better in >= 80% of all splits across folds); refuted (pooled CI upper < 0.05); else inconclusive. If neither fold yields an interpretable test, the result is: "Hippocorpus cannot test H6 with available measures."

## Allowed changes
Bug fixes only, logged with timestamps.
