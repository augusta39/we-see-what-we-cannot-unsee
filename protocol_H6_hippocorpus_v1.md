# Protocol: H6 test in Hippocorpus (v1)

Frozen: 2026-11-21T16:41Z, before any analysis relating similarity, own-memory importance and imagined-event importance. Prior contact with these data: descriptive audits of recalled vs retold importance (archive), and marginal summary statistics of imagined `importance` and `similarity`. No relationship between the variables below has been computed. Author: Claude, for Roddsi's "Pursuits and Perception" project.

## Claim under test (H6, formal)
The weight a person gives a new experience k depends on their own earlier experiences j, in proportion to how similar j is to k and how significant j was to them:
  alpha_k = alpha_0 + gamma * sim(j, k) * C_j
gamma = 0 is the null. gamma > 0 means past significance carries over to new, similar experiences ("we see what we cannot unsee").

## Design
Hippocorpus (hcV3-stories.csv). Workers wrote a recalled story about their own recent experience and rated its importance (C_j). Some of the same workers were later assigned another person's event summary, imagined it happening to themselves, wrote it, and rated its importance (outcome) and its similarity to their own life. The assigned scenario is someone else's event, so its similarity to the worker's own past varies independently of the worker.

## Sample
- Imagined stories whose writer also has at least one recalled story.
- Exclude imagined stories whose scenario source is the worker's own recalled story.
- Exclude rows missing imagined importance or own-story importance.
- Retold stories are not used.

## Variables
- y: importance rating of the imagined story (1-5).
- Scenario text: the summary shown to the imaginer (identical to the source recalled story's summary).
- Own text: the summary of each of the worker's own recalled stories.
- Text vectors: TF-IDF (unigrams and bigrams, English stop words, sublinear tf, min_df = 2) fitted on all recalled summaries, reduced to 100 dimensions with truncated SVD (random_state = 0), L2-normalized. Similarity = cosine.
- s: maximum cosine similarity between the scenario and any of the worker's own recalled summaries.
- C_own: importance rating of that most-similar own story.
- src_imp: importance rating that the source author gave the scenario event (its inherent importance to the person who lived it).
- typ: scenario typicality = mean cosine similarity of the scenario to all recalled summaries except its source and the worker's own (how generally relatable the scenario is).
- All predictors z-scored in the analysis sample. y analyzed in raw units and z-scored for standardized coefficients.

## Models
- M0: y ~ src_imp + typ + C_own
- M1 (primary): y ~ src_imp + typ + C_own + s + s x C_own
- OLS, worker-clustered standard errors (CR1).
- Within-worker version (secondary): workers with >= 2 imagined stories, worker fixed effects, terms s and s x C_own (plus src_imp, typ).

## Primary estimand and decision rule
gamma = standardized coefficient of s x C_own in M1 (y z-scored).
- Supported: gamma > 0, 95% CI excludes 0, AND placebo p < .05.
- Refuted: 95% CI upper bound < 0.05 (smallest effect of interest = 0.05 SD of y per SD x SD).
- Inconclusive: anything else.
Secondary estimand: beta_s, the standardized coefficient of s (similarity effect at average own importance).

## Placebo (specificity test)
1,000 permutations (seed 20261121): for each worker, replace each own recalled story with a recalled story drawn at random from other workers (never the scenario's source). Recompute s and C_own, refit M1. Placebo p = share of placebo gamma >= observed gamma. Same for beta_s. If the person's own past is what matters, the observed values must exceed the placebo distribution; if only generic relatability matters, they will not.

## Held-out prediction
50 splits (5-fold grouped by worker, repeated 10 times, seed 20261121). Fit M0 and M1 by OLS on training workers, predict held-out workers. Report mean RMSE (raw y), mean % improvement of M1 over M0, and share of splits where M1 is better. Required for support: mean improvement > 0 and M1 better in >= 80% of splits.

## Manipulation check (validity of s)
s must predict the worker's own similarity rating (similarity ~ s + src_imp + typ, clustered SEs). If the standardized coefficient's 95% CI includes 0, the text measure is not tracking relatedness to own experience and the H6 test is uninterpretable.

## Robustness (reported regardless)
- Cap of 10 imagined stories per worker (random subset, seed 20261121).
- Raw TF-IDF cosine (no SVD) in place of the SVD vectors.
- Ordinal check: same M1 with y dichotomized at >= 4.

## Limitations stated in advance
- The new experience is imagined, not lived.
- Writing order of recalled vs imagined tasks per worker is not recorded; the own experience itself predates the imagined task.
- Workers may have chosen which scenario tasks to accept.
- Text similarity is lexical-semantic (LSA), not a neural embedding; no embedding model is available offline here.

## Allowed changes after freezing
Only bug fixes that make the code do what this document says, each logged with a timestamp in the results file.
