# Analysis

Secondary analyses of public datasets, written and run by Claude (Anthropic) at Roddsi's direction. The raw data is not in this repository; see [`../data/README.md`](../data/README.md) for where to get it. Each script's header states what it tests and how to run it.

| Script | What it does | Output |
|---|---|---|
| `audit.py` | Recomputes the headline numbers (Hippocorpus, MainLife, experience sampling) from scratch | `audit_output.json` |
| `attention_test.py` | Goal diary: does significance draw attention, and attention build significance? | `attention_results.json` |
| `drawing_claims.py` | Tests claims read off the original sketch (z axis, x axis, the person, the path back) | `drawing_claims_results.json` |
| `h6_test.py` | First attempt at H6 in Hippocorpus, protocol v1 | `h6_results.json` |
| `h6_test_v2.py` | Second attempt, protocol v2 (cross-fitted similarity measure) | `h6_results_v2.json` |
| `h6_checks.py` | Validity checks on the H6 similarity measure | `h6_checks_output.txt` |
| `power_sim.py` | Power simulation for a future test of the attention half of H6 (no data needed) | `power_results.json` |

## Run

Python 3 with numpy, pandas, scipy, scikit-learn and openpyxl.

```bash
python3 -I audit.py <data dir>
python3 -I attention_test.py <data dir>/diarydt.csv attention_results.json
python3 -I drawing_claims.py <data dir> drawing_claims_results.json
python3 -I h6_test.py <data dir>/hcV3-stories.csv h6_results.json
python3 -I h6_test_v2.py <data dir>/hcV3-stories.csv h6_results_v2.json
python3 -I h6_checks.py <data dir>/hcV3-stories.csv
python3 -I power_sim.py power_results.json
```

`<data dir>` holds the files named in `../data/README.md`, with the MainLife spreadsheets in a `mainlife_data/` subfolder.

## What to keep in mind

- Everything here is correlational, and the effects are small.
- The H6 test came back null, but its similarity measure failed a validity check (`h6_checks_output.txt`), so it does not count for or against H6.
- The protocols in `../protocols/` were frozen and hashed before the H6 runs. That makes them dated plans, not a formal preregistration on a public registry.
