# Evaluation methodology: paired bootstrap, cross-fit calibration, ECE null, predictive dev band

Code: `judgelab.evalstats` (numpy only). Demo on synthetic data: `python examples/demo_eval_methods.py`.

## Context

The methods were used to evaluate a fine-tuned multi-label classifier against a baseline:
- the classifier is a published multilingual encoder plus about 18 M trained parameters;
- it chooses which of 5 checks ("routes") a flagged passage needs;
- the evaluation used out-of-fold (OOF) predictions on 828 labelled items, from 5 folds and 2 training seeds.

**The labels are model-adjudicated, never human-checked.** Every number below measures agreement with those labels.
The source reports carry the label EXPLORATORY_NOT_GATE. The held-out dev split was not opened. The data, labels and
weights are not published. `synthetic_oof` generates demo data of the same shape.

## Methods

| method | function | why |
|---|---|---|
| Mann-Whitney AUC, ties = 1/2; macro over routes | `auc`, `macro_auc` | ranking quality per route; NaN labels skipped per route |
| paired item bootstrap | `paired_bootstrap` | the same resampled items for every arm and route, so the difference's interval excludes resample noise |
| cross-fitted Platt scaling | `platt_fit`, `crossfit_platt` | recalibrated probabilities stay out-of-sample (fit on 4 folds, predict the 5th) |
| ECE with a null band | `ece`, `ece_null`, `calibration_report` | ECE is never 0 on finite data. RECALIBRATE only if the raw ECE exceeds the 95th percentile of the ECE that perfectly calibrated probabilities would show (1,000 Bernoulli simulations) |
| Wilson interval | `wilson` | precision at a threshold, with an honest interval |
| predictive dev band | `predictive_band` | expected range of ONE future read on k items: D_full + D(k drawn with replacement) - D(full bootstrap). Subsets of the n items already used overlap the estimate, so their spread is too small by about sqrt((1+k/n)/(1-k/n)) |

## Results on the real study (MEASURED; 2,000 paired resamples; fixed seed)

**Ranking** (macro OOF AUC):

| arm | macro AUC |
|---|---|
| baseline | 0.9233 |
| model, seed A | 0.9645 |
| model, seed B | 0.9648 |
| two-seed mean | 0.9700 |

- A minus baseline: +0.041, 95% [+0.031, +0.052].
- B minus baseline: +0.042 [+0.031, +0.053].
- Seed mean minus baseline: +0.047 [+0.036, +0.058].
- All five routes improve; the smallest gain is +0.014 [+0.002, +0.028].
- A pre-written two-seed adopt rule passed all 8 conditions.

Caveats:
- The seed mean is a two-model ensemble compared against a one-model baseline, which favours the ensemble.
- The bootstrap resamples items only. It ignores training-run variance, and near-duplicate items would make the
  intervals optimistic.

**Calibration** (two-seed mean):
- All 5 of 5 routes need recalibration: raw ECE is above the null p95. Example: one route had ECE 0.126 against a
  null p95 of 0.013.
- Platt slopes were 0.27-0.41, so the raw log-odds are about 2.4-3.8x too large (over-confident).
- 31-90% of the predictions are saturated (p within 1e-4 of 0 or 1).
- After cross-fit Platt scaling, ECE is within the null band on all 5 routes (example: 0.035 against 0.044), and
  log-loss falls to 0.47-0.62 of the raw value.

Reading (DERIVED): the raw scores are usable for ranking and as a binary decision, not as probabilities.

**Predictive band for one 129-item dev read** (seed mean minus baseline, 20,000 replicates):
- 95% range +0.018 to +0.080, centre +0.047; seed A alone +0.013 to +0.074.
- The plain-subset band is narrower: +0.022 to +0.075. That is the understatement the predictive form corrects.
- Single routes are not tested: their 95% widths are 0.06-0.23.

The band was written down before the dev read, as a shift warning, not a pass mark. The dev read had not happened at
the time of writing.

## Limits

- Item sampling only: no dataset shift, no final-model-versus-fold-model change.
- Model-adjudicated labels, as stated above.
