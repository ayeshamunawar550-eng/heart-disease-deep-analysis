# Heart Disease Classification — Random Forest vs Gradient Boosting

Predicting the presence of heart disease from clinical and diagnostic-test
attributes using the classic **UCI Heart Disease (Cleveland)** data set,
comparing two tree-based ensemble models: Random Forest and Gradient
Boosting.

## What's in here

The notebook walks through exploratory data analysis, a data-quality pass
that catches a duplicate record and a handful of invalid category codes,
hyperparameter tuning for both models, evaluation with cross-validation and
bootstrap confidence intervals, a statistical test for whether the two
models actually differ, and permutation-based feature importance.

## Results

After cleaning (dropping 1 duplicate row and 6 rows with out-of-range `ca`/`thal`
codes), the data set has 296 patients. Both models were tuned with a
randomized search under 5-fold cross-validation rather than hand-picked
hyperparameters.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC (test) | ROC-AUC (5-fold CV) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.850 | 0.829 | 0.906 | 0.866 | 0.923 (95% CI 0.846–0.980) | 0.924 ± 0.024 |
| Gradient Boosting | 0.833 | 0.806 | 0.906 | 0.853 | 0.898 (95% CI 0.806–0.966) | 0.916 ± 0.017 |

Random Forest comes out ahead on every metric, but the two models only
disagreed on 3 patients out of 60 in the test set, and McNemar's test on
those disagreements gives p = 1.0 — with this few cases to compare, the gap
between the models isn't something you can call statistically real. Both are
reasonable choices here; Random Forest is very slightly the better bet.

By permutation importance, the features that consistently mattered most
across both models were **`ca`** (vessels colored by fluoroscopy), **`thal`**
(thalassemia), **`cp`** (chest pain type), **`sex`**, and **`exang`**
(exercise-induced angina) — all recognized cardiovascular risk indicators.

## Scientific review

| Aspect | Assessment | Notes |
|---|---|---|
| Sample size | Weak point | 296–303 patients from a single clinical site is small for a machine learning benchmark. Any single train/test split carries real sampling noise, which is why cross-validation and bootstrap confidence intervals are used alongside the plain test-set numbers rather than instead of them. |
| Data cleaning | Fixed | The raw file had one exact duplicate patient record and six rows with `ca`/`thal` values outside the documented category ranges (an artifact of how missing values were originally encoded in this data mirror). Left in place, the duplicate risked leaking the same patient into both train and test, and the invalid codes would have taught the model a meaningless split. Both are now caught and removed with an explicit check before modeling. |
| Class balance | Strong point | About 54% positive across the cleaned data, close enough to balanced that accuracy is a meaningful metric on its own, though precision, recall, F1, and ROC-AUC are still reported since a missed diagnosis is costlier than a false alarm. |
| Missing values | Strong point | Genuinely zero `NaN` values across all columns — verified directly rather than assumed. |
| Hyperparameter selection | Fixed | The original model settings were picked by hand with no tuning process behind them. Both models now go through a randomized search with cross-validation, scored on ROC-AUC, with a deliberately modest grid given how little training data there is. |
| Feature scaling | Not needed | Tree-based models split on feature ordering, not magnitude, so no scaler is required. This was correctly left out. |
| Categorical encoding | Fixed | `cp`, `restecg`, `slope`, and `thal` are nominal categories but were fed to the trees as plain integers, implicitly imposing an ordering they don't really have. A one-hot encoded version of both models was tuned and evaluated the same way as a sensitivity check — CV ROC-AUC moved by less than 0.001 for both models, and test ROC-AUC stayed within the range already covered by the bootstrap confidence intervals. The encoding choice isn't a source of bias here, and that's now demonstrated rather than assumed. |
| Feature importance method | Fixed | The original notebook used impurity-based `feature_importances_` while labeling the section "permutation importance" — those are two different things, and impurity-based scores are known to be biased toward high-cardinality numeric features. The notebook now actually computes permutation importance. |
| Model comparison rigor | Fixed | Random Forest scoring higher than Gradient Boosting was previously reported without asking whether the gap was meaningful. McNemar's test is now run on the test-set disagreements, and it shows the difference isn't statistically distinguishable from chance at this sample size — an honest and fairly common outcome for two well-tuned models on the same small data set. |
| Uncertainty reporting | Fixed | ROC-AUC was reported as a single number with no sense of how much it would move on a different sample. A bootstrap confidence interval is now computed from 1,000 resamples of the test set. |
| Reproducibility | Fixed | `random_state` is fixed everywhere models or splits are created, and `requirements.txt` now pins exact package versions rather than loose lower bounds, so the environment that produced these numbers can be recreated exactly. |
| Narrative structure | Strong point | The notebook reads in a clear line: explore the data, check it for problems, build and tune models, evaluate them properly, explain what's driving the predictions, and state plainly what the analysis can't claim. |
| Clinical framing | Strong point | The write-up doesn't overstate what a benchmark exercise on 300 patients from one site in the 1980s can tell you — it's framed as a modeling comparison, not a diagnostic tool, which is the right level of caution for this kind of data set. |

## Data

`heart_disease_uci.csv` — a standard redistribution of the [UCI Machine
Learning Repository's Heart Disease data set](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland Clinic Foundation subset, Detrano et al., 1989). 303 rows as
distributed, 13 predictors, 1 binary target. The predictors cover
demographics (age, sex), symptoms (chest pain type, exercise-induced angina),
resting measurements (blood pressure, cholesterol, ECG), and stress-test
results (max heart rate, ST depression, vessels colored by fluoroscopy).

## Getting started

```bash
git clone <your-repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook heart-disease-rf-gb-analysis.ipynb
```

Then run all cells top to bottom (Kernel → Restart & Run All). The
hyperparameter search takes a minute or two on a laptop.

## Repository structure

```
.
├── heart-disease-rf-gb-analysis.ipynb   # main analysis notebook
├── heart_disease_uci.csv                # data set
├── requirements.txt                     # pinned Python dependencies
└── README.md
```

## Limitations

This is a benchmark/educational comparison of two ensemble methods, not a
validated diagnostic tool. The data set is small, comes from a single site,
and is decades old, so any conclusions apply to this specific cohort rather
than patients in general.

## License / attribution

Data set: Detrano, R., et al. (1989). Heart Disease Data Set. UCI Machine
Learning Repository. Used here for educational/benchmark purposes.
