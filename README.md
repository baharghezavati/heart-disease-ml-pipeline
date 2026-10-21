# Heart Disease Classification: ML Pipeline and Model Benchmark

An end-to-end machine learning workflow in Python (pandas, scikit-learn, XGBoost): leakage-free preprocessing, stratified cross-validation, hyperparameter tuning and an honest comparison of five model families on an imbalanced binary classification problem.

> **Key finding:** the dataset appears to be synthetic and contains almost no usable signal. **Every model scores ROC-AUC of about 0.51**, the same as models trained on randomly shuffled labels. The project is therefore a demonstration of the *workflow and of careful evaluation*, not a predictive model. For an analysis on real clinical data, see my other project: [heart-disease-logistic-regression](https://github.com/baharghezavati/heart-disease-logistic-regression) (UCI Cleveland data, R).

## Dataset
- Kaggle heart disease dataset, **10,000 patients, 20 features + target** (`Heart Disease Status`: 80% No, 20% Yes). The CSV is included as `heart_disease.csv`.
- Features: demographics (age, gender), clinical measurements (blood pressure, cholesterol, BMI, triglycerides, fasting blood sugar, CRP, homocysteine), lifestyle (exercise, smoking, alcohol, sleep, sugar, stress) and history (diabetes, family history, high blood pressure, HDL/LDL flags).
- Missing values: about 0.2-0.3% per column, except `Alcohol Consumption` (about 26%).

## Workflow
1. **Stratified 80/20 split** (8,000 train / 2,000 test), so the 80/20 class ratio is kept in both parts.
2. **Data checks:** missing values, skewness, IQR outliers (none), value ranges.
3. **Preprocessing pipeline** (`ColumnTransformer`), fitted only on training folds so there is no data leakage:
   - numeric: median imputation + standard scaling
   - binary (Yes/No) and ordinal (Low/Medium/High): most-frequent imputation + ordinal encoding
   - nominal (gender, alcohol): constant `Missing` category + one-hot encoding. Missing alcohol is kept as its own category rather than imputed, because about 26% of values are missing.
4. **Models**, each with and without class weighting: Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM (RBF).
5. **Tuning and evaluation:** `GridSearchCV` (scoring: PR-AUC) and 5-fold stratified cross-validation on the training set. Metrics: precision, recall, specificity, F1, ROC-AUC and PR-AUC. Accuracy is not used as the main metric because always predicting "No" already gives 80%.

## Results
Mean of 5-fold cross-validation on the training set. Positive class = heart disease (`Yes`).

| Model | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.000 | 0.000 | 1.000 | 0.000 | 0.511 | 0.210 |
| Logistic Regression (balanced) | 0.204 | 0.499 | 0.513 | 0.289 | 0.511 | 0.210 |
| Decision Tree | 0.238 | 0.006 | 0.996 | 0.012 | 0.504 | 0.206 |
| Decision Tree (balanced) | 0.214 | 0.438 | 0.593 | 0.281 | 0.517 | 0.211 |
| Random Forest | 0.000 | 0.000 | 1.000 | 0.000 | 0.504 | 0.207 |
| Random Forest (balanced) | 0.000 | 0.000 | 1.000 | 0.000 | 0.515 | 0.217 |
| XGBoost | 0.000 | 0.000 | 1.000 | 0.000 | 0.509 | 0.214 |
| XGBoost (weighted) | 0.204 | 0.401 | 0.608 | 0.270 | 0.511 | 0.215 |
| SVM (RBF) | 0.000 | 0.000 | 1.000 | 0.000 | 0.509 | 0.211 |
| SVM (RBF, balanced) | 0.000 | 0.000 | 1.000 | 0.000 | 0.514 | 0.213 |

**How to read this**
- A random classifier has ROC-AUC 0.50 and PR-AUC equal to the positive rate (0.20). All models are within about 0.02 of these values.
- Unweighted models "reach" 80% accuracy only by predicting "No" for almost everyone (recall near 0, specificity near 1).
- Class weighting raises recall to 0.40-0.50, but precision stays near 0.20, the base rate, which is what random guessing gives.
- On the held-out test set, the two baseline logistic regressions reach ROC-AUC of about 0.47. The tuned models were compared by cross-validation only.

## Why I believe the data is synthetic
- Numeric features sit in round, hard-bounded ranges (age 18-80, blood pressure 120-180, cholesterol 150-300, triglycerides 100-400, fasting blood sugar 80-160, sleep 4-10 hours) with skewness close to zero.
- Binary and categorical features are split close to 50/50.
- No feature is clearly related to the target. 
- **Shuffled-label check** (last code cell of the notebook): real labels give a mean cross-validated ROC-AUC of 0.511; randomly shuffled labels give 0.492 to 0.526. The real result is indistinguishable from noise.

## Limitations
- Because the data has no signal, the model comparison cannot show which algorithm is better.
- Cross-validation scores of the tuned models are computed on the same folds used for tuning, so they are slightly optimistic in general. This does not change the conclusion here.
- Only the two baseline logistic regressions were scored on the held-out test set.

## Next steps
- Repeat the same pipeline on a real clinical dataset.
- Add feature importance and probability calibration once real signal exists.

## Repository contents
| File | Description |
|---|---|
| `heart_disease_ml_pipeline.ipynb` | Full analysis notebook (code and outputs) |
| `heart_disease.csv` | Dataset (10,000 rows) |
| `requirements.txt` | Python packages |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook heart_disease_ml_pipeline.ipynb
```
Keep `heart_disease.csv` in the same folder as the notebook. The full run takes several minutes, mostly the SVM grid search.

## Tools
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Jupyter.
