# heart-disease-ml-pipeline
An end-to-end Machine Learning classification pipeline using Python, Pandas, and Scikit-Learn
# 🫀 Heart Disease Risk Prediction & Analysis

An end-to-end Data Analysis and Machine Learning pipeline designed to analyze demographic and clinical factors influencing heart disease risk and build predictive models.

---

## 📌 Project Overview
Heart disease is one of the leading causes of mortality worldwide. Early detection of risk factors can significantly aid in preventive healthcare. 

The main objective of this project is to process and analyze clinical and lifestyle attributes from a dataset of 10,000 patients and develop a machine learning pipeline for binary classification (`Heart Disease Status`: Yes / No).

---

## 🛠️ Tech Stack & Libraries
* **Python 3.x**
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `train_test_split`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)

---

## 📊 Dataset Summary
* **Total Records:** 10,000 samples
* **Total Features:** 21 attributes (Demographics, Lifestyle Habits, Clinical Measurements)
* **Target Variable:** `Heart Disease Status` (Imbalanced distribution: ~80% No, ~20% Yes)

---

## 🔄 Project Workflow & Methodology

### 1. Data Exploration & Preprocessing (EDA)
* **Missing Value Handling:** 
  * Analyzed missing value percentages across features.
  * Imputed numerical features using `Median` and categorical features using `Mode`.
  * Treated features with significant missingness (e.g., `Alcohol Consumption` at ~25.6%) by introducing a distinct `'Missing'` category to preserve data integrity.
* **Feature Distribution & Skewness Analysis:** 
  * Inspected continuous variables (`Age`, `Blood Pressure`, `Cholesterol Level`, `BMI`, `Triglyceride Level`, `Sleep Hours`) via histograms and KDE plots.
* **Feature Categorization:**
  * **Binary Features:** `Smoking`, `Family Heart Disease`, `Diabetes`, `High Blood Pressure`, `High LDL Cholesterol`, `Low HDL Cholesterol`.
  * **Ordinal Features:** `Exercise Habits`, `Alcohol Consumption`, `Stress Level`, `Sugar Consumption`.
  * **Nominal Features:** `Gender`.

### 2. Data Splitting & Pipeline Setup
* **Stratified Train-Test Split:** Split the dataset into 80% training (8,000 samples) and 20% test (2,000 samples) sets using `stratify` to maintain target class proportions and prevent data leakage.
* **Preprocessing Pipeline:** Modularized feature transformations using `ColumnTransformer` and `Pipeline` for reproducible data preprocessing.

### 3. Model Training & Evaluation
* Trained classification algorithms on preprocessed features.
* Evaluated models using key classification metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

---

## 📈 Key Learnings & Challenges
* **Imbalanced Target Variable:** The dataset exhibits class imbalance (80/20 ratio), making baseline Accuracy an unreliable evaluation metric.
* **Data Leakage Prevention:** Ensured all imputation parameters and scaling transformations were fitted exclusively on the training set.
* **Model Baseline:** The dataset demonstrated weak linear separability among clinical features, indicating that predicting heart disease from lifestyle features alone requires non-linear techniques or synthetic oversampling.

---

## 🚀 Future Improvements
To enhance model performance in future iterations:
1. **Handling Class Imbalance:** Apply resampling strategies such as **SMOTE** (Synthetic Minority Over-sampling Technique) or adjust class weights.
2. **Advanced Modeling:** Experiment with ensemble models (e.g., **Random Forest**, **XGBoost**, **LightGBM**).
3. **Hyperparameter Tuning:** Implement `GridSearchCV` or `RandomizedSearchCV` for optimal parameter configuration.
4. **Feature Engineering:** Derive interaction terms (e.g., Ratio of LDL to HDL, BMI categories) to capture complex clinical patterns.

---

## ✒️ Author
* **Bahar Ghezavati** - *Data Analysis & ML Exploration*
