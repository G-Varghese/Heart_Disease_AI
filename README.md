Project Title: Explainable Machine Learning (ML) for predicting the presence of heart disease

1. Project Overview

This project develops a prototype machine-learning solution for predicting the presence of heart disease using demographic and clinical characteristics from the UCI Cleveland Heart Disease dataset. The project investigates whether machine-learning classification models can distinguish between patients with and without heart disease and evaluates the models from both predictive-performance and responsible-AI perspectives.

2.  Research Question

-	How accurately do ML models predict the presence of heart disease from demographic and clinical characteristics?
-	Can SHAP-based explainability identify the key features contributing to individual and overall predictions?


3. Dataset

The project uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The dataset contains:

- 303 patient instances
- 13 predictor features
- 1 target variable (`num`)

The dataset is retrieved directly from the UCI repository using the `ucimlrepo` Python package.

4. Predictor Features

The 13 predictor variables are:

- `age` – age
- `sex` – sex
- `cp` – chest pain type
- `trestbps` – resting blood pressure
- `chol` – cholesterol
- `fbs` – fasting blood sugar
- `restecg` – resting electrocardiographic results
- `thalach` – maximum heart rate achieved
- `exang` – exercise-induced angina
- `oldpeak` – ST depression
- `slope` – slope of the peak exercise ST segment
- `ca` – number of major vessels
- `thal` – thalassemia-related category

5. Target Variable

The original UCI target variable ranges from 0 to 4. The target was converted into a binary classification outcome:

- 0 → No heart disease
- 1–4 → Presence of heart disease

The resulting target distribution is:

- No disease: 164 patients
- Disease: 139 patients

5. Methodology

The project follows an end-to-end machine-learning workflow:

1. Dataset loading and inspection
2. Target variable preparation
3. Missing-value inspection
4. Duplicate checking
5. Stratified train-test split
6. Missing-value imputation
7. Categorical variable encoding
8. Numerical feature scaling
9. Logistic Regression training
10. Random Forest training
11. Model evaluation
12. SHAP explainability analysis
13. Demographic subgroup analysis
14. False-negative analysis
15. Responsible AI assessment

6. Data Preprocessing

The dataset contains missing values in `ca` and `thal`.

To reduce data leakage:

- preprocessing was fitted using the training data only
- numerical features use median imputation
- categorical features use most-frequent-value imputation
- categorical variables are one-hot encoded
- numerical variables are standardised using `StandardScaler`

An 80/20 stratified train-test split was used with `random_state=42`.

7. Machine Learning Models

Two classification models were developed:

# Logistic Regression

Logistic Regression was used as an interpretable baseline classification model.

# Random Forest

Random Forest was used as a tree-based ensemble model containing 100 decision trees.

**Model Evaluation**

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

**Final Results**

| Model               | Accuracy | Precision | Recall | F1-score |    ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ---------: |
| Logistic Regression |   0.8689 |    0.8333 | 0.8929 |   0.8621 | **0.9621** |
| Random Forest       |   0.8689 |    0.8333 | 0.8929 |   0.8621 |     0.9443 |

Both models produced the same class predictions on the test dataset. Logistic Regression achieved the higher ROC-AUC and was therefore selected for subsequent Explainable AI analysis.

8. Explainable AI

SHAP (SHapley Additive exPlanations) was used to analyse the selected Logistic Regression model. The project includes:

- Global SHAP feature importance
- SHAP beeswarm analysis
- Local explanation using a SHAP waterfall plot for an individual test patient

These explanations were used to investigate how processed features contributed to model predictions.

9.  Responsible AI Analysis

Responsible AI considerations included:

Missing Data

Missing values were identified and handled within the leakage-safe preprocessing pipeline.

Class Balance

The binary target contained 164 patients without disease and 139 patients with disease. Stratified sampling was used during train-test splitting.

False Negatives

The selected Logistic Regression model produced:

* 3 false negatives
* False-negative rate: 0.1071 (10.71%)

False negatives are important because they represent patients with disease who were incorrectly classified as having no disease.

10. Demographic Subgroup Analysis

Model performance was examined across:

* Sex subgroups
* Age groups

The results showed differences in some evaluation metrics between subgroups. These findings should be interpreted cautiously because the subgroup sample sizes were relatively small.

11. Limitations

This is a prototype-level research project and should not be considered a clinical diagnostic system. Limitations include:

* The dataset contains only 303 patients.
* The dataset represents the Cleveland cohort and may not generalise to other populations.
* Demographic information is limited.
* Some subgroup sample sizes are small.
* The model still produces false-negative predictions.
* SHAP explanations describe model behaviour and should not be interpreted as causal explanations.
* External clinical validation has not been performed.

12. Project Structure

Heart_Disease_AI/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── explainability.py
│   └── responsible_ai.py
│
└── results/
    ├── figures/
    │   ├── roc_curve_comparison.png
    │   ├── model_performance_comparison.png
    │   ├── shap_bar.png
    │   ├── shap_beeswarm.png
    │   └── shap_waterfall_patient_0.png
    │
    └── tables/
        ├── model_results.csv
        ├── logistic_confusion_matrix.csv
        ├── random_forest_confusion_matrix.csv
        ├── shap_feature_importance.csv
        ├── individual_patient_prediction.csv
        ├── sex_subgroup_results.csv
        ├── age_subgroup_results.csv
        └── false_negative_analysis.csv


13.  Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The local `.venv` directory is excluded from version control and is not required to be uploaded to GitHub.

14.  Running the Project

Run the scripts from the project root in the following order:

```bash
python src/preprocessing.py
```

```bash
python src/train_models.py
```

```bash
python src/evaluate_models.py
```

```bash
python src/explainability.py
```

```bash
python src/responsible_ai.py
```

The scripts generate model evaluation tables and figures in the `results/` directory.

15. Notebook

The Jupyter notebook provides the detailed development and analysis process, including:

- Dataset inspection
- Preprocessing
- Model development
- Model evaluation
- SHAP analysis
- Responsible AI analysis
- Results and discussion


16. Project Status

This repository contains a prototype-level healthcare machine-learning workflow covering data preparation, predictive modelling, model evaluation, explainability, subgroup analysis, and responsible-AI considerations.
