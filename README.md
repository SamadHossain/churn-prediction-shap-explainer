# Customer Churn Prediction & Explainability

**Mohammad Samad Hossain**  
**PGD in Data Science | United International University (UIU)**

## 📌 Project Overview

This project develops a machine learning workflow to predict customer churn and understand the factors driving churn decisions.

It demonstrates an end-to-end data science workflow, covering **data cleaning, exploratory data analysis, feature engineering, machine learning, model evaluation, threshold optimization, customer segmentation, and SHAP-based explainability**.

## 🎯 Objectives

- Clean and prepare customer data for machine learning
- Explore patterns associated with customer churn
- Build and compare predictive models
- Handle class imbalance using SMOTE
- Optimize model performance through cross-validation and hyperparameter tuning
- Identify the most important factors influencing churn
- Segment customers based on behavioral characteristics
- Translate model results into actionable business insights

## 🔎 Workflow

**Data Cleaning → EDA → Feature Engineering → Modeling → Evaluation → Threshold Optimization → SHAP Explainability → Customer Segmentation**

## 🤖 Machine Learning

Two main approaches were evaluated:

- Logistic Regression as a baseline model
- XGBoost for nonlinear prediction

The project also uses:

- Stratified train/test splitting
- SMOTE for class imbalance
- Cross-validation
- Randomized hyperparameter search
- ROC-AUC evaluation
- Probability threshold tuning

## 🧠 Model Explainability

SHAP (SHapley Additive exPlanations) was used to understand:

- Which features have the greatest influence on churn
- How individual features increase or decrease churn probability
- Why specific customers are predicted to churn

## 👥 Customer Segmentation

K-Means clustering was used to identify distinct customer segments based on customer characteristics and behavior.

## 🛠️ Technologies

**Python • Pandas • NumPy • Matplotlib • Seaborn • Scikit-learn • XGBoost • SHAP • Imbalanced-learn • Joblib**

## 📂 Files

- `01_churn_prediction.ipynb` — Complete analysis and machine learning workflow
- `Telco-Customer-Churn.csv` — Dataset
- `requirements.txt` — Python dependencies

## 📊 Key Skills Demonstrated

**Data Cleaning | Data Preprocessing | Exploratory Data Analysis | Feature Engineering | Machine Learning | Imbalanced Data | Model Evaluation | Hyperparameter Tuning | SHAP Explainability | Customer Segmentation**

---

### 👤 About the Author

**Mohammad Samad Hossain** is currently pursuing a **Postgraduate Diploma (PGD) in Data Science at United International University (UIU)** and is developing practical skills in data analysis, data preprocessing, machine learning, and business analytics.
