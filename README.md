<div align="center">

# **Telecom Customer Churn Prediction & Retention Analytics**

</div>


## 📊 Project Overview
Customer Churn Prediction capstone project leveraging machine learning models such as Decision Tree and Random Forest. The project includes data preprocessing, class imbalance handling (SMOTE), model evaluation using ROC-AUC, and SHAP-based explainability to generate actionable business insights.



## 📂 Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Telco customer churn: IBM dataset |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset) |
| **Total Observations** | 7043 observations |
| **Features** | 33 variables |
| **Target** | Binary — `1` (the customer left the company this quarter) / `0` (the customer remained with the company) |

---

## 🔬 Top 10 Important Features

| Feature | Description |
|---------|-------------|
| `Contract_Month-to-month` | Customer’s current contract type |
| `Tenure_Months` | Total amount of months that the customer has been with the company |
| `Total_Charges` | Customer’s total charges |
| `Monthly_Charges` | Customer’s current total monthly charge |
| `CLTV` | Customer Lifetime Value |
| `Latitude` |The latitude of the customer’s primary residence |
| `Longitude` | The longitude of the customer’s primary residence |
| `Online_Security_No` | Customer subscribes to an additional online security service provided by the company |
| `Tech_Support_No` | Customer subscribes to an additional technical support plan from the company with reduced wait times |
| `Payment_Method_Electronic check` | How the customer pays their bill |

---

## Model & Performance

Multiple classifiers were trained and evaluated. The best-performing model was selected for deployment.

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|:--------:|:---------:|:------:|:--------:|
| Random Forest | ~78% | ~58% | ~61% | ~60% |
| XGBoost | ~78% | ~57% | ~66% | ~61% |
| **Logistic Regression** ✅ | **~75%** | **~52%** | **~78%** | **~62%** |
| **AdaBoost Classifier** ✅ | **~76%** | **~53%** | **~75%** | **~62%** |


> In churn prediction, recall is usually prioritized because missing churners is more costly than targeting a few extra non-churners. That means Logistic Regression or AdaBoost will be the most useful.

---

## Methodology & Machine Learning Pipeline

### 1. Data Preprocessing

### 2. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is an important step in data analysis where we explore and visualise the data to understand its main features, find patterns and see how different variables are related.

### 3. Handling Class Imbalance (SMOTE)

### 4. Predictive Modeling

### 5. Explainable AI (SHAP)
### 6.Integrating Streamlit
