<div align="center">

# **Telecom Customer Churn Prediction & Retention Analytics**

</div>

| Name | Reg No | Course |
| --- | --- | --- |
| [Member A Name] | [Reg No] | [Course] |
| [Member B Name] | [Reg No] | [Course] |
| Arun P S| 253206 | MSc DataScience and BioAI |
## 📊 Project Overview
Customer Churn Prediction capstone project leveraging machine learning models such as Random Forest, XG Boost. In addition Logistic regression and adaboost were also implemented. The project includes data preprocessing, class imbalance handling (SMOTE), model evaluation using ROC-AUC, and SHAP-based explainability to generate actionable business insights.



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


> In churn prediction, recall is usually prioritised because missing churners is more costly than targeting a few extra non-churners. That means Logistic Regression or AdaBoost will be the most useful.

---

## Methodology & Machine Learning Pipeline

### 1. Data Preprocessing
Data preprocessing involved cleaning the dataset, handling missing values, encoding categorical variables, and scaling features to prepare the data for modeling.

### 2. Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) is an important step in data analysis where we explore and visualise the data to understand its main features, find patterns and see how different variables are related.

### 3. Handling Class Imbalance (SMOTE)
SMOTE (Synthetic Minority Over-sampling Technique) was used in the model to balance the dataset by generating synthetic samples for the minority class, improving the model’s ability to learn from underrepresented data and enhancing overall performance.

### 4. Predictive Modeling
Predictive models, including XGBoost, Adaboost, Logistic Regression, and Random Forest, were implemented to improve accuracy and compare performance.

### 5. Explainable AI (SHAP)
SHAP was used to interpret the model by explaining the contribution of each feature to the predictions.

### 6.Integrating Streamlit
Streamlit was used to develop an interactive web application for deploying and visualising the model results.


## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Clone or download this repository
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate         # macOS/Linux
   venv\Scripts\activate             # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To launch the Streamlit web app:
```bash
streamlit run churn_rev_app.py
```

The app will open at `http://localhost:8501`.

---

## 🌐 Live Deployment

- **Streamlit Cloud**: https://customerchurnprediction-23z6zwjak8kgu2sktpvqrm.streamlit.app/

---

## 📁 Repository Structure

```
Customer_churn_prediction-main/
├── aap.py                          # Helper functions and utilities
├── churn_rev_app.py                # Main Streamlit application
├── IBM_CHURN_PRED.ipynb            # Complete analysis notebook
├── IBM_CHURN_PRED.PY               # Standalone Python pipeline
├── README.md                        # This file
├── requirements.txt                # Project dependencies
├── Telco_customer_churn.xlsx       # Dataset file
└── individual_profiles/            # Team member GitHub activity screenshots
```

---


## 📝 License

This project is provided for educational purposes as part of a capstone project.

