import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn SaaS App", layout="wide")

st.markdown("<h1 style='text-align:center;'>📊 Customer Churn SaaS App</h1>", unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📂 Upload Data", "🤖 Train Model", "🔮 Predict"]
)

# =========================
# SESSION STATE
# =========================
for key in ["df", "model", "scaler", "features"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# HOME
# =========================
if page == "🏠 Home":
    st.info("🚀 Welcome to Churn Prediction SaaS App")

    col1, col2, col3 = st.columns(3)
    col1.metric("Steps", "4")
    col2.metric("Models", "4 ML Models")
    col3.metric("Output", "Churn Prediction")


# =========================
# UPLOAD DATA
# =========================
elif page == "📂 Upload Data":
    st.subheader("Upload Dataset")

    file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    if file:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

        st.session_state.df = df

        st.success("Dataset Loaded Successfully 🎉")
        st.write(df.head())


# =========================
# TRAIN MODEL
# =========================
elif page == "🤖 Train Model":

    df = st.session_state.df

    if df is None:
        st.warning("Please upload dataset first.")
    else:
        st.subheader("Model Training")

        df = df.copy()

        # Clean columns
        drop_cols = ['CustomerID','Lat Long','Zip Code','City','State','Country',
                     'Churn Reason','Count']
        df = df.drop(columns=drop_cols, errors='ignore')

        df.columns = df.columns.str.replace(" ", "_")

        # Fix numeric column
        if "Total_Charges" in df.columns:
            df["Total_Charges"] = pd.to_numeric(df["Total_Charges"], errors="coerce")
            df["Total_Charges"] = df["Total_Charges"].fillna(df["Total_Charges"].median())

        X = df.drop(['Churn_Label','Churn_Value','Churn_Score'], axis=1, errors='ignore')
        y = df['Churn_Value']

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, y_train = SMOTE().fit_resample(X_train, y_train)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=200),
            "XGBoost": XGBClassifier(eval_metric="logloss"),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, prob)

            results[name] = auc

            st.write(f"### {name}")
            st.write("Accuracy:", round(acc, 3))
            st.write("ROC-AUC:", round(auc, 3))

        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]

        st.success(f"🏆 Best Model: {best_model_name}")

        st.session_state.model = best_model
        st.session_state.scaler = scaler
        st.session_state.features = X.columns


# =========================
# PREDICT PAGE
# =========================
elif page == "🔮 Predict":

    st.subheader("🔮 Predict Customer Churn")

    model = st.session_state.model
    scaler = st.session_state.scaler
    features = st.session_state.features

    if model is None:
        st.warning("Please train model first.")
    else:
        st.success("Model is ready 🚀")

        st.write("Enter customer details:")

        input_data = {}

        col1, col2 = st.columns(2)

        for i, col in enumerate(features):
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):
            input_array = np.array(list(input_data.values())).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")

            if pred == 1:
                st.error("⚠️ Customer WILL CHURN")
            else:
                st.success("✅ Customer will NOT churn")

            st.metric("Churn Probability", f"{prob:.2f}")