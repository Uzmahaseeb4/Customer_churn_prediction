import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_excel(r"C:\Users\asus\Downloads\Churn_pred\Customer_churn_prediction\Telco_customer_churn.xlsx")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# ==============================
# 2. CLEANING
# ==============================
df = df.drop(columns=[
    'CustomerID', 'Lat Long', 'Zip Code', 'City', 'State', 'Country',
    'Churn Reason', 'Count'
], errors='ignore')

df.columns = df.columns.str.replace(' ', '_')

df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
df['Total_Charges'] = df['Total_Charges'].fillna(df['Total_Charges'].median())

# ==============================
# 3. EDA (optional plots kept same)
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Churn_Value')
plt.title("Churn Distribution")
plt.show()

# ==============================
# 4. SPLIT FEATURES
# ==============================
X = df.drop(['Churn_Label', 'Churn_Value', 'Churn_Score'], axis=1, errors='ignore')
y = df['Churn_Value']

# One-hot encoding
X = pd.get_dummies(X, drop_first=False)
feature_names = X.columns.tolist()

# ==============================
# 5. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 6. SCALING
# ==============================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=feature_names)

# ==============================
# 7. SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("\nBefore SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_sm).value_counts().to_dict())

# ==============================
# 8. MODELS
# ==============================
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
lr = LogisticRegression(max_iter=1000, random_state=42)
ada = AdaBoostClassifier(n_estimators=100, random_state=42)

models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "Logistic Regression": lr,
    "AdaBoost": ada
}

results = {}

# ==============================
# 9. TRAIN + EVALUATE
# ==============================
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = auc

    print(f"\n========== {name} ==========")
    print("Accuracy:", acc)
    print("ROC-AUC :", auc)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# ==============================
# 10. BEST MODEL
# ==============================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

# ==============================
# 11. ROC CURVE
# ==============================
plt.figure(figsize=(8,6))

for name, model in models.items():
    prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")

plt.plot([0,1],[0,1],'--',color='gray')
plt.title("ROC Curve Comparison")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()

# ==============================
# 12. FEATURE IMPORTANCE (RF)
# ==============================
feat_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(8,5))
feat_imp.head(10).plot(kind='barh')
plt.title("Top Features (Random Forest)")
plt.gca().invert_yaxis()
plt.show()

# ==============================
# 13. SHAP (FIXED CLEAN VERSION)
# ==============================
explainer = shap.TreeExplainer(xgb)

X_test_reset = X_test_scaled.reset_index(drop=True)

shap_values = explainer.shap_values(X_test_reset)

# small sample for plots
plot_idx = np.random.RandomState(42).choice(len(X_test_reset), 200, replace=False)
X_plot = X_test_reset.iloc[plot_idx]
shap_plot = shap_values[plot_idx]

shap.summary_plot(shap_plot, X_plot, plot_type="dot")
shap.summary_plot(shap_plot, X_plot, plot_type="bar")

# ==============================
# 14. FINAL ROC (BEST MODEL ONLY)
# ==============================
fpr, tpr, _ = roc_curve(y_test, y_prob_best)
auc = roc_auc_score(y_test, y_prob_best)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.show()

print("Final ROC-AUC:", auc)