# ==============================
# 1. IMPORTS
# ==============================
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_excel("/content/Telco_customer_churn.xlsx")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# ==============================
# 3. DATA CLEANING
# ==============================
# Drop irrelevant / location-based / leakage columns
df = df.drop(columns=[
    'CustomerID', 'Lat Long', 'Zip Code', 'City', 'State', 'Country',
    'Churn Reason', 'Count'
], errors='ignore')

df.columns = df.columns.str.replace(' ', '_')

# Fix Total_Charges — sometimes read as string due to spaces
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
df['Total_Charges'] = df['Total_Charges'].fillna(df['Total_Charges'].median())

# ==============================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================
print("\n--- Starting Exploratory Data Analysis ---")

# a. Target Variable Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn_Value', palette='viridis')
plt.title('Churn Distribution')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# b. Categorical Features vs Churn
categorical_cols = ['Gender', 'Senior_Citizen', 'Partner', 'Dependents', 
                    'Contract', 'Payment_Method', 'Internet_Service']
existing_cat_cols = [c for c in categorical_cols if c in df.columns]

for col in existing_cat_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, hue='Churn_Value', palette='Set2')
    plt.title(f'Churn by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# c. Numerical Features Distribution
numerical_cols = ['Tenure_Months', 'Monthly_Charges', 'Total_Charges']
existing_num_cols = [c for c in numerical_cols if c in df.columns]

for col in existing_num_cols:
    plt.figure(figsize=(8, 4))
    sns.kdeplot(data=df, x=col, hue='Churn_Value', fill=True, palette='magma', common_norm=False)
    plt.title(f'{col} Distribution by Churn')
    plt.tight_layout()
    plt.show()

# d. Correlation Heatmap for Numerical Features
if existing_num_cols:
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[existing_num_cols + ['Churn_Value']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

# ==============================
# 5. FEATURE / TARGET SPLIT
# Drop leakage columns:
#   - Churn_Label : string version of target
#   - Churn_Score : derived directly from churn probability
#   - Churn_Value : the target itself
# ==============================
X = df.drop(['Churn_Label', 'Churn_Value', 'Churn_Score'], axis=1, errors='ignore')
y = df['Churn_Value']

print(f"\nTarget distribution:\n{y.value_counts()}")
print(f"Churn rate: {y.mean()*100:.1f}%")

# ==============================
# 6. ENCODING
# drop_first=False for full interpretability across all models
# ==============================
X = pd.get_dummies(X, drop_first=False)
feature_names = X.columns.tolist()

# ==============================
# 7. TRAIN / TEST SPLIT
# stratify=y ensures same churn ratio in both sets
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 8. SCALING
# Fit only on train — transform both
# Keep as DataFrame to preserve feature names (needed for SHAP)
# ==============================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test),  columns=feature_names)

# ==============================
# 9. SMOTE (on train only — never touch test)
# Applied after scaling so synthetic points are in the same feature space
# ==============================
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"\nClass distribution BEFORE SMOTE: {dict(y_train.value_counts())}")
print(f"Class distribution AFTER  SMOTE: {dict(pd.Series(y_train_sm).value_counts())}")

# ==============================
# 10. RANDOM FOREST
# ==============================
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_sm, y_train_sm)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf  = rf.predict_proba(X_test_scaled)[:, 1]

print("\n========== Random Forest ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ==============================
# 11. XGBOOST
# eval_metric set explicitly to suppress warnings
# use_label_encoder=False for compatibility with newer XGBoost versions
# ==============================
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_sm, y_train_sm)

y_pred_xgb = xgb.predict(X_test_scaled)
y_prob_xgb  = xgb.predict_proba(X_test_scaled)[:, 1]

print("\n========== XGBoost ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# ==============================
# 12. LOGISTIC REGRESSION 
# ==============================
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_sm, y_train_sm)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print("\n========== Logistic Regression ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_lr))
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ==============================
# 13. ADABOOST CLASSIFIER 
# ==============================
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train_sm, y_train_sm)

y_pred_ada = ada.predict(X_test_scaled)
y_prob_ada = ada.predict_proba(X_test_scaled)[:, 1]

print("\n========== AdaBoost Classifier ==========")
print("Accuracy :", accuracy_score(y_test, y_pred_ada))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob_ada))
print(confusion_matrix(y_test, y_pred_ada))
print(classification_report(y_test, y_pred_ada))

# ==============================
# 14. CONFUSION MATRICES (side by side)
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, y_pred, title in zip(
    axes,
    [y_pred_rf, y_pred_xgb,y_pred_lr,y_pred_ada],
    ["Random Forest", "XGBoost","Logistic Regression","ADABOOST CLASSIFIER"]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()

# ==============================
# 15. ROC CURVE COMPARISON
# ==============================
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_lr,  tpr_lr,  _ = roc_curve(y_test, y_prob_lr)
fpr_ada, tpr_ada, _ = roc_curve(y_test, y_prob_ada)

auc_rf  = roc_auc_score(y_test, y_prob_rf)
auc_xgb = roc_auc_score(y_test, y_prob_xgb)
auc_lr  = roc_auc_score(y_test, y_prob_lr)
auc_ada = roc_auc_score(y_test, y_prob_ada)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf,  tpr_rf,  label=f"Random Forest          AUC = {auc_rf:.3f}")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost                AUC = {auc_xgb:.3f}")
plt.plot(fpr_lr,  tpr_lr,  label=f"Logistic Regression    AUC = {auc_lr:.3f}")
plt.plot(fpr_ada, tpr_ada, label=f"AdaBoost               AUC = {auc_ada:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — All Models")
plt.legend()
plt.tight_layout()
plt.show()

# ==============================
# 14. FEATURE IMPORTANCE (Random Forest)
# ==============================
feat_imp = pd.Series(rf.feature_importances_, index=feature_names)
feat_imp = feat_imp.sort_values(ascending=False)

print("\nTop 10 Important Features (Random Forest):")
print(feat_imp.head(10))

plt.figure(figsize=(8, 5))
feat_imp.head(10).plot(kind='barh')
plt.title("Top 10 Feature Importances — Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==============================
# 15. SHAP ANALYSIS
# TreeExplainer is exact and fast for both RF and XGBoost.
# We use XGBoost for SHAP (typically sharper explanations).
# FIX: Compute SHAP on the FULL test set (not a 200-row sample).
#      This prevents IndexError in Section 16 when high-risk customers
#      fall outside the sampled 200 rows.
# For plots we use a 200-row subset so the charts remain readable.
# ==============================
explainer   = shap.TreeExplainer(xgb)

# Full test SHAP values — used for retention insights (Section 16)
X_test_reset = X_test_scaled.reset_index(drop=True)
shap_values  = explainer.shap_values(X_test_reset)   # shape: (n_test, n_features)

# 200-row subset — used only for summary plots (readability)
plot_idx     = np.random.RandomState(42).choice(len(X_test_reset), 200, replace=False)
X_plot       = X_test_reset.iloc[plot_idx]
shap_plot    = shap_values[plot_idx]

# Summary beeswarm plot — shows feature impact direction and magnitude
shap.summary_plot(shap_plot, X_plot, plot_type="dot")

# Bar plot — mean absolute SHAP value per feature
shap.summary_plot(shap_plot, X_plot, plot_type="bar")

# ==============================
# 16. CROSS VALIDATION (XGBoost Pipeline)
# ImbPipeline ensures SMOTE runs inside each fold — no leakage.
# StratifiedKFold is the default for classification in cross_val_score.
# ==============================
cv_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote',  SMOTE(random_state=42)),
    ('xgb',    XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
               ))
])

cv_scores = cross_val_score(cv_pipeline, X, y, cv=5, scoring='roc_auc')

print("\n========== Cross Validation (XGBoost) ==========")
print("Fold AUC Scores :", cv_scores)
print("Mean  CV AUC    :", round(cv_scores.mean(), 4))
print("Std   CV AUC    :", round(cv_scores.std(),  4))

# ==============================
# 17. ACTIONABLE CUSTOMER RETENTION INSIGHTS
# Identify high-risk customers from the test set.
# Use SHAP to find the top churn driver for each at-risk customer.
# Translate those drivers into business retention actions.
# ==============================

# --- Map top SHAP drivers to retention strategies ---
RETENTION_ACTIONS = {
    'Contract':          "Offer a discounted 1- or 2-year contract to lock in the customer.",
    'Tenure':            "Assign a dedicated account manager to new/short-tenure customers.",
    'Monthly_Charges':   "Provide a personalised discount or bundle upgrade to reduce bill.",
    'Tech_Support':      "Proactively reach out with free technical support or onboarding.",
    'Internet_Service':  "Upgrade internet plan or offer a free speed boost trial.",
    'Online_Security':   "Offer a free 3-month online security add-on.",
    'Payment_Method':    "Incentivise switch to auto-pay with a small monthly discount.",
    'Total_Charges':     "Review billing history and offer a goodwill credit if anomalies found.",
    'Dependents':        "Promote family/multi-line plans with bundled discounts.",
    'Partner':           "Offer couples or household bundling deals.",
}

def get_retention_action(feature):
    """Return a retention action for the most relevant feature keyword."""
    for key, action in RETENTION_ACTIONS.items():
        if key.lower() in feature.lower():
            return action
    return "Conduct a personalised retention call to understand dissatisfaction."

# Build high-risk customer table (churn probability > 0.7)
# FIX: Use X_test_reset (positional index 0..n) so it aligns with shap_values rows.
risk_df = X_test_reset.copy()
risk_df['Churn_Probability'] = y_prob_xgb
risk_df['Predicted_Churn']   = y_pred_xgb
risk_df['Actual_Churn']      = y_test.values

high_risk = risk_df[risk_df['Churn_Probability'] > 0.7].copy()
print(f"\nHigh-risk customers (churn probability > 70%): {len(high_risk)}")

# FIX: high_risk.index is now positional (0-based), matching shap_values rows exactly.
shap_high = shap_values[high_risk.index]   # shape: (n_high_risk, n_features)

# Build insight report
insights = []
for pos, (idx, row) in enumerate(high_risk.iterrows()):
    top_feature_idx  = np.argmax(np.abs(shap_high[pos]))
    top_feature_name = feature_names[top_feature_idx]
    top_shap_value   = shap_high[pos][top_feature_idx]
    action           = get_retention_action(top_feature_name)

    insights.append({
        'Customer_Index'     : idx,
        'Churn_Probability'  : round(row['Churn_Probability'], 3),
        'Top_Churn_Driver'   : top_feature_name,
        'SHAP_Impact'        : round(float(top_shap_value), 4),
        'Recommended_Action' : action
    })

insights_df = pd.DataFrame(insights)

print("\n========== Customer Retention Insights (Top 20 High-Risk) ==========")
pd.set_option('display.max_colwidth', 80)
print(insights_df.head(20).to_string(index=False))

# --- Summary: Most common churn drivers across all high-risk customers ---
print("\n========== Most Common Churn Drivers Across High-Risk Segment ==========")
driver_counts = insights_df['Top_Churn_Driver'].value_counts().head(5)
print(driver_counts)

plt.figure(figsize=(8, 4))
driver_counts.plot(kind='barh', color='salmon')
plt.title("Top Churn Drivers Across High-Risk Customers")
plt.xlabel("Number of Customers Affected")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- Print top 5 prioritised retention actions ---
print("\n========== Priority Retention Actions for Business Team ==========")
top_drivers = driver_counts.index.tolist()
for rank, driver in enumerate(top_drivers, 1):
    action = get_retention_action(driver)
    print(f"{rank}. [{driver}] → {action}")
# AUC score
auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", auc)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1])  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
