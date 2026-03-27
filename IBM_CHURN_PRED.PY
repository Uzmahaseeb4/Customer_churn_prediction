import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
df =pd.read_excel("C:/Users/asus/Downloads/archive (7)/Telco_customer_churn.xlsx")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df[df.duplicated()])
df = df.drop(columns=[
    'CustomerID', 'Lat Long', 'Zip Code', 'City', 'State', 'Country',
    'Churn Reason'
], errors='ignore')
df.columns = df.columns.str.replace(' ', '_')
df['Total_Charges'] = pd.to_numeric(df['Total_Charges'], errors='coerce')
df['Total_Charges'] = df['Total_Charges'].fillna(df['Total_Charges'].median())
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])


X = df.drop(['Churn_Label', 'Churn_Value', 'Churn_Score'], axis=1)
y = df['Churn_Value']


from sklearn.model_selection import train_test_split

X = df.drop(['Churn_Label', 'Churn_Value', 'Churn_Score'], axis=1)
y = df['Churn_Value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_sm, y_train_sm)
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# predictions
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# metrics
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree ROC-AUC:", roc_auc_score(y_test, y_prob_dt))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_sm, y_train_sm)
import shap

explainer = shap.TreeExplainer(rf)

X_sample = X_test.sample(200, random_state=42)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values[:, :, 1], X_sample)
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

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