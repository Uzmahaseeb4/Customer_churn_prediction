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
print(df.info)
df.isnull().sum()