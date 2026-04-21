#!/usr/bin/env python
# coding: utf-8

# # 8. Random Forest Ensemble

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# --- Load and preprocess churn dataset (Classification) ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_churn = pd.read_csv('churn_df.csv')
# Dropping ID columns which have no predictive value
df_churn = df_churn.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')

X_clf = df_churn.drop(columns=['Exited'])
y_clf = df_churn['Exited']

# Handling missing values
X_clf.fillna(X_clf.mean(numeric_only=True), inplace=True)
for col in X_clf.select_dtypes(include=['object']).columns:
    X_clf[col].fillna(X_clf[col].mode()[0], inplace=True)

# Categorical mapping using get_dummies
X_clf = pd.get_dummies(X_clf, drop_first=True)

# Train Test Split
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Scaling
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)
print("Churn dataset ready for classification!")


# ## Random Forest Classification (Churn Dataset)

# In[3]:


from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_clf_scaled, y_train_clf)
rf_preds = rf_clf.predict(X_test_clf_scaled)

print("RF Classification Accuracy:", accuracy_score(y_test_clf, rf_preds))
print(classification_report(y_test_clf, rf_preds))


# In[4]:


# --- Load and preprocess housing dataset (Regression) ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_housing = pd.read_csv('housing.csv')
X_reg = df_housing.drop(columns=['median_house_value'])
y_reg = df_housing['median_house_value']

# Handling missing values
X_reg.fillna(X_reg.mean(numeric_only=True), inplace=True)
for col in X_reg.select_dtypes(include=['object']).columns:
    X_reg[col].fillna(X_reg[col].mode()[0], inplace=True)

# Categorical mapping using get_dummies
X_reg = pd.get_dummies(X_reg, drop_first=True)

# Train Test Split
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Scaling
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)
print("Housing dataset ready for regression!")


# ## Random Forest Regression (Housing Dataset)

# In[5]:


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg_scaled, y_train_reg)
rf_reg_preds = rf_reg.predict(X_test_reg_scaled)

print("RF Regression MSE:", mean_squared_error(y_test_reg, rf_reg_preds))
print("RF Regression R2:", r2_score(y_test_reg, rf_reg_preds))


# In[ ]:




