# Create a comprehensive Colab-ready Python script with active (not commented-out) code snippets
# separated by detailed comments and covering multiple dataset-loading methods and algorithms.

# ============================================================
# COMPLETE GOOGLE COLAB ML WORKFLOW (FULLY EXECUTABLE)
# Covers:
# - Loading datasets (inbuilt + external)
# - Preprocessing
# - Classification, Regression, Clustering, PCA
# - Multiple permutations of workflows
# ============================================================


# ============================================================
# SECTION 1: INSTALL & IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# SECTION 2: LOAD DATASET - METHOD 1 (INBUILT DATASET)
# ============================================================
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df_iris = iris.frame

print("INBUILT DATASET (IRIS)")
print(df_iris.head())


# ============================================================
# SECTION 3: LOAD DATASET - METHOD 2 (DIRECT URL CSV)
# ============================================================
url = "https://raw.githubusercontent.com/rolandmueller/titanic/main/titanic3.csv"
df_url = pd.read_csv(url)

print("\\nEXTERNAL DATASET (URL)")
print(df_url.head())


# ============================================================
# SECTION 4: LOAD DATASET - METHOD 3 (UPLOAD FILE IN COLAB)
# ============================================================
from google.colab import files

print("\\nUPLOAD YOUR FILE")
uploaded = files.upload()

for file_name in uploaded.keys():
    df_upload = pd.read_csv(file_name)
    print("\\nUPLOADED DATASET")
    print(df_upload.head())


# ============================================================
# SECTION 5: BASIC DATA INSPECTION
# ============================================================
print("\\nDATA INFO (URL DATASET)")
print(df_url.info())
print(df_url.isnull().sum())


# ============================================================
# SECTION 6: DATA CLEANING (TITANIC EXAMPLE)
# ============================================================
df_clean = df_url[['pclass','sex','age','fare','survived']].copy()

df_clean['age'] = df_clean['age'].fillna(df_clean['age'].mean())
df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].mean())
df_clean['sex'] = df_clean['sex'].map({'male':0, 'female':1})

print("\\nCLEANED DATA")
print(df_clean.head())


# ============================================================
# SECTION 7: FEATURE & TARGET SPLIT
# ============================================================
X = df_clean.drop("survived", axis=1)
y = df_clean["survived"]


# ============================================================
# SECTION 8: TRAIN TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# SECTION 9: SCALING (FOR KNN, SVM, LOGISTIC)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# SECTION 10: CLASSIFICATION MODELS
# ============================================================

# Logistic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(X_train_scaled, y_train)
y_pred_lr = model_lr.predict(X_test_scaled)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled, y_train)
y_pred_knn = model_knn.predict(X_test_scaled)

# SVM
from sklearn.svm import SVC
model_svm = SVC(kernel='rbf')
model_svm.fit(X_train_scaled, y_train)
y_pred_svm = model_svm.predict(X_test_scaled)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
model_ab = AdaBoostClassifier(n_estimators=50)
model_ab.fit(X_train, y_train)
y_pred_ab = model_ab.predict(X_test)


# ============================================================
# SECTION 11: CLASSIFICATION EVALUATION
# ============================================================
from sklearn.metrics import accuracy_score

print("\\nCLASSIFICATION RESULTS")
print("Logistic:", accuracy_score(y_test, y_pred_lr))
print("Decision Tree:", accuracy_score(y_test, y_pred_dt))
print("KNN:", accuracy_score(y_test, y_pred_knn))
print("SVM:", accuracy_score(y_test, y_pred_svm))
print("Random Forest:", accuracy_score(y_test, y_pred_rf))
print("AdaBoost:", accuracy_score(y_test, y_pred_ab))


# ============================================================
# SECTION 12: REGRESSION DATASET (CALIFORNIA HOUSING)
# ============================================================
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df_reg = housing.frame

Xr = df_reg.drop("MedHouseVal", axis=1)
yr = df_reg["MedHouseVal"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2)


# ============================================================
# SECTION 13: REGRESSION MODELS
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

model_lr_reg = LinearRegression()
model_lr_reg.fit(Xr_train, yr_train)
y_pred_lr_reg = model_lr_reg.predict(Xr_test)

model_dt_reg = DecisionTreeRegressor()
model_dt_reg.fit(Xr_train, yr_train)
y_pred_dt_reg = model_dt_reg.predict(Xr_test)


# ============================================================
# SECTION 14: REGRESSION EVALUATION
# ============================================================
from sklearn.metrics import mean_squared_error
import numpy as np

rmse_lr = np.sqrt(mean_squared_error(yr_test, y_pred_lr_reg))
rmse_dt = np.sqrt(mean_squared_error(yr_test, y_pred_dt_reg))

print("\\nREGRESSION RESULTS")
print("Linear Regression RMSE:", rmse_lr)
print("Decision Tree RMSE:", rmse_dt)


# ============================================================
# SECTION 15: CLUSTERING (K-MEANS)
# ============================================================
from sklearn.cluster import KMeans

X_cluster = df_reg[['MedInc','HouseAge']]

kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(X_cluster)

print("\\nKMEANS CLUSTER LABELS")
print(clusters[:10])


# ============================================================
# SECTION 16: PCA (DIMENSIONALITY REDUCTION)
# ============================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_cluster)

print("\\nPCA REDUCED SHAPE:", X_reduced.shape)


# ============================================================
# END OF FILE
# ============================================================