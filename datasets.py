# ==============================
# IMPORTS
# ==============================
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.datasets import fetch_california_housing, make_blobs

# ==============================
# 1. IRIS (Multi-class classification)
# ==============================
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = iris.target

print("Iris Dataset")
print(X_iris.head())

# ==============================
# 2. WINE (Multi-class classification)
# ==============================
wine = load_wine()
X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
y_wine = wine.target

print("\nWine Dataset")
print(X_wine.head())

# ==============================
# 3. BREAST CANCER (Binary classification)
# ==============================
cancer = load_breast_cancer()
X_cancer = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y_cancer = cancer.target

print("\nBreast Cancer Dataset")
print(X_cancer.head())

# ==============================
# 4. CALIFORNIA HOUSING (Regression)
# ==============================
housing = fetch_california_housing()
X_house = pd.DataFrame(housing.data, columns=housing.feature_names)
y_house = housing.target

print("\nHousing Dataset")
print(X_house.head())

# ==============================
# 5. K-MEANS DATA (Clustering)
# ==============================
X_cluster, _ = make_blobs(n_samples=200, centers=3, random_state=42)

print("\nClustering Dataset")
print(X_cluster[:5])

# ==============================
# READY TO USE:
# Pick any dataset and apply model
# ==============================
