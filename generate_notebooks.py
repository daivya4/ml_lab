import nbformat as nbf
import os

def write_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

def md(text):
    return nbf.v4.new_markdown_cell(text)

def code(text):
    return nbf.v4.new_code_cell(text)

imports = """\
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
"""

prep_housing = """\
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
"""

prep_churn = """\
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
"""

# ----------------- 01 -----------------
nb1 = [
    md("# 1. Pandas Import & Export Data"),
    code(imports),
    md("## Loading Housing Dataset"),
    code("df_housing = pd.read_csv('housing.csv')\ndf_housing.head()"),
    code("print(df_housing.info())\nprint(df_housing.describe())"),
    md("## Loading Churn Dataset"),
    code("df_churn = pd.read_csv('churn_df.csv')\ndf_churn.head()"),
    md("## Exporting a sample"),
    code("df_churn_sample = df_churn.sample(10)\ndf_churn_sample.to_csv('sample_churn.csv', index=False)")
]
write_notebook("01_Pandas_Import_Export.ipynb", nb1)

# ----------------- 02 -----------------
nb2 = [
    md("# 2. Data Preprocessing"),
    code(imports),
    md("## Preprocessing Housing Data"),
    code(prep_housing),
    md("## Preprocessing Churn Data"),
    code(prep_churn),
]
write_notebook("02_Data_Preprocessing.ipynb", nb2)

# ----------------- 03 -----------------
nb3 = [
    md("# 3. Decision Tree"),
    code(imports),
    code(prep_churn),
    code(prep_housing),
    md("## ID3 Classification (using entropy) - Churn Data\nOptimization: Added `class_weight='balanced'` to massively improve Class 1 recall."),
    code("""\
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

id3_clf = DecisionTreeClassifier(criterion='entropy', max_depth=6, class_weight='balanced', random_state=42)
id3_clf.fit(X_train_clf_scaled, y_train_clf)

print("ID3 Classification Accuracy:", accuracy_score(y_test_clf, id3_clf.predict(X_test_clf_scaled)))
print(classification_report(y_test_clf, id3_clf.predict(X_test_clf_scaled)))
"""),
    md("## CART Classification (using gini) - Churn Data\nOptimization: Added `class_weight='balanced'`"),
    code("""\
cart_clf = DecisionTreeClassifier(criterion='gini', max_depth=6, class_weight='balanced', random_state=42)
cart_clf.fit(X_train_clf_scaled, y_train_clf)

print("CART Classification Accuracy:", accuracy_score(y_test_clf, cart_clf.predict(X_test_clf_scaled)))
print(classification_report(y_test_clf, cart_clf.predict(X_test_clf_scaled)))
"""),
    md("## CART Regression (using squared_error) - Housing Data\nOptimization: Deepened `max_depth` to 10 and calculated RMSE instead of MSE."),
    code("""\
cart_reg = DecisionTreeRegressor(criterion='squared_error', max_depth=10, random_state=42)
cart_reg.fit(X_train_reg_scaled, y_train_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg, cart_reg.predict(X_test_reg_scaled)))
print(f"CART Regression RMSE: ${rmse:,.2f}")
print("CART Regression R2:", r2_score(y_test_reg, cart_reg.predict(X_test_reg_scaled)))
""")
]
write_notebook("03_Decision_Tree_ID3.ipynb", nb3)

# ----------------- 04 -----------------
nb4 = [
    md("# 4. Linear & Multi-Linear Regression"),
    code(imports),
    code(prep_housing),
    md("## Scikit-learn Linear Regression\nOptimization: Outputting RMSE which shows the error magnitude directly in dollars."),
    code("""\
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_reg_scaled, y_train_reg)
preds_reg = lr.predict(X_test_reg_scaled)

rmse = np.sqrt(mean_squared_error(y_test_reg, preds_reg))
print(f"RMSE: ${rmse:,.2f}")
print("R2 Score:", r2_score(y_test_reg, preds_reg))
"""),
    md("## Multi-Linear Regression from scratch (Gradient Descent)"),
    code("""\
class MyMultiLinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

my_lr = MyMultiLinearRegression(lr=0.01, epochs=1000)
my_lr.fit(X_train_reg_scaled, y_train_reg.values)
my_preds = my_lr.predict(X_test_reg_scaled)

my_rmse = np.sqrt(mean_squared_error(y_test_reg, my_preds))
print(f"Scratch RMSE: ${my_rmse:,.2f}")
print("Scratch R2:", r2_score(y_test_reg, my_preds))
""")
]
write_notebook("04_Linear_MultiLinear_Regression.ipynb", nb4)

# ----------------- 05 -----------------
nb5 = [
    md("# 5. Logistic Regression"),
    code(imports),
    code(prep_churn),
    md("## Scikit-Learn Logistic Regression\nOptimization: Added `class_weight='balanced'` for vastly improved minority class recall."),
    code("""\
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(class_weight='balanced', random_state=42)
log_clf.fit(X_train_clf_scaled, y_train_clf)
log_preds = log_clf.predict(X_test_clf_scaled)

print("Accuracy:", accuracy_score(y_test_clf, log_preds))
print(classification_report(y_test_clf, log_preds))
"""),
    md("## Logistic Regression from scratch (Gradient Descent) - Predict Adjust\nManually adjusted prediction threshold to 0.3 to increase Class 1 recall naturally."),
    code("""\
class MyLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

my_log_clf = MyLogisticRegression(lr=0.1, epochs=500)
my_log_clf.fit(X_train_clf_scaled, y_train_clf.values)

# Lowering threshold from standard 0.5 to 0.3 to improve recall significantly
my_log_probs = my_log_clf.predict_proba(X_test_clf_scaled)
my_log_preds = (my_log_probs >= 0.3).astype(int)

print("Scratch Accuracy:", accuracy_score(y_test_clf, my_log_preds))
print(classification_report(y_test_clf, my_log_preds))
""")
]
write_notebook("05_Logistic_Regression.ipynb", nb5)

# ----------------- 06 -----------------
nb6 = [
    md("# 6. KNN - K-Nearest Neighbors"),
    code(imports),
    code(prep_churn),
    code(prep_housing),
    md("## KNN Classification (Churn Data)\nOptimization: Used `predict_proba()` to lower probability threshold to 0.3, vastly increasing Class 1 recall without libraries."),
    code("""\
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_clf_scaled, y_train_clf)

# Custom threshold for recall improvement
knn_probs = knn_clf.predict_proba(X_test_clf_scaled)[:, 1]
knn_preds = (knn_probs >= 0.3).astype(int)

print("KNN Classification Accuracy:", accuracy_score(y_test_clf, knn_preds))
print(classification_report(y_test_clf, knn_preds))
"""),
    md("## KNN Regression (Housing Data)\nOptimization: Used `weights='distance'` and RMSE."),
    code("""\
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=15, weights='distance')
knn_reg.fit(X_train_reg_scaled, y_train_reg)
knn_reg_preds = knn_reg.predict(X_test_reg_scaled)

rmse = np.sqrt(mean_squared_error(y_test_reg, knn_reg_preds))
print(f"KNN Regression RMSE: ${rmse:,.2f}")
print("KNN Regression R2:", r2_score(y_test_reg, knn_reg_preds))
""")
]
write_notebook("06_KNN.ipynb", nb6)

# ----------------- 07 -----------------
nb7 = [
    md("# 7. SVM - Support Vector Machine"),
    code(imports),
    code(prep_churn),
    code(prep_housing),
    md("## SVM Classification (Churn Data)\nOptimization: Added `class_weight='balanced'` for Class 1 recall."),
    code("""\
from sklearn.svm import SVC

svm_clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_clf.fit(X_train_clf_scaled, y_train_clf)
svm_preds = svm_clf.predict(X_test_clf_scaled)

print("SVM Classification Accuracy:", accuracy_score(y_test_clf, svm_preds))
print(classification_report(y_test_clf, svm_preds))
"""),
    md("## SVM Regression (Housing Data)\nOptimization: Upgraded from linear kernel to highly potent `rbf` kernel (C=100) to cut MSE strictly and drastically on non-linear datasets!"),
    code("""\
from sklearn.svm import SVR

# SVM 'rbf' with normalized features is massively superior to linear format for geospatial data.
svm_reg = SVR(kernel='rbf', C=100, gamma='scale')
svm_reg.fit(X_train_reg_scaled, y_train_reg)
svm_reg_preds = svm_reg.predict(X_test_reg_scaled)

rmse = np.sqrt(mean_squared_error(y_test_reg, svm_reg_preds))
print(f"SVM Regression RMSE: ${rmse:,.2f}")
print("SVM Regression R2:", r2_score(y_test_reg, svm_reg_preds))
""")
]
write_notebook("07_SVM.ipynb", nb7)

# ----------------- 08 -----------------
nb8 = [
    md("# 8. Random Forest Ensemble"),
    code(imports),
    code(prep_churn),
    md("## Random Forest Classification (Churn Dataset)\nOptimization: Set `class_weight='balanced'` to repair Class 1 recall natively."),
    code("""\
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_clf.fit(X_train_clf_scaled, y_train_clf)
rf_preds = rf_clf.predict(X_test_clf_scaled)

print("RF Classification Accuracy:", accuracy_score(y_test_clf, rf_preds))
print(classification_report(y_test_clf, rf_preds))
"""),
    code(prep_housing),
    md("## Random Forest Regression (Housing Dataset)\nOptimization: Calculating standard RMSE and ensuring estimators map out deeply."),
    code("""\
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=150, max_depth=None, random_state=42)
rf_reg.fit(X_train_reg_scaled, y_train_reg)
rf_reg_preds = rf_reg.predict(X_test_reg_scaled)

rmse = np.sqrt(mean_squared_error(y_test_reg, rf_reg_preds))
print(f"RF Regression RMSE: ${rmse:,.2f}")
print("RF Regression R2:", r2_score(y_test_reg, rf_reg_preds))
""")
]
write_notebook("08_Random_Forest.ipynb", nb8)

# ----------------- 09 -----------------
nb9 = [
    md("# 9. Boosting Ensemble"),
    code(imports),
    code(prep_churn),
    md("## AdaBoost and Gradient Boosting Classification (Churn Dataset)"),
    code("""\
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_clf.fit(X_train_clf_scaled, y_train_clf)

gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gb_clf.fit(X_train_clf_scaled, y_train_clf)

ada_probs = ada_clf.predict_proba(X_test_clf_scaled)[:, 1]
gb_probs = gb_clf.predict_proba(X_test_clf_scaled)[:, 1]

# Adjusting threshold to 0.3 for Boosting to ensure excellent Class 1 recall natively
ada_preds = (ada_probs >= 0.3).astype(int)
gb_preds = (gb_probs >= 0.3).astype(int)

print("AdaBoost Class 1 Optimized Classification Matrix:")
print(classification_report(y_test_clf, ada_preds))

print("GradientBoosting Class 1 Optimized Classification Matrix:")
print(classification_report(y_test_clf, gb_preds))
"""),
    code(prep_housing),
    md("## AdaBoost and Gradient Boosting Regression (Housing Dataset)\nOptimization: Added RMSE representation. Boosted estimators and depth for better accuracy."),
    code("""\
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

ada_reg = AdaBoostRegressor(n_estimators=100, random_state=42)
ada_reg.fit(X_train_reg_scaled, y_train_reg)

gb_reg = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
gb_reg.fit(X_train_reg_scaled, y_train_reg)

ada_rmse = np.sqrt(mean_squared_error(y_test_reg, ada_reg.predict(X_test_reg_scaled)))
print(f"AdaBoost RMSE: ${ada_rmse:,.2f}")

gb_rmse = np.sqrt(mean_squared_error(y_test_reg, gb_reg.predict(X_test_reg_scaled)))
print(f"GradientBoosting RMSE: ${gb_rmse:,.2f}")
""")
]
write_notebook("09_Boosting.ipynb", nb9)

# ----------------- 10 -----------------
nb10 = [
    md("# 10. K-Means Clustering"),
    code(imports),
    code(prep_housing),
    md("## Clustering Housing Dataset using K-Means"),
    code("""\
from sklearn.cluster import KMeans

# Determine Optimal K using elbow method on scaled features
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_train_reg_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on elbow curve
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
clusters = final_kmeans.fit_predict(X_train_reg_scaled)
print(f"Assigned clusters to data with K={optimal_k}")
"""),
    md("## Visualizing Clusters (2D Projection with PCA)"),
    code("""\
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_X = pca.fit_transform(X_train_reg_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=reduced_X[:, 0], y=reduced_X[:, 1], hue=clusters, palette='viridis')
plt.title('K-Means Clusters shown via 2 Principal Components')
plt.show()
""")
]
write_notebook("10_KMeans_Clustering.ipynb", nb10)

# ----------------- 11 -----------------
nb11 = [
    md("# 11. Principal Component Analysis (PCA)"),
    code(imports),
    code(prep_churn),
    md("## PCA implementation on Churn dataset"),
    code("""\
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_train_clf_scaled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,5))
plt.plot(np.cumsum(explained_variance), marker='o')
plt.title('Cumulative Explained Variance over Principal Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.grid(True)
plt.show()
"""),
    code("""\
# Apply PCA preserving 90% variance
pca_90 = PCA(n_components=0.90)
X_reduced = pca_90.fit_transform(X_train_clf_scaled)

print("Original dimensions:", X_train_clf_scaled.shape[1])
print("Reduced dimensions to keep 90% variance:", X_reduced.shape[1])
""")
]
write_notebook("11_PCA.ipynb", nb11)

print("All simple optimized static notebooks created successfully!")
