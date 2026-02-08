import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("data/heart.csv")

# =========================
# 2. FEATURE ENGINEERING (NEW FEATURE)
# =========================
# Derived feature to meet minimum feature requirement
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 40, 55, 100],
    labels=[0, 1, 2]
)

# =========================
# 3. Split Features & Target
# =========================
y = df["HeartDisease"]
X = df.drop("HeartDisease", axis=1)

# =========================
# 4. One-Hot Encoding
# =========================
X = pd.get_dummies(X, drop_first=True)

os.makedirs("model", exist_ok=True)

# Save feature structure
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

# =========================
# 5. Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# =========================
# 6. Scaling (SAVE SCALER)
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# =========================
# 7. Models
# =========================
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

print("\nMODEL PERFORMANCE RESULTS\n")

# =========================
# 8. Train, Evaluate, Save
# =========================
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("AUC      :", roc_auc_score(y_test, y_prob))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))
    print("MCC      :", matthews_corrcoef(y_test, y_pred))

    joblib.dump(model, f"model/{name}.pkl")
