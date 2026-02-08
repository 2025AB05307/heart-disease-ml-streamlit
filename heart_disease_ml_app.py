import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Heart Disease ML App", layout="centered")
st.title("Heart Disease Classification App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model_name = st.selectbox(
    "Select Machine Learning Model",
    [
        "Logistic_Regression",
        "Decision_Tree",
        "KNN",
        "Naive_Bayes",
        "Random_Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "HeartDisease" not in df.columns:
        st.error("CSV must contain 'HeartDisease' column")
    else:
        # SAME FEATURE ENGINEERING AS TRAINING
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 40, 55, 100],
            labels=[0, 1, 2]
        )

        y = df["HeartDisease"]
        X = df.drop("HeartDisease", axis=1)

        X = pd.get_dummies(X, drop_first=True)

        feature_columns = joblib.load("model/feature_columns.pkl")
        X = X.reindex(columns=feature_columns, fill_value=0)

        scaler = joblib.load("model/scaler.pkl")
        X = scaler.transform(X)

        model = joblib.load(f"model/{model_name}.pkl")
        y_pred = model.predict(X)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(
            confusion_matrix(y, y_pred),
            annot=True, fmt="d", cmap="Blues", ax=ax
        )
        st.pyplot(fig)
