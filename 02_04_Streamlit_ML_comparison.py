import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, classification_report, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("📘 ML Classifier Comparison for CST Airfoils")

# --- Load Dataset ---
st.sidebar.header("Step 1: Load Dataset")
use_default = st.sidebar.checkbox("Use default: cst_training_dataset.csv", value=True)

if use_default:
    csv_path = "cst_training_dataset.csv"
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        csv_path = uploaded_file
    else:
        st.stop()

df = pd.read_csv(csv_path)

if "valid" not in df.columns:
    st.error("Dataset must contain a 'valid' column as label.")
    st.stop()

X = df.drop(columns=["valid", "airfoil"], errors='ignore')
y = df["valid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Load Models ---
st.sidebar.header("Step 2: Select Models")
use_rf = st.sidebar.checkbox("Include Random Forest", value=True)
use_gb = st.sidebar.checkbox("Include Gradient Boosting", value=True)

# --- Hyperparameter Tuning ---
st.sidebar.header("Step 3: Hyperparameter Tuning")
# Random Forest
rf_n = st.sidebar.slider("RF: n_estimators", 1, 300, 1, step=1)
rf_max_depth = st.sidebar.slider("RF: max_depth", 1, 50, 1, step=1)
rf_min_samples_split = st.sidebar.slider("RF: min_samples_split", 2, 20, 2, step=1)

# Gradient Boosting
gb_lr = st.sidebar.slider("GB: learning_rate", 0.001, 1.0, 0.001, step=0.001)
gb_max_iter = st.sidebar.slider("GB: max_iter", 10, 500, 10, step=10)

models = {}

if use_rf:
    rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

if use_gb:
    gb = HistGradientBoostingClassifier(learning_rate=gb_lr, max_iter=gb_max_iter, random_state=42)
    gb.fit(X_train, y_train)
    models["Gradient Boosting"] = gb

# --- Evaluation ---
st.header("🔍 Model Performance")
metric_table = []

cols = st.columns(len(models))

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    metric_table.append([name, acc, prec, rec, f1, roc])

    with cols[i]:
        st.subheader(name)
        st.write("**Classification Report**")
        st.text(classification_report(y_test, y_pred))

        st.write("**Confusion Matrix**")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

        st.write("**ROC Curve**")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

st.write("### 📊 Metric Summary Table")
st.dataframe(pd.DataFrame(metric_table, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]))

# --- Feature Importances ---
st.header("📌 Feature Importances")
importance_mode = st.radio("Choose importance type", ["Model-Based", "Permutation"])

for name, model in models.items():
    st.subheader(f"{name} Feature Importance")
    if importance_mode == "Model-Based" and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        importances = result.importances_mean

    sorted_idx = np.argsort(importances)[::-1]
    feature_names = X.columns[sorted_idx]
    sorted_importances = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(importances)), sorted_importances)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_title(f"{importance_mode} Feature Importances")
    fig.tight_layout()
    st.pyplot(fig)

st.success("Done! Adjust parameters on the left to explore model behavior.")
