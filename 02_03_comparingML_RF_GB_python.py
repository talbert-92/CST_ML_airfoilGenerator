# Side-by-side Evaluation of Random Forest and Gradient Boosting for CST Airfoil Classification

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("cst_training_dataset.csv")

# Features and target
X = df.drop(columns=["valid", "airfoil"], errors='ignore')
y = df["valid"]

print(f"Dataset shape: {X.shape}, Positive samples: {y.sum()}, Negative samples: {len(y) - y.sum()}")

# Split for hold-out test (optional)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define stratified k-fold cross-validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Setup complete. Ready to start cross-validation on both models.")

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# Define models with default parameters for baseline evaluation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = HistGradientBoostingClassifier(random_state=42)

# Initialize lists to hold metrics for each fold
rf_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
gb_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}

# Perform stratified k-fold cross-validation on training data
for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    print(f"Fold {fold}...")

    # Split data into fold-specific training and validation sets
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # ---- Random Forest Training and Evaluation ----
    rf_model.fit(X_tr, y_tr)
    y_pred_rf = rf_model.predict(X_val)
    y_proba_rf = rf_model.predict_proba(X_val)[:, 1]

    # Compute metrics
    rf_metrics["accuracy"].append(accuracy_score(y_val, y_pred_rf))
    rf_metrics["precision"].append(precision_score(y_val, y_pred_rf))
    rf_metrics["recall"].append(recall_score(y_val, y_pred_rf))
    rf_metrics["f1"].append(f1_score(y_val, y_pred_rf))
    rf_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba_rf))

    # ---- Gradient Boosting Training and Evaluation ----
    gb_model.fit(X_tr, y_tr)
    y_pred_gb = gb_model.predict(X_val)
    y_proba_gb = gb_model.predict_proba(X_val)[:, 1]

    # Compute metrics
    gb_metrics["accuracy"].append(accuracy_score(y_val, y_pred_gb))
    gb_metrics["precision"].append(precision_score(y_val, y_pred_gb))
    gb_metrics["recall"].append(recall_score(y_val, y_pred_gb))
    gb_metrics["f1"].append(f1_score(y_val, y_pred_gb))
    gb_metrics["roc_auc"].append(roc_auc_score(y_val, y_proba_gb))

print("\n--- Cross-Validation Results ---\n")

def print_avg_metrics(name, metrics_dict):
    print(f"{name} Performance:")
    print(f" Accuracy : {np.mean(metrics_dict['accuracy']):.4f} ± {np.std(metrics_dict['accuracy']):.4f}")
    print(f" Precision: {np.mean(metrics_dict['precision']):.4f} ± {np.std(metrics_dict['precision']):.4f}")
    print(f" Recall   : {np.mean(metrics_dict['recall']):.4f} ± {np.std(metrics_dict['recall']):.4f}")
    print(f" F1 Score : {np.mean(metrics_dict['f1']):.4f} ± {np.std(metrics_dict['f1']):.4f}")
    print(f" ROC AUC  : {np.mean(metrics_dict['roc_auc']):.4f} ± {np.std(metrics_dict['roc_auc']):.4f}")
    print()

print_avg_metrics("Random Forest", rf_metrics)
print_avg_metrics("Gradient Boosting", gb_metrics)

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score
import joblib

# Load dataset
df = pd.read_csv("cst_training_dataset.csv")
X = df.drop(columns=["valid", "airfoil"], errors='ignore')
y = df["valid"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Define base GB model
gb_clf = HistGradientBoostingClassifier(random_state=42)

# Define hyperparameter search space
param_dist = {
    "max_iter": [100, 200, 300, 400, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_leaf_nodes": [15, 31, 50, 100, None],
    "l2_regularization": [0.0, 0.1, 0.5, 1.0],
}

# Use F1-score as tuning metric (balance of precision & recall)
scorer = make_scorer(f1_score)

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=gb_clf,
    param_distributions=param_dist,
    n_iter=50,  # number of parameter settings sampled
    scoring=scorer,
    n_jobs=-1,
    cv=5,
    verbose=2,
    random_state=42,
)

# Run search
random_search.fit(X_train, y_train)

print("Best hyperparameters found:")
print(random_search.best_params_)

# Save the best model
joblib.dump(random_search.best_estimator_, "gb_cst_classifier_tuned.joblib")
print("Tuned GB model saved as 'gb_cst_classifier_tuned.joblib'")

# Evaluate on test set
best_gb = random_search.best_estimator_
y_pred = best_gb.predict(X_test)
y_proba = best_gb.predict_proba(X_test)[:, 1]

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
print("Performance on test set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("cst_training_dataset.csv")
X = df.drop(columns=["valid", "airfoil"], errors='ignore')
y = df["valid"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Base RF model
rf_clf = RandomForestClassifier(random_state=42)

# Hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 10],
    'max_features': [None, 'sqrt', 'log2'],  # replaced 'auto' with None
}

# Use F1 score as evaluation metric
scorer = make_scorer(f1_score)

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    estimator=rf_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring=scorer,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run hyperparameter tuning
random_search.fit(X_train, y_train)

print("Best hyperparameters found:")
print(random_search.best_params_)

# Save best RF model
joblib.dump(random_search.best_estimator_, "rf_cst_classifier_tuned.joblib")
print("Tuned RF model saved as 'rf_cst_classifier_tuned.joblib'")

# Evaluate on test set
best_rf = random_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("Performance on test set:")
print(classification_report(y_test, y_pred))


