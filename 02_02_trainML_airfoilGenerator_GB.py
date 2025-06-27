import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
import os

# Load dataset
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
csv_path = os.path.join(script_dir, "cst_training_dataset.csv")
df = pd.read_csv(csv_path)
X = df.drop(columns=["valid", "airfoil"], errors='ignore')
y = df["valid"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Gradient Boosting Classifier
gb_clf = HistGradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = gb_clf.predict(X_test)
y_proba = gb_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Gradient Boosting Classifier performance:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(gb_clf, X_test, y_test, cmap=plt.cm.Greens)
plt.title("Confusion Matrix (Gradient Boosting)")
plt.show()

# Permutation Importance
print("Calculating permutation feature importances...")
result = permutation_importance(gb_clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importances = result.importances_mean
feature_names = X.columns
sorted_idx = importances.argsort()[::-1]

plt.figure(figsize=(10,6))
plt.bar(range(len(importances)), importances[sorted_idx], align="center")
plt.xticks(range(len(importances)), feature_names[sorted_idx], rotation=90)
plt.title("Permutation Feature Importances (Gradient Boosting)")
plt.tight_layout()
plt.show()

# Save model
joblib.dump(gb_clf, "gb_cst_classifier.joblib")
print("Gradient Boosting model saved as 'gb_cst_classifier.joblib'")
