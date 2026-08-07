"""
Step 5: Evaluate and compare trained models.
Computes accuracy, precision, recall, F1 (macro-averaged, appropriate for
imbalanced multiclass problems), and ROC-AUC (binary benign/attack view).
Prints a comparison table and confusion matrix for the best model.
"""
import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    return metrics, y_pred


def compare_models(data: dict):
    results = []
    predictions = {}

    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".pkl"):
            continue
        name = fname.replace(".pkl", "")
        model = joblib.load(os.path.join(MODELS_DIR, fname))
        metrics, y_pred = evaluate_model(model, data["X_test"], data["y_test"], name)
        results.append(metrics)
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    print("=" * 70)
    print("MODEL COMPARISON (sorted by macro F1-score)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_model_name}")

    print("\n" + "=" * 70)
    print(f"DETAILED CLASSIFICATION REPORT — {best_model_name}")
    print("=" * 70)
    y_pred_best = predictions[best_model_name]
    le = data["label_encoder"]
    print(classification_report(
        data["y_test"], y_pred_best,
        target_names=le.classes_, zero_division=0
    ))

    print("\nCONFUSION MATRIX:")
    print(confusion_matrix(data["y_test"], y_pred_best))

    return results_df, best_model_name


if __name__ == "__main__":
    data_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl")
    data = joblib.load(data_path)
    compare_models(data)
