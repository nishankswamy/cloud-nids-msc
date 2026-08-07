"""
Step 4: Train and compare multiple models.
Trains Random Forest, XGBoost, and a simple Neural Network (MLP) on the
preprocessed, SMOTE-balanced training data. Each model is saved to models/.
"""
import os
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def get_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            eval_metric="mlogloss",
            n_jobs=-1, random_state=42
        ),
        "mlp_neural_net": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=100,
            early_stopping=True, random_state=42
        ),
    }


def train_all(data: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(data["X_train"], data["y_train"])
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

        out_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, out_path)
        print(f"  Saved to {out_path}")
        trained[name] = model

    return trained


if __name__ == "__main__":
    data_path = os.path.join(PROCESSED_DATA_DIR, "preprocessed.pkl")
    data = joblib.load(data_path)
    trained_models = train_all(data)
    print(f"\nTrained {len(trained_models)} models: {list(trained_models.keys())}")
