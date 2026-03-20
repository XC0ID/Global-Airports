"""
classification.py
-----------------
Random Forest + Gradient Boosting classifier to predict
airport hub_type (International / Regional / Domestic).
Includes training, evaluation, feature importance, and model persistence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 1. TRAIN RANDOM FOREST
# ──────────────────────────────────────────────

def train_random_forest(X_train: np.ndarray,
                        y_train: np.ndarray,
                        n_estimators: int = 200,
                        max_depth: int = None,
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Returns
    -------
    model : fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"[✔] Random Forest trained  |  Trees: {n_estimators}")
    return model


# ──────────────────────────────────────────────
# 2. TRAIN GRADIENT BOOSTING
# ──────────────────────────────────────────────

def train_gradient_boosting(X_train: np.ndarray,
                             y_train: np.ndarray,
                             n_estimators: int = 150,
                             learning_rate: float = 0.1,
                             random_state: int = 42) -> GradientBoostingClassifier:
    """Train a Gradient Boosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print(f"[✔] Gradient Boosting trained  |  Estimators: {n_estimators}  |  LR: {learning_rate}")
    return model


# ──────────────────────────────────────────────
# 3. COMPARE MULTIPLE MODELS
# ──────────────────────────────────────────────

def compare_models(X_train: np.ndarray,
                   y_train: np.ndarray,
                   cv: int = 5) -> dict:
    """
    Train and cross-validate multiple classifiers.
    Returns a dict of {model_name: cv_score}.
    """
    models = {
        "Logistic Regression"  : LogisticRegression(max_iter=500, random_state=42),
        "Random Forest"        : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)"            : SVC(kernel="rbf", random_state=42),
    }

    results = {}
    print(f"\n══════════ MODEL COMPARISON (CV={cv}) ══════════")
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        mean_s, std_s = scores.mean(), scores.std()
        results[name] = {"mean": mean_s, "std": std_s}
        print(f"  {name:<25}  Acc = {mean_s:.4f} ± {std_s:.4f}")

    best = max(results, key=lambda k: results[k]["mean"])
    print(f"\n[✔] Best model: {best}  ({results[best]['mean']:.4f})")
    return results


# ──────────────────────────────────────────────
# 4. EVALUATE MODEL
# ──────────────────────────────────────────────

def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   class_names: list = None,
                   save_path: str = None) -> dict:
    """
    Evaluate a trained classifier on the test set.
    Prints accuracy, classification report, and plots confusion matrix.

    Returns
    -------
    metrics : dict with accuracy and report
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"\n══════════ EVALUATION RESULTS ══════════")
    print(f"  Accuracy : {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return {"accuracy": acc, "report": report}


# ──────────────────────────────────────────────
# 5. FEATURE IMPORTANCE
# ──────────────────────────────────────────────

def plot_feature_importance(model,
                             feature_names: list,
                             top_n: int = 10,
                             save_path: str = None) -> None:
    """
    Plot feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        print("[!] Model does not have feature_importances_. Skipping.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    names       = [feature_names[i] for i in indices]
    values      = importances[indices]

    plt.figure(figsize=(9, 5))
    bars = plt.barh(names[::-1], values[::-1], color="steelblue", alpha=0.85)
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[✔] Feature importance plot done.")


# ──────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────

def tune_random_forest(X_train: np.ndarray,
                       y_train: np.ndarray,
                       cv: int = 3) -> RandomForestClassifier:
    """Grid search to tune Random Forest hyperparameters."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth"   : [None, 5, 10],
        "max_features": ["sqrt", "log2"],
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    gs = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print(f"\n[✔] Best params : {gs.best_params_}")
    print(f"[✔] Best CV acc  : {gs.best_score_:.4f}")
    return gs.best_estimator_


# ──────────────────────────────────────────────
# 7. SAVE / LOAD MODEL
# ──────────────────────────────────────────────

def save_model(model, filepath: str = "models/airport_classifier.pkl") -> None:
    """Persist a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[✔] Model saved → {filepath}")


def load_model(filepath: str = "models/airport_classifier.pkl"):
    """Load a previously saved model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at: {filepath}")
    model = joblib.load(filepath)
    print(f"[✔] Model loaded ← {filepath}")
    return model


# ──────────────────────────────────────────────
# 8. PREDICT NEW AIRPORTS
# ──────────────────────────────────────────────

def predict_airport(model,
                    scaler,
                    label_encoder,
                    input_data: dict) -> str:
    """
    Predict the hub_type for a single new airport.

    Parameters
    ----------
    input_data : dict with keys matching CLASSIFICATION_FEATURES

    Returns
    -------
    prediction : str label (e.g. 'International')
    """
    from src.preprocessing import CLASSIFICATION_FEATURES
    row = pd.DataFrame([input_data])[CLASSIFICATION_FEATURES].fillna(0)
    row_scaled = scaler.transform(row)
    pred_enc   = model.predict(row_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    print(f"[✔] Prediction: {pred_label}")
    return pred_label


# ──────────────────────────────────────────────
# 9. FULL CLASSIFICATION PIPELINE
# ──────────────────────────────────────────────

def run_classification_pipeline(df: pd.DataFrame) -> dict:
    """
    End-to-end classification pipeline:
      load → split → compare → tune → evaluate → save
    """
    from src.preprocessing import (
        prepare_classification_data,
        CLASSIFICATION_FEATURES
    )

    print("\n══════════ CLASSIFICATION PIPELINE ══════════")

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, le = prepare_classification_data(df)

    # Compare models
    compare_models(X_train, y_train)

    # Train best model (Random Forest)
    model = train_random_forest(X_train, y_train)

    # Evaluate
    class_names = list(le.classes_)
    metrics = evaluate_model(model, X_test, y_test, class_names=class_names)

    # Feature importances
    plot_feature_importance(model, feature_names=CLASSIFICATION_FEATURES)

    # Save
    save_model(model)

    print("\n[✔] Classification pipeline complete.")
    return {"model": model, "scaler": scaler, "label_encoder": le, "metrics": metrics}


if __name__ == "__main__":
    from src.preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline()
    results = run_classification_pipeline(df)
