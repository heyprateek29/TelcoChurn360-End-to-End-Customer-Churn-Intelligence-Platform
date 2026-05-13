"""
src/train.py
------------
Phase 4: ML Training Pipeline

Steps:
  1. Load cleaned data
  2. Define features (categorical + numerical)
  3. Build a preprocessing pipeline (scaling + encoding)
  4. Handle class imbalance with class_weight
  5. Train a Random Forest classifier
  6. Evaluate with ROC-AUC, confusion matrix, classification report
  7. Save the trained model pipeline to disk

Why Random Forest?
  - Handles mixed feature types well
  - Robust to outliers
  - Provides feature importances directly
  - Good baseline before trying gradient boosting
  - Explainable with SHAP (Phase 5)

Run:
    python src/train.py
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.preprocessing    import StandardScaler, OneHotEncoder
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLEAN_CSV    = "data/processed/telco_cleaned.csv"
MODEL_PATH   = "models/churn_model.pkl"
METRICS_PATH = "models/model_metrics.json"
PLOTS_DIR    = "models/plots"

RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────

# Numerical features — will be scaled with StandardScaler
NUMERICAL_FEATURES = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "num_services",
    "avg_monthly_spend",
    "contract_risk_score",
    "support_score",
]

# Categorical features — will be one-hot encoded
CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract",
    "paperless_billing",
    "payment_method",
    "tenure_segment",
    "monthly_charges_tier",
]

TARGET = "churn_flag"

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str):
    """Load cleaned CSV and split into features and target."""
    print(f"[1/7] Loading data from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned data not found at '{path}'.\n"
            "Run Phase 2 first: python src/ingest.py"
        )

    df = pd.read_csv(path)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    print(f"      Rows: {len(df):,}  |  Features: {len(ALL_FEATURES)}")
    print(f"      Class balance — Churn: {y.mean():.1%}  "
          f"Retained: {1 - y.mean():.1%}")
    return X, y, df


# ─────────────────────────────────────────────
# STEP 2: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series):
    """
    Stratified split — preserves churn ratio in both sets.
    Stratification is important here because churn is imbalanced (26/74).
    """
    print(f"\n[2/7] Splitting data (train={1-TEST_SIZE:.0%}, test={TEST_SIZE:.0%})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y           # Preserve class ratio
    )

    print(f"      Train: {len(X_train):,} rows  "
          f"(churn: {y_train.mean():.1%})")
    print(f"      Test : {len(X_test):,} rows  "
          f"(churn: {y_test.mean():.1%})")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 3: BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    A sklearn Pipeline chains preprocessing + model into one object.

    Benefits:
      - No data leakage: fit only on train, transform both train and test
      - One .pkl file saves everything (preprocessor + model)
      - Predict on raw features directly — no manual preprocessing at inference

    Preprocessing:
      - Numerical: StandardScaler (mean=0, std=1)
      - Categorical: OneHotEncoder (creates binary columns per category)
    """
    print("\n[3/7] Building preprocessing pipeline ...")

    # Numerical transformer — scale to unit variance
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # Categorical transformer — encode to binary columns
    # handle_unknown='ignore' prevents crash on unseen categories at inference
    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer applies the right transformer to the right columns
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer,  NUMERICAL_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    # Full pipeline: preprocess → classify
    # class_weight='balanced' handles the 26/74 imbalance automatically
    # It upweights the minority class (churners) during training
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   RandomForestClassifier(
            n_estimators=300,       # 300 trees — good balance of speed vs accuracy
            max_depth=15,           # Prevent overfitting
            min_samples_leaf=5,     # Each leaf needs at least 5 samples
            class_weight="balanced",# Handle 26/74 class imbalance
            random_state=RANDOM_STATE,
            n_jobs=-1,              # Use all CPU cores
        ))
    ])

    print("      Pipeline: StandardScaler + OneHotEncoder + RandomForest(300 trees)")
    return pipeline


# ─────────────────────────────────────────────
# STEP 4: CROSS-VALIDATION
# ─────────────────────────────────────────────

def cross_validate(pipeline: Pipeline,
                   X_train: pd.DataFrame,
                   y_train: pd.Series) -> float:
    """
    5-fold stratified cross-validation on the training set.
    Gives a reliable estimate of model performance before final test.
    """
    print("\n[4/7] Running 5-fold cross-validation ...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    print(f"      CV ROC-AUC scores : {[round(s, 4) for s in cv_scores]}")
    print(f"      Mean ROC-AUC      : {cv_scores.mean():.4f}")
    print(f"      Std deviation     : {cv_scores.std():.4f}")

    return cv_scores.mean()


# ─────────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────────

def train(pipeline: Pipeline,
          X_train: pd.DataFrame,
          y_train: pd.Series) -> Pipeline:
    """Fit the full pipeline on training data."""
    print("\n[5/7] Training model on full training set ...")
    pipeline.fit(X_train, y_train)
    print("      Training complete.")
    return pipeline


# ─────────────────────────────────────────────
# STEP 6: EVALUATE
# ─────────────────────────────────────────────

def evaluate(pipeline: Pipeline,
             X_test: pd.DataFrame,
             y_test: pd.Series,
             cv_roc_auc: float) -> dict:
    """
    Evaluate on held-out test set.
    Returns a metrics dict that is saved to JSON for the dashboard.
    """
    print("\n[6/7] Evaluating on test set ...")

    # Predicted class (0/1)
    y_pred = pipeline.predict(X_test)

    # Predicted probability of churn (for ROC-AUC)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Core metrics
    roc_auc   = roc_auc_score(y_test, y_prob)
    cm        = confusion_matrix(y_test, y_pred)
    report    = classification_report(y_test, y_pred,
                                      target_names=["Retained", "Churned"],
                                      output_dict=True)

    # Unpack confusion matrix
    tn, fp, fn, tp = cm.ravel()

    print(f"\n      ROC-AUC        : {roc_auc:.4f}")
    print(f"      CV ROC-AUC     : {cv_roc_auc:.4f}")
    print(f"\n      Confusion Matrix:")
    print(f"        True Negative  (correctly predicted Retained): {tn}")
    print(f"        False Positive (predicted Churn, actually Retained): {fp}")
    print(f"        False Negative (predicted Retain, actually Churned): {fn}")
    print(f"        True Positive  (correctly predicted Churned): {tp}")
    print(f"\n      Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Retained", "Churned"]))

    metrics = {
        "roc_auc":      round(roc_auc, 4),
        "cv_roc_auc":   round(cv_roc_auc, 4),
        "precision_churn": round(report["Churned"]["precision"], 4),
        "recall_churn":    round(report["Churned"]["recall"], 4),
        "f1_churn":        round(report["Churned"]["f1-score"], 4),
        "accuracy":        round(report["accuracy"], 4),
        "true_negatives":  int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives":  int(tp),
        "test_size":       len(y_test),
    }

    return metrics, y_pred, y_prob


# ─────────────────────────────────────────────
# STEP 6b: PLOT EVALUATION CHARTS
# ─────────────────────────────────────────────

def save_plots(pipeline, X_test, y_test, y_pred, y_prob) -> None:
    """Save confusion matrix and ROC curve as PNG files."""

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── Confusion matrix ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Churn Prediction", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"      Saved → {PLOTS_DIR}/confusion_matrix.png")

    # ── ROC Curve ─────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction", fontsize=13)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=150)
    plt.close()
    print(f"      Saved → {PLOTS_DIR}/roc_curve.png")

    # ── Feature importances ────────────────────────────────────
    rf_model     = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get feature names after encoding
    cat_encoder    = preprocessor.named_transformers_["cat"]["encoder"]
    cat_feat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feat_names = NUMERICAL_FEATURES + cat_feat_names

    importances = pd.Series(
        rf_model.feature_importances_,
        index=all_feat_names
    ).sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(
        kind="barh", ax=ax,
        color="#2563EB", edgecolor="white"
    )
    ax.set_title("Top 20 Feature Importances — Random Forest", fontsize=13)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/feature_importances.png", dpi=150)
    plt.close()
    print(f"      Saved → {PLOTS_DIR}/feature_importances.png")


# ─────────────────────────────────────────────
# STEP 7: SAVE MODEL + METRICS
# ─────────────────────────────────────────────

def save_artifacts(pipeline: Pipeline, metrics: dict) -> None:
    """
    Save the trained pipeline and metrics to disk.

    The .pkl file contains everything needed to make predictions:
      - The fitted preprocessor (scaler + encoder)
      - The trained RandomForest model

    This is what the FastAPI endpoint will load in Phase 7.
    """
    print(f"\n[7/7] Saving model and metrics ...")

    os.makedirs("models", exist_ok=True)

    # Save pipeline as a pickle file
    joblib.dump(pipeline, MODEL_PATH)
    print(f"      Model saved  → {MODEL_PATH}")

    # Save feature list alongside the model
    # FastAPI needs this to validate incoming request fields
    feature_meta = {
        "numerical_features":   NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features":         ALL_FEATURES,
        "target":               TARGET,
    }
    joblib.dump(feature_meta, "models/feature_meta.pkl")
    print(f"      Features saved → models/feature_meta.pkl")

    # Save metrics as JSON (readable by humans and the dashboard)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics saved  → {METRICS_PATH}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_training() -> Pipeline:

    print("\n" + "=" * 60)
    print("  TELCOCHURN360 — PHASE 4: ML TRAINING PIPELINE")
    print("=" * 60)

    X, y, df          = load_data(CLEAN_CSV)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipeline          = build_pipeline()
    cv_roc_auc        = cross_validate(pipeline, X_train, y_train)
    pipeline          = train(pipeline, X_train, y_train)
    metrics, y_pred, y_prob = evaluate(pipeline, X_test, y_test, cv_roc_auc)
    save_plots(pipeline, X_test, y_test, y_pred, y_prob)
    save_artifacts(pipeline, metrics)

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  ROC-AUC (test)    : {metrics['roc_auc']}")
    print(f"  ROC-AUC (CV mean) : {metrics['cv_roc_auc']}")
    print(f"  Precision (churn) : {metrics['precision_churn']}")
    print(f"  Recall    (churn) : {metrics['recall_churn']}")
    print(f"  F1-score  (churn) : {metrics['f1_churn']}")
    print(f"  Accuracy          : {metrics['accuracy']}")
    print("=" * 60)
   

    return pipeline


if __name__ == "__main__":
    run_training()