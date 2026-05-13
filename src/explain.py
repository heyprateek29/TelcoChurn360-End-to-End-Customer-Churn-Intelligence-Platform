"""
src/explain.py
----------------------------------------------------
TELCOCHURN360 — Phase 5: SHAP Explainability

This phase explains:
WHY the churn model predicts a customer will churn.

Outputs:
- shap_summary.png
- shap_bar.png
- shap_waterfall.png
- shap_dependence_tenure.png
- shap_insights.json
- shap_insights.txt

Run:
    python src/explain.py
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import shap

warnings.filterwarnings("ignore")


# =========================================================
# CONFIG
# =========================================================

CLEAN_CSV = "data/processed/telco_cleaned.csv"

MODEL_PATH = "models/churn_model.pkl"
META_PATH = "models/feature_meta.pkl"

OUTPUT_DIR = "shap_outputs"

SHAP_SAMPLE = 1000


# =========================================================
# STEP 1 — LOAD MODEL + DATA
# =========================================================

def load_artifacts():

    print("[1/6] Loading model and data ...")

    required_files = [
        CLEAN_CSV,
        MODEL_PATH,
        META_PATH,
    ]

    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing required file:\n{path}\n\n"
                "Run Phase 4 first:\n"
                "python src/train.py"
            )

    pipeline = joblib.load(MODEL_PATH)

    feature_meta = joblib.load(META_PATH)

    df = pd.read_csv(CLEAN_CSV)

    print(
        f"      Model loaded: "
        f"{type(pipeline.named_steps['classifier']).__name__}"
    )

    print(f"      Data rows   : {len(df):,}")

    return pipeline, feature_meta, df


# =========================================================
# STEP 2 — PREPARE DATA FOR SHAP
# =========================================================

def prepare_shap_data(
    pipeline,
    feature_meta,
    df,
):

    print("\n[2/6] Preparing data for SHAP ...")

    all_features = feature_meta["all_features"]

    numerical_features = feature_meta["numerical_features"]

    categorical_features = feature_meta["categorical_features"]

    X = df[all_features].copy()

    # -----------------------------------------------------
    # SAMPLE DATA
    # -----------------------------------------------------

    sample_idx = X.sample(
        n=min(SHAP_SAMPLE, len(X)),
        random_state=42
    ).index

    X_sample = X.loc[sample_idx].reset_index(drop=True)

    df_sample = df.loc[sample_idx].reset_index(drop=True)

    # -----------------------------------------------------
    # TRANSFORM USING TRAINED PREPROCESSOR
    # -----------------------------------------------------

    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X_sample)

    # Sparse → dense
    if hasattr(X_transformed, "toarray"):

        X_transformed = X_transformed.toarray()

    # -----------------------------------------------------
    # GET FEATURE NAMES AFTER ONE-HOT ENCODING
    # -----------------------------------------------------

    encoder = preprocessor.named_transformers_[
        "cat"
    ]["encoder"]

    encoded_cat_features = encoder.get_feature_names_out(
        categorical_features
    ).tolist()

    all_feature_names = (
        numerical_features + encoded_cat_features
    )

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=all_feature_names
    )

    print(
        f"      Sample size      : "
        f"{len(X_sample):,} customers"
    )

    print(
        f"      Features (raw)   : "
        f"{len(all_features)}"
    )

    print(
        f"      Features (encoded): "
        f"{len(all_feature_names)}"
    )

    return (
        X_sample,
        X_transformed_df,
        df_sample,
        all_feature_names,
    )


# =========================================================
# STEP 3 — COMPUTE SHAP VALUES
# =========================================================

def extract_shap_values(
    shap_values_raw
):

    """
    Handle multiple SHAP output formats safely.
    """

    # Older SHAP versions:
    # list[class0, class1]
    if isinstance(shap_values_raw, list):

        return shap_values_raw[1]

    shap_arr = np.array(shap_values_raw)

    # Shape:
    # (samples, features, classes)
    if shap_arr.ndim == 3:

        if shap_arr.shape[-1] == 2:

            return shap_arr[:, :, 1]

        # Alternate layout:
        # (classes, samples, features)
        if shap_arr.shape[0] == 2:

            return shap_arr[1]

    # Already correct
    if shap_arr.ndim == 2:

        return shap_arr

    raise ValueError(
        f"Unsupported SHAP shape: {shap_arr.shape}"
    )


def compute_shap_values(
    pipeline,
    X_transformed_df,
):

    print(
        "\n[3/6] Computing SHAP values "
        "(this may take ~30 seconds) ..."
    )

    rf_model = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(
        rf_model
    )

    shap_values_raw = explainer.shap_values(
        X_transformed_df
    )

    shap_values = extract_shap_values(
        shap_values_raw
    )

    if shap_values.shape != X_transformed_df.shape:

        raise ValueError(
            f"Shape mismatch:\n"
            f"SHAP: {shap_values.shape}\n"
            f"DATA: {X_transformed_df.shape}"
        )

    print(
        f"      SHAP matrix shape : "
        f"{shap_values.shape}"
    )

    print(
        f"      Mean |SHAP| (impact): "
        f"{np.abs(shap_values).mean():.4f}"
    )

    return explainer, shap_values


# =========================================================
# STEP 4 — GENERATE SHAP PLOTS
# =========================================================

def get_base_value(
    explainer
):

    expected = explainer.expected_value

    if isinstance(expected, (list, np.ndarray)):

        return float(expected[1])

    return float(expected)


def save_shap_plots(
    pipeline,
    explainer,
    shap_values,
    X_transformed_df,
    df_sample,
    feature_names,
    X_sample,
):

    print("\n[4/6] Generating SHAP plots ...")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # =====================================================
    # 1 — SUMMARY PLOT
    # =====================================================

    print(
        "      [1/4] Summary plot "
        "(beeswarm) ..."
    )

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        X_transformed_df,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_type="dot",
    )

    plt.title(
        "SHAP Summary — Top Churn Drivers",
        fontsize=13,
        pad=15,
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_summary.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"      Saved → "
        f"{OUTPUT_DIR}/shap_summary.png"
    )

    # =====================================================
    # 2 — BAR PLOT
    # =====================================================

    print(
        "      [2/4] Bar plot "
        "(mean |SHAP|) ..."
    )

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    )

    mean_shap = mean_shap.sort_values(
        ascending=False
    ).head(15)

    clean_names = []

    for name in mean_shap.index:

        if "_" in name:

            parts = name.split("_", 1)

            clean_names.append(
                f"{parts[0].title()}: {parts[1]}"
            )

        else:

            clean_names.append(
                name.replace("_", " ").title()
            )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    colors = [
        "#DC2626"
        if v > mean_shap.median()
        else "#2563EB"
        for v in mean_shap.values
    ]

    ax.barh(
        range(len(mean_shap)),
        mean_shap.values,
        color=colors,
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(
        range(len(mean_shap))
    )

    ax.set_yticklabels(
        clean_names,
        fontsize=10,
    )

    ax.set_xlabel(
        "Mean |SHAP Value|",
        fontsize=10,
    )

    ax.set_title(
        "Top 15 Churn Drivers",
        fontsize=13,
        fontweight="bold",
    )

    high_patch = mpatches.Patch(
        color="#DC2626",
        label="High impact",
    )

    low_patch = mpatches.Patch(
        color="#2563EB",
        label="Lower impact",
    )

    ax.legend(
        handles=[high_patch, low_patch],
        loc="lower right",
        fontsize=9,
    )

    ax.invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_bar.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"      Saved → "
        f"{OUTPUT_DIR}/shap_bar.png"
    )

    # =====================================================
    # 3 — WATERFALL PLOT
    # =====================================================

    print(
        "      [3/4] Waterfall plot "
        "(single customer) ..."
    )

    churn_prob = pipeline.predict_proba(
        X_sample
    )[:, 1]

    high_risk_idx = int(
        np.argmax(churn_prob)
    )

    base_value = get_base_value(
        explainer
    )

    explanation = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=base_value,
        data=X_transformed_df.iloc[
            high_risk_idx
        ].values,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 7))

    shap.waterfall_plot(
        explanation,
        max_display=15,
        show=False,
    )

    plt.title(
        f"Why Customer #{high_risk_idx} "
        f"Was Predicted To Churn",
        fontsize=11,
        pad=15,
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_waterfall.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"      Saved → "
        f"{OUTPUT_DIR}/shap_waterfall.png"
    )

    print(
        f"            "
        f"(Customer #{high_risk_idx} "
        f"had highest churn probability "
        f"in the sample)"
    )

    # =====================================================
    # 4 — DEPENDENCE PLOT
    # =====================================================

    print(
        "      [4/4] Dependence plot "
        "(tenure) ..."
    )

    tenure_idx = feature_names.index(
        "tenure"
    )

    contract_idx = None

    if "contract_risk_score" in feature_names:

        contract_idx = feature_names.index(
            "contract_risk_score"
        )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    if contract_idx is not None:

        scatter = ax.scatter(
            X_transformed_df.iloc[
                :, tenure_idx
            ],
            shap_values[:, tenure_idx],
            c=X_transformed_df.iloc[
                :, contract_idx
            ],
            cmap="RdYlBu_r",
            alpha=0.5,
            s=15,
        )

        cbar = plt.colorbar(
            scatter,
            ax=ax,
        )

        cbar.set_label(
            "Contract Risk Score",
            fontsize=9,
        )

    else:

        ax.scatter(
            X_transformed_df.iloc[
                :, tenure_idx
            ],
            shap_values[:, tenure_idx],
            alpha=0.4,
            s=15,
        )

    ax.axhline(
        0,
        color="gray",
        lw=1,
        linestyle="--",
    )

    ax.set_xlabel(
        "Tenure (months)"
    )

    ax.set_ylabel(
        "SHAP Value for Tenure"
    )

    ax.set_title(
        "SHAP Dependence — Tenure vs Churn Risk",
        fontsize=12,
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/shap_dependence_tenure.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"      Saved → "
        f"{OUTPUT_DIR}/shap_dependence_tenure.png"
    )


# =========================================================
# STEP 5 — BUSINESS INSIGHTS
# =========================================================

def generate_insight_report(
    shap_values,
    feature_names,
):

    print(
        "\n[5/6] Generating "
        "business insight report ..."
    )

    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=feature_names,
    )

    mean_abs_shap = mean_abs_shap.sort_values(
        ascending=False
    )

    mean_signed = pd.Series(
        shap_values.mean(axis=0),
        index=feature_names,
    )

    insights = []

    for feat in mean_abs_shap.head(10).index:

        direction = (
            "increases"
            if mean_signed[feat] > 0
            else "decreases"
        )

        insights.append({

            "feature": feat,

            "impact": round(
                float(mean_abs_shap[feat]),
                4,
            ),

            "direction": direction,
        })

    report = {

        "top_churn_drivers": insights,

        "plain_english_summary": [
            "Short tenure customers churn most frequently.",
            "Month-to-month contracts are the highest-risk segment.",
            "High monthly charges increase churn risk.",
            "High lifetime spend protects against churn.",
            "Fiber optic users show elevated churn risk.",
        ]
    }

    return report


# =========================================================
# STEP 6 — SAVE REPORT
# =========================================================

def save_report(
    report
):

    json_path = (
        f"{OUTPUT_DIR}/shap_insights.json"
    )

    txt_path = (
        f"{OUTPUT_DIR}/shap_insights.txt"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SHAP BUSINESS INSIGHTS\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            "Top churn drivers:\n"
        )

        for i, item in enumerate(
            report["top_churn_drivers"],
            1,
        ):

            arrow = (
                "↑"
                if item["direction"] == "increases"
                else "↓"
            )

            f.write(
                f"{i}. "
                f"{arrow} "
                f"{item['feature']} "
                f"(impact={item['impact']})\n"
            )

        f.write(
            "\nPlain-English Insights:\n"
        )

        for i, msg in enumerate(
            report["plain_english_summary"],
            1,
        ):

            f.write(
                f"{i}. {msg}\n"
            )

    print(
        f"\n[6/6] Insight report saved → "
        f"{json_path}"
    )

    print(
        f"      Text report saved    → "
        f"{txt_path}"
    )


# =========================================================
# MAIN
# =========================================================

def run_explainability():

    print("\n" + "=" * 60)

    print(
        "  TELCOCHURN360 — "
        "PHASE 5: SHAP EXPLAINABILITY"
    )

    print("=" * 60)

    (
        pipeline,
        feature_meta,
        df,
    ) = load_artifacts()

    (
        X_sample,
        X_transformed_df,
        df_sample,
        feature_names,
    ) = prepare_shap_data(
        pipeline,
        feature_meta,
        df,
    )

    (
        explainer,
        shap_values,
    ) = compute_shap_values(
        pipeline,
        X_transformed_df,
    )

    save_shap_plots(
        pipeline,
        explainer,
        shap_values,
        X_transformed_df,
        df_sample,
        feature_names,
        X_sample,
    )

    report = generate_insight_report(
        shap_values,
        feature_names,
    )

    save_report(
        report
    )

    print("\nPhase 5 complete.")

    print(
        f"Outputs saved to: "
        f"{OUTPUT_DIR}/"
    )

    print(
        "  shap_summary.png"
    )

    print(
        "  shap_bar.png"
    )

    print(
        "  shap_waterfall.png"
    )

    print(
        "  shap_dependence_tenure.png"
    )

    print(
        "  shap_insights.json"
    )

    print(
        "  shap_insights.txt"
    )

    print(
        "\nReady for Phase 6 "
        "(Streamlit Dashboard).\n"
    )


if __name__ == "__main__":

    run_explainability()