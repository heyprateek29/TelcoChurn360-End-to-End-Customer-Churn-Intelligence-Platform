"""
src/ingest.py
-------------
Phase 2: Data ingestion and cleaning pipeline.

Responsibilities:
  1. Load raw IBM Telco CSV
  2. Fix data types and missing values
  3. Standardise column names
  4. Engineer business features
  5. Save cleaned parquet + CSV for downstream phases

Run:
    python src/ingest.py
"""

import os
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RAW_PATH       = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_CSV      = "data/processed/telco_cleaned.csv"
CLEAN_PARQUET  = "data/processed/telco_cleaned.parquet"


# ─────────────────────────────────────────────
# STEP 1: LOAD
# ─────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and do the first sanity check."""
    print(f"[1/6] Loading raw data from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
            "and place the CSV at data/raw/"
        )

    df = pd.read_csv(path)
    print(f"      Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# STEP 2: STANDARDISE COLUMN NAMES
# ─────────────────────────────────────────────

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all column names to lowercase snake_case.
    E.g.  'TotalCharges' → 'total_charges'
          'customerID'   → 'customer_id'
    """
    print("[2/6] Standardising column names ...")

    df = df.copy()

    # Insert underscore before capital letters, then lowercase everything
    import re
    def to_snake(name: str) -> str:
        name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', name)
        return name.lower().strip()

    df.columns = [to_snake(c) for c in df.columns]

    # Rename a few columns for extra clarity
    rename_map = {
        "customerid":        "customer_id",
        "seniorcitizen":     "senior_citizen",
        "phoneservice":      "phone_service",
        "multiplelines":     "multiple_lines",
        "internetservice":   "internet_service",
        "onlinesecurity":    "online_security",
        "onlinebackup":      "online_backup",
        "deviceprotection":  "device_protection",
        "techsupport":       "tech_support",
        "streamingtv":       "streaming_tv",
        "streamingmovies":   "streaming_movies",
        "paperlessbilling":  "paperless_billing",
        "paymentmethod":     "payment_method",
        "monthlycharges":    "monthly_charges",
        "totalcharges":      "total_charges",
    }
    df.rename(columns=rename_map, inplace=True)

    print(f"      Columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────
# STEP 3: FIX DATA TYPES & MISSING VALUES
# ─────────────────────────────────────────────

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Known issues in this dataset:
      - 'total_charges' is object (string) due to spaces for new customers
      - 'senior_citizen' is 0/1 integer, convert to Yes/No for consistency
      - 'churn' is Yes/No string, create a numeric 'churn_flag' column
    """
    print("[3/6] Fixing data types and missing values ...")

    df = df.copy()

    # --- total_charges: coerce spaces to NaN, then fill with monthly_charges
    # (New customers with tenure=0 have no total_charges recorded)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    missing_tc = df["total_charges"].isna().sum()
    print(f"      total_charges: {missing_tc} blank values found (new customers)")

    # Business logic: if tenure=0, total_charges should equal monthly_charges
    df["total_charges"] = df["total_charges"].fillna(df["monthly_charges"])

    # --- senior_citizen: 0/1 → "No"/"Yes" (consistent with other binary cols)
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})

    # --- churn: create numeric flag (0/1) alongside original Yes/No
    df["churn_flag"] = df["churn"].map({"Yes": 1, "No": 0})

    # --- tenure: should be integer
    df["tenure"] = df["tenure"].astype(int)

    print(f"      Churn rate: {df['churn_flag'].mean():.1%}  "
          f"({df['churn_flag'].sum():,} churned / {len(df):,} total)")

    return df


# ─────────────────────────────────────────────
# STEP 4: CLEAN FREE-TEXT VALUES
# ─────────────────────────────────────────────

def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some service columns contain 'No internet service' or 'No phone service'
    when the parent service is not subscribed.  Collapse these to plain 'No'
    so ML encoders see a binary flag, not 3 categories.
    """
    print("[4/6] Cleaning categorical values ...")

    df = df.copy()

    # Columns that contain 'No internet service'
    internet_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    for col in internet_cols:
        before = df[col].value_counts().to_dict()
        df[col] = df[col].replace("No internet service", "No")
        after  = df[col].value_counts().to_dict()
        if before != after:
            print(f"      {col}: collapsed 'No internet service' → 'No'")

    # 'multiple_lines' can contain 'No phone service'
    df["multiple_lines"] = df["multiple_lines"].replace("No phone service", "No")
    print("      multiple_lines: collapsed 'No phone service' → 'No'")

    return df


# ─────────────────────────────────────────────
# STEP 5: ENGINEER BUSINESS FEATURES
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-meaningful derived columns.

    These features are what a real product analyst would create:
      - tenure_segment       : Customer lifecycle stage
      - monthly_charges_tier : Price bucket (Low / Mid / High / Premium)
      - num_services         : How many add-on services the customer has
      - has_fiber            : Quick flag for high-risk segment
      - revenue_at_risk      : Monthly revenue that would be lost if churned
      - avg_monthly_spend    : total_charges / tenure (normalised spend)
      - contract_risk_score  : Ordinal risk by contract type
      - support_score        : Customers with tech support are less likely to churn
    """
    print("[5/6] Engineering business features ...")

    df = df.copy()

    # ── Tenure segment ──────────────────────────────────────────────────────
    # Buckets based on subscription age in months
    def tenure_segment(months: int) -> str:
        if months <= 6:
            return "New"          # 0–6 months: highest churn risk
        elif months <= 12:
            return "Growing"      # 7–12 months
        elif months <= 24:
            return "Established"  # 1–2 years
        elif months <= 48:
            return "Loyal"        # 2–4 years
        else:
            return "Champion"     # 4+ years: lowest churn risk

    df["tenure_segment"] = df["tenure"].apply(tenure_segment)

    # ── Monthly charges tier ────────────────────────────────────────────────
    # Quartile-based pricing buckets
    q1, q2, q3 = (
        df["monthly_charges"].quantile(0.25),
        df["monthly_charges"].quantile(0.50),
        df["monthly_charges"].quantile(0.75),
    )

    def charges_tier(charge: float) -> str:
        if charge <= q1:   return "Low"
        elif charge <= q2: return "Mid"
        elif charge <= q3: return "High"
        else:              return "Premium"

    df["monthly_charges_tier"] = df["monthly_charges"].apply(charges_tier)
    print(f"      Charge tier thresholds — Low≤{q1:.0f}  Mid≤{q2:.0f}  "
          f"High≤{q3:.0f}  Premium>{q3:.0f}")

    # ── Number of add-on services ───────────────────────────────────────────
    # Counts how many value-added services the customer subscribes to
    service_cols = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
        "multiple_lines",
    ]
    df["num_services"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )

    # ── Fiber flag ──────────────────────────────────────────────────────────
    df["has_fiber"] = (df["internet_service"] == "Fiber optic").astype(int)

    # ── Revenue at risk ─────────────────────────────────────────────────────
    # If this customer churns, this is the monthly revenue we lose
    # Only meaningful for churned customers; kept for all for prediction
    df["revenue_at_risk"] = df["monthly_charges"]

    # ── Average monthly spend ───────────────────────────────────────────────
    # Handles tenure=0 edge case
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0,
        df["total_charges"] / df["tenure"],
        df["monthly_charges"],
    )

    # ── Contract risk score ─────────────────────────────────────────────────
    # Month-to-month customers are easiest to churn; two-year hardest
    contract_risk = {
        "Month-to-month": 3,   # High risk
        "One year":        2,   # Medium risk
        "Two year":        1,   # Low risk
    }
    df["contract_risk_score"] = df["contract"].map(contract_risk)

    # ── Support score ───────────────────────────────────────────────────────
    # More support services → more engaged → less likely to churn
    df["support_score"] = (
        (df["tech_support"] == "Yes").astype(int) +
        (df["online_security"] == "Yes").astype(int) +
        (df["device_protection"] == "Yes").astype(int)
    )

    print(f"      New features added: tenure_segment, monthly_charges_tier, "
          f"num_services, has_fiber, revenue_at_risk, avg_monthly_spend, "
          f"contract_risk_score, support_score")

    return df


# ─────────────────────────────────────────────
# STEP 6: SAVE
# ─────────────────────────────────────────────

def save_clean(df: pd.DataFrame) -> None:
    """Save cleaned data in both CSV (readable) and Parquet (fast)."""
    print("[6/6] Saving cleaned data ...")

    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(CLEAN_CSV, index=False)
    print(f"      CSV saved     → {CLEAN_CSV}")

    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"      Parquet saved → {CLEAN_PARQUET}")


# ─────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print a quick data quality and business summary."""

    print("\n" + "="*55)
    print("  INGESTION SUMMARY")
    print("="*55)
    print(f"  Total customers     : {len(df):,}")
    print(f"  Churned             : {df['churn_flag'].sum():,}  "
          f"({df['churn_flag'].mean():.1%})")
    print(f"  Retained            : {(1 - df['churn_flag']).sum():,}  "
          f"({1 - df['churn_flag'].mean():.1%})")
    print(f"  Avg monthly charges : ${df['monthly_charges'].mean():.2f}")
    print(f"  Avg tenure          : {df['tenure'].mean():.1f} months")
    print(f"  Missing values      : {df.isna().sum().sum()}")
    print()
    print("  Churn by tenure segment:")
    seg = (
        df.groupby("tenure_segment")["churn_flag"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .sort_values("churn_rate", ascending=False)
    )
    for seg_name, row in seg.iterrows():
        bar = "█" * int(row["churn_rate"] * 20)
        print(f"    {seg_name:<12} {bar:<20} {row['churn_rate']:.1%}  "
              f"(n={int(row['customers']):,})")
    print()
    print("  Churn by contract type:")
    con = (
        df.groupby("contract")["churn_flag"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "churn_rate", "count": "customers"})
        .sort_values("churn_rate", ascending=False)
    )
    for con_name, row in con.iterrows():
        bar = "█" * int(row["churn_rate"] * 20)
        print(f"    {con_name:<20} {bar:<20} {row['churn_rate']:.1%}  "
              f"(n={int(row['customers']):,})")
    print("="*55)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """Run all ingestion steps in order and return the cleaned DataFrame."""
    print("\n" + "="*55)
    print("  TELCOCHURN360 — PHASE 2: INGESTION & CLEANING")
    print("="*55 + "\n")

    df = load_raw(RAW_PATH)
    df = standardise_columns(df)
    df = fix_dtypes(df)
    df = clean_categories(df)
    df = engineer_features(df)
    save_clean(df)
    print_summary(df)

    print("\nPhase 2 complete. Cleaned data ready for Phase 3.\n")
    return df


if __name__ == "__main__":
    run_pipeline()