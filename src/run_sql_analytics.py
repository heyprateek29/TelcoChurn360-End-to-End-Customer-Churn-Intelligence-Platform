"""
src/run_sql_analytics.py
------------------------
Phase 3: Execute all SQL analytics queries using DuckDB.

DuckDB reads the parquet file directly — no database server needed.
Each query result is printed to the terminal and saved as a CSV
under data/processed/sql_outputs/ for use in the dashboard.

Run:
    python src/run_sql_analytics.py
"""

import os
import duckdb
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

PARQUET_PATH = "data/processed/telco_cleaned.parquet"
OUTPUT_DIR   = "data/processed/sql_outputs"
SQL_DIR      = "sql"


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

def setup_duckdb(parquet_path: str) -> duckdb.DuckDBPyConnection:
    """
    Open an in-memory DuckDB connection and register the parquet file
    as a virtual table called 'telco'.

    DuckDB can query parquet files directly with zero ETL overhead.
    This is production-grade — companies like MotherDuck use this pattern.
    """
    print(f"[DuckDB] Loading: {parquet_path}")
    con = duckdb.connect()

    # Register parquet as a view called 'telco'
    # All SQL files reference this view name
    con.execute(f"""
        CREATE VIEW telco AS
        SELECT * FROM read_parquet('{parquet_path}')
    """)

    row_count = con.execute("SELECT COUNT(*) FROM telco").fetchone()[0]
    print(f"[DuckDB] View 'telco' ready — {row_count:,} rows\n")
    return con


# ─────────────────────────────────────────────
# QUERY RUNNER
# ─────────────────────────────────────────────

def run_query(con: duckdb.DuckDBPyConnection,
              sql_file: str,
              label: str) -> pd.DataFrame:
    """Read a SQL file, execute it, return a DataFrame."""

    sql_path = os.path.join(SQL_DIR, sql_file)

    with open(sql_path, "r") as f:
        sql = f.read()

    df = con.execute(sql).df()
    return df


def print_result(df: pd.DataFrame, label: str, max_rows: int = 20) -> None:
    """Pretty-print a query result to the terminal."""
    print("─" * 60)
    print(f"  {label}")
    print("─" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df.head(max_rows).to_string(index=False))
    print(f"\n  [{len(df)} rows returned]\n")


def save_result(df: pd.DataFrame, filename: str) -> None:
    """Save query result CSV for use in the Streamlit dashboard."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# MAIN — Run all queries
# ─────────────────────────────────────────────

def run_all_analytics() -> None:

    print("\n" + "=" * 60)
    print("  TELCOCHURN360 — PHASE 3: SQL ANALYTICS")
    print("=" * 60 + "\n")

    # Verify parquet exists
    if not os.path.exists(PARQUET_PATH):
        raise FileNotFoundError(
            f"Cleaned parquet not found at '{PARQUET_PATH}'.\n"
            "Please run Phase 2 first: python src/ingest.py"
        )

    # Connect DuckDB
    con = setup_duckdb(PARQUET_PATH)

    # ── Query 1: Overall KPIs ─────────────────────────────────
    print("\n[1/6] Churn KPIs")
    df_kpis = run_query(con, "churn_kpis.sql", "OVERALL CHURN KPIs")
    print_result(df_kpis, "OVERALL CHURN KPIs")
    save_result(df_kpis, "kpis.csv")

    # ── Query 2: Revenue at risk ──────────────────────────────
    print("[2/6] Revenue at risk")
    df_rev = run_query(con, "revenue_at_risk.sql", "REVENUE AT RISK BY SEGMENT")
    print_result(df_rev, "REVENUE AT RISK BY SEGMENT")
    save_result(df_rev, "revenue_at_risk.csv")

    # ── Query 3: Churn by contract ────────────────────────────
    print("[3/6] Churn by contract type")
    df_contract = run_query(con, "churn_by_contract.sql", "CHURN BY CONTRACT TYPE")
    print_result(df_contract, "CHURN BY CONTRACT TYPE")
    save_result(df_contract, "churn_by_contract.csv")

    # ── Query 4: Churn by tenure segment ─────────────────────
    print("[4/6] Churn by tenure segment")
    df_tenure = run_query(con, "churn_by_tenure.sql", "CHURN BY TENURE SEGMENT")
    print_result(df_tenure, "CHURN BY TENURE SEGMENT")
    save_result(df_tenure, "churn_by_tenure.csv")

    # ── Query 5: Service usage analysis ──────────────────────
    print("[5/6] Service usage and churn correlation")
    df_services = run_query(con, "service_usage_analysis.sql", "SERVICE USAGE ANALYSIS")
    print_result(df_services, "SERVICE USAGE ANALYSIS")
    save_result(df_services, "service_usage.csv")

    # ── Query 6: Customer segmentation ───────────────────────
    print("[6/6] Customer segmentation")
    df_segments = run_query(con, "customer_segmentation.sql", "CUSTOMER SEGMENTATION")
    print_result(df_segments, "CUSTOMER SEGMENTATION")
    save_result(df_segments, "customer_segments.csv")

    # ── Bonus: Retention analysis (saved but not printed in full) ──
    df_retention = run_query(con, "retention_analysis.sql", "RETENTION ANALYSIS")
    save_result(df_retention, "retention_analysis.csv")
    print("[+]  Retention analysis saved to sql_outputs/retention_analysis.csv")

    # ── Business insight summary ──────────────────────────────
    print_business_insights(df_kpis, df_contract, df_tenure, df_services)

    con.close()
    print("\nPhase 3 complete. SQL outputs saved to data/processed/sql_outputs/\n")


def print_business_insights(df_kpis, df_contract, df_tenure, df_services) -> None:
    """Extract and print key business insights from query results."""

    print("\n" + "=" * 60)
    print("  KEY BUSINESS INSIGHTS")
    print("=" * 60)

    # KPI insights
    churn_rate   = df_kpis["churn_rate_pct"].iloc[0]
    rev_lost     = df_kpis["monthly_revenue_lost"].iloc[0]
    rev_risk_pct = df_kpis["revenue_at_risk_pct"].iloc[0]
    tenure_churn = df_kpis["avg_tenure_churned_months"].iloc[0]
    tenure_ret   = df_kpis["avg_tenure_retained_months"].iloc[0]

    print(f"""
  Overall churn rate   : {churn_rate}%
  Monthly revenue lost : ${rev_lost:,.0f}
  Revenue at risk      : {rev_risk_pct}% of total MRR
  Avg tenure (churned) : {tenure_churn:.0f} months
  Avg tenure (retained): {tenure_ret:.0f} months

  → Retained customers stay {tenure_ret - tenure_churn:.0f} months longer on average.
  → We lose ${rev_lost * 12:,.0f} annualised revenue to churn.
    """)

    # Contract insight
    mtm = df_contract[df_contract["contract"] == "Month-to-month"]
    if not mtm.empty:
        mtm_rate = mtm["churn_rate_pct"].iloc[0]
        print(f"  Month-to-month churn : {mtm_rate}%  ← primary driver")

    # Tenure insight
    new_seg = df_tenure[df_tenure["tenure_segment"] == "New"]
    if not new_seg.empty:
        new_rate = new_seg["churn_rate_pct"].iloc[0]
        print(f"  New customer churn   : {new_rate}%  ← critical onboarding risk")

    # Service insight
    most_protective = df_services[df_services["churn_lift_pct"] < 0].head(1)
    if not most_protective.empty:
        svc  = most_protective["service"].iloc[0]
        lift = most_protective["churn_lift_pct"].iloc[0]
        print(f"  Most protective svc  : {svc} ({lift:+.1f}% churn lift)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_all_analytics()