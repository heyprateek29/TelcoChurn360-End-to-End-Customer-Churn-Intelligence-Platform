"""
src/run_dbt.py
--------------
Phase 6: Load cleaned data into DuckDB then run the dbt project.

dbt does NOT load raw data — it transforms data that's already
in a database. This script bridges the gap:
  1. Load the cleaned parquet into a DuckDB file
  2. Run dbt debug   → verify connection
  3. Run dbt run     → build all models
  4. Run dbt test    → validate data quality
  5. Run dbt docs    → generate lineage docs (optional)

Run:
    python src/run_dbt.py
"""

import os
import subprocess
import duckdb
import pandas as pd


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLEAN_PARQUET  = "data/processed/telco_cleaned.parquet"
DUCKDB_PATH    = "data/processed/churn_dbt.duckdb"
DBT_PROJECT    = "dbt_churn"


# ─────────────────────────────────────────────
# STEP 1: LOAD PARQUET INTO DUCKDB
# ─────────────────────────────────────────────

def seed_duckdb() -> None:
    """
    Read the cleaned parquet and write it into the DuckDB file
    as a table called 'customers' in the 'main' schema.

    dbt's source() macro will point to this table.
    """
    print("[1/4] Seeding DuckDB from cleaned parquet ...")

    if not os.path.exists(CLEAN_PARQUET):
        raise FileNotFoundError(
            f"Cleaned parquet not found: {CLEAN_PARQUET}\n"
            "Run Phase 2 first: python src/ingest.py"
        )

    # Connect to the DuckDB file (creates it if it doesn't exist)
    con = duckdb.connect(DUCKDB_PATH)

    # Drop and recreate so re-runs are idempotent
    con.execute("DROP TABLE IF EXISTS customers")

    # Load directly from parquet — no pandas needed
    con.execute(f"""
        CREATE TABLE customers AS
        SELECT * FROM read_parquet('{CLEAN_PARQUET}')
    """)

    row_count = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    col_count = len(con.execute("DESCRIBE customers").fetchall())

    con.close()

    print(f"      DuckDB file   : {DUCKDB_PATH}")
    print(f"      Table         : customers")
    print(f"      Rows          : {row_count:,}")
    print(f"      Columns       : {col_count}")


# ─────────────────────────────────────────────
# STEP 2: RUN DBT COMMAND
# ─────────────────────────────────────────────

def run_dbt_command(args: list, label: str) -> bool:
    """
    Run a dbt CLI command from inside the dbt_churn/ directory.
    dbt must be run from the project folder (where dbt_project.yml lives).
    """
    cmd = ["dbt"] + args
    print(f"\n[{label}] Running: {' '.join(cmd)}")
    print("─" * 50)

    result = subprocess.run(
        cmd,
        cwd=DBT_PROJECT,           # Always run from inside the project folder
        capture_output=False,      # Stream output directly to terminal
        text=True,
    )

    if result.returncode != 0:
        print(f"\n  ERROR: dbt command failed (exit code {result.returncode})")
        return False

    return True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_dbt_pipeline() -> None:

    print("\n" + "=" * 55)
    print("  TELCOCHURN360 — PHASE 6: dbt MODELS")
    print("=" * 55 + "\n")

    # Step 1: Seed the database
    seed_duckdb()

    # Step 2: dbt debug — verify connection and config
    print("\n[2/4] Verifying dbt connection ...")
    ok = run_dbt_command(["debug"], "2/4")
    if not ok:
        print("\n  Check that ~/.dbt/profiles.yml is created correctly.")
        print("  See setup instructions above.")
        return

    # Step 3: dbt run — build all models
    print("\n[3/4] Building dbt models ...")
    ok = run_dbt_command(["run"], "3/4")
    if not ok:
        print("\n  Check model SQL files for syntax errors.")
        return

    # Step 4: dbt test — run all data quality tests
    print("\n[4/4] Running dbt tests ...")
    run_dbt_command(["test"], "4/4")

    # Summary of what was built
    print_dbt_summary()

    print("\n" + "=" * 55)
    print("  Phase 6 complete. dbt models built and tested.")
    print("  Ready for Phase 7 (FastAPI).")
    print("=" * 55 + "\n")


def print_dbt_summary() -> None:
    """Query the built mart tables and show row counts."""
    print("\n" + "=" * 55)
    print("  DBT BUILD SUMMARY")
    print("=" * 55)

    try:
        con = duckdb.connect(DUCKDB_PATH)

        tables = [
            ("staging.stg_customers",       "Staging: all customers"),
            ("marts.mart_churn_kpis",        "Mart: churn KPIs"),
            ("marts.mart_churn_by_segment",  "Mart: churn by segment"),
            ("marts.mart_customer_risk",     "Mart: customer risk scores"),
        ]

        for table, label in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {label:<38} {n:>6} rows")
            except Exception:
                print(f"  {label:<38} (not found — check dbt run output)")

        # Show a sample of the risk mart
        print("\n  Sample from mart_customer_risk (top 5 high-risk):")
        try:
            sample = con.execute("""
                SELECT customer_id, tenure, contract,
                       risk_score, risk_category, recommended_action
                FROM marts.mart_customer_risk
                WHERE risk_category IN ('Critical','High')
                ORDER BY risk_score DESC
                LIMIT 5
            """).df()
            pd.set_option("display.max_colwidth", 45)
            pd.set_option("display.width", 120)
            print(sample.to_string(index=False))
        except Exception as e:
            print(f"  (Could not query mart: {e})")

        con.close()

    except Exception as e:
        print(f"  Could not connect to DuckDB: {e}")

    print("=" * 55)


if __name__ == "__main__":
    run_dbt_pipeline()