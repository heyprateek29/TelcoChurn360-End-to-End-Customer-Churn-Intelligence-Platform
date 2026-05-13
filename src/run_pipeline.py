"""
src/run_pipeline.py
-------------------
Local pipeline runner — runs the full pipeline without Airflow.

Use this if:
  - Airflow is not installed
  - You want to run everything in one command for demos
  - You're testing changes end-to-end quickly

It mirrors exactly what the Airflow DAG does, with the same
validation logic and logging, but runs synchronously in one process.

Run:
    python src/run_pipeline.py

Optional flags:
    python src/run_pipeline.py --skip-train    # Skip ML training (use existing model)
    python src/run_pipeline.py --skip-shap     # Skip SHAP (saves ~45s)
    python src/run_pipeline.py --only-ingest   # Just run ingestion + SQL
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

class PipelineStep:
    """Context manager that times each pipeline step."""

    def __init__(self, name: str, step_num: int, total: int):
        self.name     = name
        self.step_num = step_num
        self.total    = total
        self.start    = None

    def __enter__(self):
        print(f"\n{'─'*55}")
        print(f"  Step {self.step_num}/{self.total}: {self.name}")
        print(f"{'─'*55}")
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            print(f"\n  ✓ {self.name} completed in {elapsed:.1f}s")
        else:
            print(f"\n  ✗ {self.name} FAILED after {elapsed:.1f}s")
            print(f"  Error: {exc_val}")
        return False   # Don't suppress exceptions


def check_file(path: str, min_bytes: int = 100) -> bool:
    """Check a file exists and is non-empty."""
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes


# ─────────────────────────────────────────────
# PIPELINE STEPS
# ─────────────────────────────────────────────

def step_ingest() -> dict:
    """Run ingestion and cleaning pipeline."""
    from src.ingest import run_pipeline
    df = run_pipeline()
    return {
        "rows":       len(df),
        "churn_rate": float(round(df["churn_flag"].mean(), 4)),
    }


def step_sql() -> dict:
    """Run all SQL analytics queries."""
    from src.run_sql_analytics import run_all_analytics
    run_all_analytics()

    output_dir = "data/processed/sql_outputs"
    files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    return {"files_written": len(files)}


def step_train() -> dict:
    """Train and evaluate the ML model."""
    from src.train import run_training
    run_training()

    with open("models/model_metrics.json") as f:
        metrics = json.load(f)

    roc_auc = metrics["roc_auc"]

    # Quality gate — same as Airflow DAG
    if roc_auc < 0.75:
        raise ValueError(
            f"Model quality gate FAILED: ROC-AUC={roc_auc:.4f} "
            f"(minimum: 0.75). Check data quality."
        )

    return {"roc_auc": roc_auc, "accuracy": metrics["accuracy"]}


def step_shap() -> dict:
    """Generate SHAP explanations."""
    from src.explain import run_explainability
    run_explainability()

    plots = [f for f in os.listdir("shap_outputs")
             if f.endswith(".png")]
    return {"plots_generated": len(plots)}


def step_validate() -> dict:
    """Validate all pipeline outputs are present and non-empty."""

    print("  Checking output files ...")

    checks = [
        ("data/processed/telco_cleaned.csv",              10_000),
        ("data/processed/telco_cleaned.parquet",          10_000),
        ("data/processed/sql_outputs/kpis.csv",           100),
        ("data/processed/sql_outputs/customer_segments.csv", 100),
        ("models/churn_model.pkl",                        10_000),
        ("models/model_metrics.json",                     100),
        ("shap_outputs/shap_insights.json",               100),
    ]

    passed = 0
    failed = []

    for path, min_bytes in checks:
        ok = check_file(path, min_bytes)
        icon = "✓" if ok else "✗"
        size = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"    [{icon}] {path:<50} ({size:,} bytes)")
        if ok:
            passed += 1
        else:
            failed.append(path)

    if failed:
        raise ValueError(
            f"Validation failed — {len(failed)} file(s) missing or empty:\n"
            + "\n".join(f"  - {f}" for f in failed)
        )

    return {"checks_passed": passed}


def step_log_run(results: dict) -> None:
    """Write a run log for monitoring."""
    os.makedirs("logs", exist_ok=True)
    log = {
        "completed_at": datetime.now().isoformat(),
        "status":       "success",
        **results,
    }
    with open("logs/last_run.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Run log saved → logs/last_run.json")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="TelcoChurn360 — local pipeline runner"
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip ML training (use existing model)")
    parser.add_argument("--skip-shap",  action="store_true",
                        help="Skip SHAP generation (saves ~45 seconds)")
    parser.add_argument("--only-ingest", action="store_true",
                        help="Run only ingestion and SQL analytics")
    return parser.parse_args()


def run_full_pipeline():

    args     = parse_args()
    t_start  = time.time()
    results  = {}

    print("\n" + "="*55)
    print("  TELCOCHURN360 — FULL PIPELINE RUN")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    # Determine active steps
    steps_needed = []
    total        = 2 if args.only_ingest else (
                   4 if args.skip_shap else
                   3 if args.skip_train else 5
                   )

    try:
        # ── Step 1: Ingest ─────────────────────────────────────
        with PipelineStep("Data Ingestion & Cleaning", 1, total) as step:
            results["ingest"] = step_ingest()

        # ── Step 2: SQL ────────────────────────────────────────
        with PipelineStep("SQL Analytics", 2, total) as step:
            results["sql"] = step_sql()

        if args.only_ingest:
            print("\n  --only-ingest flag set. Stopping after SQL.")
        else:
            # ── Step 3: Train ──────────────────────────────────
            if not args.skip_train:
                with PipelineStep("ML Model Training", 3, total) as step:
                    results["train"] = step_train()
            else:
                print("\n  Skipping training (--skip-train). "
                      "Using existing model.")
                with open("models/model_metrics.json") as f:
                    results["train"] = {"roc_auc": json.load(f)["roc_auc"]}

            # ── Step 4: SHAP ───────────────────────────────────
            if not args.skip_shap:
                step_num = 4 if not args.skip_train else 3
                with PipelineStep("SHAP Explainability", step_num, total) as s:
                    results["shap"] = step_shap()
            else:
                print("\n  Skipping SHAP (--skip-shap).")

            # ── Step 5: Validate ───────────────────────────────
            step_num = total
            with PipelineStep("Pipeline Validation", step_num, total) as step:
                results["validate"] = step_validate()

        # ── Log run ────────────────────────────────────────────
        flat_results = {
            "row_count":  results.get("ingest", {}).get("rows"),
            "churn_rate": results.get("ingest", {}).get("churn_rate"),
            "roc_auc":    results.get("train",  {}).get("roc_auc"),
        }
        step_log_run(flat_results)

        # ── Final summary ──────────────────────────────────────
        elapsed = time.time() - t_start
        print("\n" + "="*55)
        print("  PIPELINE COMPLETE ✓")
        print("="*55)
        print(f"  Total time   : {elapsed:.0f}s ({elapsed/60:.1f} min)")

        if "ingest" in results:
            print(f"  Rows         : {results['ingest']['rows']:,}")
            print(f"  Churn rate   : {results['ingest']['churn_rate']:.1%}")
        if "train" in results:
            print(f"  ROC-AUC      : {results['train']['roc_auc']}")
        if "shap" in results:
            print(f"  SHAP plots   : {results['shap']['plots_generated']}")

        print("\n  What to do next:")
        print("  → API:       uvicorn api.main:app --reload --port 8000")
        print("  → Dashboard: streamlit run dashboard/app.py")
        print("="*55 + "\n")

    except Exception as e:
        elapsed = time.time() - t_start
        print("\n" + "="*55)
        print("  PIPELINE FAILED ✗")
        print("="*55)
        print(f"  Error: {e}")
        print(f"  After: {elapsed:.0f}s")
        print("  Check the logs above for details.")
        print("="*55 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    run_full_pipeline()