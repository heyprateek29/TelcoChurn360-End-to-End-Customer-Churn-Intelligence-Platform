# verify_setup.py
# Run this to confirm all packages installed correctly

print("Checking imports...")

import pandas as pd
print(f"  pandas {pd.__version__} ✓")

import numpy as np
print(f"  numpy {np.__version__} ✓")

import duckdb
print(f"  duckdb {duckdb.__version__} ✓")

import sklearn
print(f"  scikit-learn {sklearn.__version__} ✓")

import shap
print(f"  shap {shap.__version__} ✓")

import fastapi
print(f"  fastapi {fastapi.__version__} ✓")

import streamlit
print(f"  streamlit {streamlit.__version__} ✓")

import plotly
print(f"  plotly {plotly.__version__} ✓")

import dbt
print(f"  dbt ✓")

import os
raw = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if os.path.exists(raw):
    df = pd.read_csv(raw)
    print(f"\n  Dataset found: {len(df)} rows, {len(df.columns)} columns ✓")
else:
    print(f"\n  WARNING: Dataset not found at {raw}")
    print("  Please download from Kaggle and place it there.")

print("\nSetup complete! Ready for Phase 2.")