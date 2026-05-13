"""
dashboard/app.py
----------------
Phase 9: TelcoChurn360 Streamlit Dashboard

Multi-page dashboard with:
  Page 1 — Overview KPIs
  Page 2 — Churn Analysis (by segment, contract, service)
  Page 3 — SHAP Churn Drivers
  Page 4 — Revenue Risk
  Page 5 — Customer Segments
  Page 6 — Live Prediction (calls FastAPI)

Run:
    streamlit run dashboard/app.py

The dashboard reads from:
  - data/processed/telco_cleaned.csv
  - data/processed/sql_outputs/*.csv
  - models/model_metrics.json
  - shap_outputs/shap_insights.json
  - shap_outputs/*.png
"""

import streamlit as st

# ── Page config — must be the very first Streamlit call ───────
st.set_page_config(
    page_title = "TelcoChurn360",
    page_icon  = "📡",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

CLEAN_CSV     = "data/processed/telco_cleaned.csv"
SQL_DIR       = "data/processed/sql_outputs"
METRICS_JSON  = "models/model_metrics.json"
INSIGHTS_JSON = "shap_outputs/shap_insights.json"
SHAP_DIR      = "shap_outputs"


# ─────────────────────────────────────────────
# DATA LOADERS (cached for performance)
# ─────────────────────────────────────────────

@st.cache_data
def load_customers() -> pd.DataFrame:
    return pd.read_csv(CLEAN_CSV)

@st.cache_data
def load_sql(filename: str) -> pd.DataFrame:
    path = os.path.join(SQL_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_metrics() -> dict:
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_insights() -> dict:
    if os.path.exists(INSIGHTS_JSON):
        with open(INSIGHTS_JSON) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# STYLING HELPERS
# ─────────────────────────────────────────────

BLUE    = "#2563EB"
RED     = "#DC2626"
GREEN   = "#16A34A"
AMBER   = "#D97706"
GRAY    = "#6B7280"

def kpi_card(label: str, value: str, delta: str = "",
             colour: str = BLUE) -> None:
    """Render a styled KPI metric card."""
    st.markdown(f"""
    <div style="
        background: #F9FAFB;
        border-left: 4px solid {colour};
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 4px;
    ">
        <div style="font-size:12px; color:#6B7280; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.05em;">
            {label}
        </div>
        <div style="font-size:28px; font-weight:700; color:#111827;
                    margin-top:4px;">
            {value}
        </div>
        <div style="font-size:12px; color:{colour}; margin-top:2px;">
            {delta}
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<p style='color:{GRAY}; margin-top:-12px;'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("---")


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 📡 TelcoChurn360")
        st.markdown("*Analytics Intelligence Platform*")
        st.markdown("---")

        page = st.radio(
            "Navigate to:",
            options=[
                "🏠 Overview",
                "📊 Churn Analysis",
                "🔍 Churn Drivers (SHAP)",
                "💰 Revenue Risk",
                "👥 Customer Segments",
                "🤖 Live Prediction",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Data**")

        df = load_customers()
        if not df.empty:
            st.caption(f"Customers: {len(df):,}")
            st.caption(f"Churn rate: {df['churn_flag'].mean():.1%}")
            st.caption(f"Features: {len(df.columns)}")

        metrics = load_metrics()
        if metrics:
            st.markdown("**Model**")
            st.caption(f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
            st.caption(f"F1 (churn): {metrics.get('f1_churn', 'N/A')}")

        st.markdown("---")
        st.caption("Phase 9 — Streamlit Dashboard")
        st.caption("TelcoChurn360 Portfolio Project")

    return page


# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW KPIs
# ─────────────────────────────────────────────

def page_overview():
    st.title("🏠 Overview — Churn Intelligence Summary")

    df      = load_customers()
    metrics = load_metrics()
    kpis    = load_sql("kpis.csv")

    if df.empty:
        st.error("Data not found. Run `python src/ingest.py` first.")
        return

    # ── KPI Row ────────────────────────────────────────────────
    section_header("Key Performance Indicators",
                   "Top-level churn health metrics")

    col1, col2, col3, col4 = st.columns(4)

    total       = len(df)
    churned     = int(df["churn_flag"].sum())
    churn_rate  = df["churn_flag"].mean()
    mrr         = df["monthly_charges"].sum()
    rev_lost    = df.loc[df["churn_flag"]==1, "monthly_charges"].sum()

    with col1:
        kpi_card("Total Customers", f"{total:,}",
                 "IBM Telco dataset", BLUE)
    with col2:
        kpi_card("Churn Rate", f"{churn_rate:.1%}",
                 f"{churned:,} customers lost", RED)
    with col3:
        kpi_card("Monthly Revenue Lost", f"${rev_lost:,.0f}",
                 f"${rev_lost*12:,.0f} annualised", AMBER)
    with col4:
        roc = metrics.get("roc_auc", "N/A")
        kpi_card("Model ROC-AUC", str(roc),
                 "Random Forest (300 trees)", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Second KPI row ─────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_tenure_c = df.loc[df["churn_flag"]==1, "tenure"].mean()
        avg_tenure_r = df.loc[df["churn_flag"]==0, "tenure"].mean()
        kpi_card("Avg Tenure (Churned)", f"{avg_tenure_c:.0f} mo",
                 f"vs {avg_tenure_r:.0f} mo retained", GRAY)
    with col2:
        kpi_card("Revenue at Risk",
                 f"{rev_lost/mrr:.1%} of MRR",
                 f"Total MRR: ${mrr:,.0f}", RED)
    with col3:
        avg_svc_c = df.loc[df["churn_flag"]==1, "num_services"].mean()
        avg_svc_r = df.loc[df["churn_flag"]==0, "num_services"].mean()
        kpi_card("Avg Services (Churned)", f"{avg_svc_c:.1f}",
                 f"vs {avg_svc_r:.1f} retained", GRAY)
    with col4:
        f1 = metrics.get("f1_churn", "N/A")
        kpi_card("F1-Score (Churn)", str(f1),
                 "Precision/recall balance", GREEN)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Churn by tenure chart ──────────────────────────────────
    section_header("Churn Rate by Customer Lifecycle Stage")

    col1, col2 = st.columns([3, 2])

    with col1:
        tenure_df = load_sql("churn_by_tenure.csv")
        if not tenure_df.empty:
            order = ["New","Growing","Established","Loyal","Champion"]
            tenure_df["tenure_segment"] = pd.Categorical(
                tenure_df["tenure_segment"], categories=order, ordered=True
            )
            tenure_df = tenure_df.sort_values("tenure_segment")

            fig = go.Figure()
            fig.add_bar(
                x=tenure_df["tenure_segment"],
                y=tenure_df["churn_rate_pct"],
                marker_color=[RED, AMBER, AMBER, GREEN, GREEN],
                text=tenure_df["churn_rate_pct"].apply(lambda x: f"{x:.1f}%"),
                textposition="outside",
            )
            fig.update_layout(
                title="Churn Rate by Tenure Segment",
                yaxis_title="Churn Rate (%)",
                xaxis_title="Customer Lifecycle Stage",
                showlegend=False,
                height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Key insight**")
        st.info(
            "New customers (0–6 months) churn at over 50% — "
            "nearly 10× the rate of Champion customers. "
            "The first 6 months are the highest-risk window."
        )
        st.markdown("**Recommended actions**")
        st.markdown("""
        - 🎯 **New**: Dedicated onboarding program
        - 📞 **Growing**: Proactive check-in call at month 6
        - 📦 **Established**: Bundle upsell offers
        - 🏆 **Loyal/Champion**: Loyalty rewards + referral program
        """)


# ─────────────────────────────────────────────
# PAGE 2: CHURN ANALYSIS
# ─────────────────────────────────────────────

def page_churn_analysis():
    st.title("📊 Churn Analysis")

    df = load_customers()
    if df.empty:
        st.error("Data not found. Run `python src/ingest.py` first.")
        return

    tab1, tab2, tab3 = st.tabs(
        ["By Contract", "By Service", "By Demographics"]
    )

    # ── Tab 1: Contract ────────────────────────────────────────
    with tab1:
        section_header("Churn by Contract Type",
                       "Contract length is the strongest retention lever")

        contract_df = load_sql("churn_by_contract.csv")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                contract_df,
                x="contract", y="churn_rate_pct",
                color="churn_rate_pct",
                color_continuous_scale=["#16A34A","#D97706","#DC2626"],
                text="churn_rate_pct",
                title="Churn Rate by Contract Type",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, height=380,
                              yaxis_title="Churn Rate (%)",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                contract_df,
                x="contract",
                y=["total_mrr","revenue_at_risk"],
                barmode="overlay",
                title="MRR vs Revenue at Risk by Contract",
                color_discrete_map={
                    "total_mrr":     BLUE,
                    "revenue_at_risk": RED,
                },
            )
            fig.update_layout(height=380, yaxis_title="Monthly Revenue ($)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            contract_df[[
                "contract","total_customers","churned",
                "churn_rate_pct","avg_monthly_charges",
                "revenue_at_risk","avg_tenure_months"
            ]].rename(columns={
                "contract":             "Contract",
                "total_customers":      "Customers",
                "churned":              "Churned",
                "churn_rate_pct":       "Churn %",
                "avg_monthly_charges":  "Avg Monthly $",
                "revenue_at_risk":      "Rev at Risk $",
                "avg_tenure_months":    "Avg Tenure (mo)",
            }),
            use_container_width=True, hide_index=True
        )

    # ── Tab 2: Services ────────────────────────────────────────
    with tab2:
        section_header("Service Impact on Churn",
                       "Negative lift = service reduces churn (protective)")

        svc_df = load_sql("service_usage.csv")

        if not svc_df.empty:
            svc_df = svc_df.sort_values("churn_lift_pct")
            svc_df["colour"] = svc_df["churn_lift_pct"].apply(
                lambda x: GREEN if x < 0 else RED
            )

            fig = go.Figure(go.Bar(
                x=svc_df["churn_lift_pct"],
                y=svc_df["service"],
                orientation="h",
                marker_color=svc_df["colour"],
                text=svc_df["churn_lift_pct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
            ))
            fig.add_vline(x=0, line_color="gray", line_dash="dash")
            fig.update_layout(
                title="Churn Lift by Service (negative = protective)",
                xaxis_title="Churn Rate Difference (%)",
                height=420,
                margin=dict(l=160),
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.success("**Most protective services (churn lift < 0)**")
                protective = svc_df[svc_df["churn_lift_pct"] < 0]
                for _, row in protective.iterrows():
                    st.write(f"✅ {row['service']}: {row['churn_lift_pct']:+.1f}%")
            with col2:
                st.error("**Services that correlate with higher churn**")
                risky = svc_df[svc_df["churn_lift_pct"] > 0]
                for _, row in risky.iterrows():
                    st.write(f"⚠️ {row['service']}: {row['churn_lift_pct']:+.1f}%")

    # ── Tab 3: Demographics ────────────────────────────────────
    with tab3:
        section_header("Churn by Demographics",
                       "Understanding which customer profiles churn most")

        col1, col2 = st.columns(2)

        with col1:
            senior = df.groupby("senior_citizen")["churn_flag"].agg(
                ["mean","count"]).reset_index()
            senior.columns = ["senior_citizen","churn_rate","count"]
            fig = px.bar(senior, x="senior_citizen", y="churn_rate",
                         color="senior_citizen",
                         color_discrete_map={"Yes": RED, "No": BLUE},
                         title="Churn Rate: Senior vs Non-Senior",
                         text=senior["churn_rate"].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                              height=320)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            dep = df.groupby("dependents")["churn_flag"].agg(
                ["mean","count"]).reset_index()
            dep.columns = ["dependents","churn_rate","count"]
            fig = px.bar(dep, x="dependents", y="churn_rate",
                         color="dependents",
                         color_discrete_map={"Yes": GREEN, "No": RED},
                         title="Churn Rate: Has Dependents",
                         text=dep["churn_rate"].apply(lambda x: f"{x:.1%}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                              height=320)
            st.plotly_chart(fig, use_container_width=True)

        # Churn by payment method
        pay = df.groupby("payment_method")["churn_flag"].agg(
            ["mean","count"]).reset_index()
        pay.columns = ["payment_method","churn_rate","count"]
        pay = pay.sort_values("churn_rate", ascending=False)

        fig = px.bar(pay, x="payment_method", y="churn_rate",
                     color="churn_rate",
                     color_continuous_scale=["#16A34A","#D97706","#DC2626"],
                     title="Churn Rate by Payment Method",
                     text=pay["churn_rate"].apply(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%",
                          coloraxis_showscale=False, height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "**Insight:** Electronic check users churn at nearly 2× the rate "
            "of automatic payment users. Auto-pay signup could be a meaningful "
            "retention intervention."
        )


# ─────────────────────────────────────────────
# PAGE 3: SHAP DRIVERS
# ─────────────────────────────────────────────

def page_shap():
    st.title("🔍 Churn Drivers — SHAP Explainability")

    insights = load_insights()
    metrics  = load_metrics()

    if not insights:
        st.warning(
            "SHAP outputs not found. "
            "Run `python src/explain.py` first."
        )
        return

    # ── Insight cards ──────────────────────────────────────────
    section_header("Top Churn Drivers",
                   "What the ML model says drives churn risk")

    drivers = insights.get("top_churn_drivers", [])[:5]
    cols    = st.columns(5)
    for i, (col, d) in enumerate(zip(cols, drivers)):
        with col:
            arrow = "▲" if d["direction"] == "increases" else "▼"
            colour = RED if d["direction"] == "increases" else GREEN
            feat_short = d["feature"].replace("_"," ").title()[:20]
            kpi_card(
                feat_short,
                f"{d['impact']:.4f}",
                f"{arrow} {d['direction']} churn",
                colour,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP plots ─────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "Summary Plot", "Bar Chart", "Waterfall", "Dependence"
    ])

    plots = {
        "Summary Plot":  "shap_summary.png",
        "Bar Chart":     "shap_bar.png",
        "Waterfall":     "shap_waterfall.png",
        "Dependence":    "shap_dependence_tenure.png",
    }

    explanations = {
        "Summary Plot":
            "Each dot is one customer. "
            "**Red = high feature value, Blue = low.** "
            "Right side = pushes toward churn, Left = pushes away from churn. "
            "Features are ordered by overall importance (top = most important).",
        "Bar Chart":
            "Average absolute SHAP value per feature across all customers. "
            "This is the **executive version** — which levers matter most on average. "
            "Red bars have above-median impact.",
        "Waterfall":
            "Explains **one specific high-risk customer's prediction** step by step. "
            "Starting from the base rate, each bar shows a feature's contribution. "
            "Red bars push toward churn, blue bars push away.",
        "Dependence":
            "Shows how tenure's churn effect changes with its value. "
            "**Colour = contract risk** (red = month-to-month, blue = two-year). "
            "Bottom-right cluster = long tenure, locked-in contracts = lowest risk.",
    }

    for tab, (name, fname) in zip([tab1,tab2,tab3,tab4], plots.items()):
        with tab:
            path = os.path.join(SHAP_DIR, fname)
            if os.path.exists(path):
                st.image(path, use_column_width=True)
                st.info(explanations[name])
            else:
                st.warning(f"Plot not found: {path}. Run `python src/explain.py`.")

    # ── Plain English insights ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Business Insights",
                   "What these SHAP values mean in plain English")

    plain = insights.get("plain_english_summary", [])
    for i, msg in enumerate(plain, 1):
        with st.expander(f"Insight {i} — {msg[:60]}..."):
            st.write(msg)


# ─────────────────────────────────────────────
# PAGE 4: REVENUE RISK
# ─────────────────────────────────────────────

def page_revenue_risk():
    st.title("💰 Revenue Risk Analysis")

    df = load_customers()
    if df.empty:
        st.error("Data not found.")
        return

    # ── Top-level revenue KPIs ─────────────────────────────────
    section_header("Revenue Impact of Churn")

    total_mrr  = df["monthly_charges"].sum()
    lost_mrr   = df.loc[df["churn_flag"]==1,"monthly_charges"].sum()
    lost_annual= lost_mrr * 12

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total MRR",         f"${total_mrr:,.0f}", "Monthly recurring revenue", BLUE)
    with col2:
        kpi_card("Lost MRR (monthly)", f"${lost_mrr:,.0f}", "From churned customers", RED)
    with col3:
        kpi_card("Lost ARR",          f"${lost_annual:,.0f}", "Annualised revenue lost", RED)
    with col4:
        kpi_card("Revenue at Risk",   f"{lost_mrr/total_mrr:.1%}", "Share of MRR lost to churn", AMBER)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Revenue by segment chart ───────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_header("Revenue Lost by Tenure Segment")
        seg_rev = df.groupby("tenure_segment").agg(
            total_mrr=("monthly_charges","sum"),
            lost_mrr=("monthly_charges", lambda x:
                      x[df.loc[x.index,"churn_flag"]==1].sum()),
            customers=("customer_id","count"),
            churned=("churn_flag","sum"),
        ).reset_index()

        order = ["New","Growing","Established","Loyal","Champion"]
        seg_rev["tenure_segment"] = pd.Categorical(
            seg_rev["tenure_segment"], categories=order, ordered=True
        )
        seg_rev = seg_rev.sort_values("tenure_segment")

        fig = go.Figure()
        fig.add_bar(name="Total MRR",  x=seg_rev["tenure_segment"],
                    y=seg_rev["total_mrr"], marker_color=BLUE, opacity=0.4)
        fig.add_bar(name="Lost MRR",   x=seg_rev["tenure_segment"],
                    y=seg_rev["lost_mrr"],  marker_color=RED)
        fig.update_layout(barmode="overlay", height=360,
                          yaxis_title="Monthly Revenue ($)",
                          title="MRR vs Revenue Lost by Lifecycle Stage")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_header("Revenue Lost by Contract Type")
        con_rev = df.groupby("contract").agg(
            total_mrr=("monthly_charges","sum"),
            lost_mrr=("monthly_charges", lambda x:
                      x[df.loc[x.index,"churn_flag"]==1].sum()),
        ).reset_index()
        con_rev["retained_mrr"] = con_rev["total_mrr"] - con_rev["lost_mrr"]

        fig = px.pie(
            con_rev, names="contract", values="lost_mrr",
            color_discrete_sequence=[RED, AMBER, GREEN],
            title="Share of Lost MRR by Contract Type",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── Revenue waterfall ──────────────────────────────────────
    section_header("Revenue Loss Breakdown",
                   "How churn erodes monthly revenue across segments")

    rev_risk = load_sql("revenue_at_risk.csv")
    if not rev_risk.empty:
        top20 = rev_risk.head(10)
        fig = px.bar(
            top20,
            x="revenue_lost_monthly",
            y=top20.apply(
                lambda r: f"{r['tenure_segment']} / {r['contract']}", axis=1
            ),
            orientation="h",
            color="churn_rate_pct",
            color_continuous_scale=["#16A34A","#D97706","#DC2626"],
            title="Top 10 Revenue-Loss Segments (Monthly)",
            text="revenue_lost_monthly",
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(height=420, xaxis_title="Monthly Revenue Lost ($)",
                          yaxis_title="Segment",
                          coloraxis_colorbar_title="Churn %")
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"**Bottom line:** The top 3 segments alone account for "
            f"${top20['revenue_lost_monthly'].head(3).sum():,.0f}/month "
            f"in lost revenue. These are your highest-priority retention targets."
        )


# ─────────────────────────────────────────────
# PAGE 5: CUSTOMER SEGMENTS
# ─────────────────────────────────────────────

def page_segments():
    st.title("👥 Customer Segmentation")

    df = load_customers()
    if df.empty:
        st.error("Data not found.")
        return

    section_header("RFM-style Customer Segments",
                   "Customers grouped by value and churn risk for campaign targeting")

    seg_df = load_sql("customer_segments.csv")

    if not seg_df.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.scatter(
                seg_df,
                x="avg_monthly_charges",
                y="actual_churn_rate_pct",
                size="customers",
                color="customer_segment",
                hover_name="customer_segment",
                hover_data=["customers","revenue_lost","avg_services"],
                title="Customer Segments — Value vs Churn Risk",
                labels={
                    "avg_monthly_charges":   "Avg Monthly Charges ($)",
                    "actual_churn_rate_pct": "Actual Churn Rate (%)",
                    "customer_segment":      "Segment",
                },
                size_max=60,
            )
            fig.update_layout(height=440)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Segment Descriptions**")
            segment_desc = {
                "Champions":      ("🏆", GREEN,  "High value, low risk. Reward & protect."),
                "At Risk":        ("🚨", RED,    "High value, high risk. Urgent action."),
                "Needs Attention":("⚠️", AMBER,  "High value, medium risk. Nurture."),
                "Promising":      ("📈", BLUE,   "Low value, medium risk. Develop."),
                "Sleepers":       ("💤", GRAY,   "Low value, low risk. Upsell target."),
                "Lost Cause":     ("❌", RED,    "Low value, high risk. Low priority."),
            }
            for seg, (icon, colour, desc) in segment_desc.items():
                row = seg_df[seg_df["customer_segment"] == seg]
                n   = int(row["customers"].iloc[0]) if not row.empty else 0
                st.markdown(
                    f"**{icon} {seg}** ({n:,} customers)  \n"
                    f"<span style='color:{colour}'>{desc}</span>",
                    unsafe_allow_html=True
                )
                st.markdown("")

    # ── Segment table ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Segment Details")

    if not seg_df.empty:
        display_cols = [
            "customer_segment","customers","actual_churn_rate_pct",
            "avg_monthly_charges","total_segment_mrr","revenue_lost","avg_services"
        ]
        available = [c for c in display_cols if c in seg_df.columns]

        styled = seg_df[available].rename(columns={
            "customer_segment":      "Segment",
            "customers":             "Customers",
            "actual_churn_rate_pct": "Churn %",
            "avg_monthly_charges":   "Avg MRR $",
            "total_segment_mrr":     "Segment MRR $",
            "revenue_lost":          "Revenue Lost $",
            "avg_services":          "Avg Services",
        }).sort_values("Revenue Lost $", ascending=False)

        st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PAGE 6: LIVE PREDICTION
# ─────────────────────────────────────────────

def page_live_prediction():
    st.title("🤖 Live Churn Prediction")
    st.markdown(
        "Enter customer details below. The form will call the trained model "
        "directly (no API needed) and return an instant churn prediction."
    )

    # ── Load model directly (works without FastAPI running) ────
    try:
        import joblib
        import numpy as np

        pipeline     = joblib.load("models/churn_model.pkl")
        feature_meta = joblib.load("models/feature_meta.pkl")
        model_ready  = True
    except FileNotFoundError:
        model_ready = False
        st.warning("Model not found. Run `python src/train.py` first.")

    if not model_ready:
        return

    # ── Input form ─────────────────────────────────────────────
    section_header("Customer Profile", "Fill in the customer details")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Account Info**")
            tenure   = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract",
                ["Month-to-month","One year","Two year"])
            payment  = st.selectbox("Payment Method", [
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ])
            paperless = st.selectbox("Paperless Billing", ["Yes","No"])

        with col2:
            st.markdown("**Services**")
            internet = st.selectbox("Internet Service",
                                    ["Fiber optic","DSL","No"])
            phone    = st.selectbox("Phone Service", ["Yes","No"])
            multi    = st.selectbox("Multiple Lines", ["Yes","No"])
            security = st.selectbox("Online Security", ["Yes","No"])
            backup   = st.selectbox("Online Backup",   ["Yes","No"])
            device   = st.selectbox("Device Protection",["Yes","No"])
            tech     = st.selectbox("Tech Support",    ["Yes","No"])
            tv       = st.selectbox("Streaming TV",    ["Yes","No"])
            movies   = st.selectbox("Streaming Movies",["Yes","No"])

        with col3:
            st.markdown("**Demographics**")
            gender   = st.selectbox("Gender",         ["Male","Female"])
            senior   = st.selectbox("Senior Citizen", ["No","Yes"])
            partner  = st.selectbox("Has Partner",    ["No","Yes"])
            depend   = st.selectbox("Has Dependents", ["No","Yes"])
            st.markdown("**Billing**")
            monthly_charges = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, 0.5)
            total_charges   = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=float(monthly_charges * max(tenure, 1)),
                step=10.0
            )

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary")

    if submitted:
        # ── Derive engineered features ────────────────────────
        # These mirror exactly what src/ingest.py computes
        num_services_val = sum([
            phone=="Yes", multi=="Yes", security=="Yes",
            backup=="Yes", device=="Yes", tech=="Yes",
            tv=="Yes", movies=="Yes",
        ])
        support_score_val = sum([tech=="Yes", security=="Yes", device=="Yes"])
        has_fiber_val     = 1 if internet == "Fiber optic" else 0

        contract_risk = {"Month-to-month":3,"One year":2,"Two year":1}[contract]

        avg_spend = total_charges / max(tenure, 1)

        if tenure <= 6:    tseg = "New"
        elif tenure <= 12: tseg = "Growing"
        elif tenure <= 24: tseg = "Established"
        elif tenure <= 48: tseg = "Loyal"
        else:              tseg = "Champion"

        q1, q2, q3 = 35, 65, 89
        if monthly_charges <= q1:      ctier = "Low"
        elif monthly_charges <= q2:    ctier = "Mid"
        elif monthly_charges <= q3:    ctier = "High"
        else:                          ctier = "Premium"

        # Build feature dict
        features = {
            "tenure":               tenure,
            "monthly_charges":      monthly_charges,
            "total_charges":        total_charges,
            "num_services":         num_services_val,
            "avg_monthly_spend":    round(avg_spend, 2),
            "contract_risk_score":  contract_risk,
            "support_score":        support_score_val,
            "gender":               gender,
            "senior_citizen":       senior,
            "partner":              partner,
            "dependents":           depend,
            "phone_service":        phone,
            "multiple_lines":       multi,
            "internet_service":     internet,
            "online_security":      security,
            "online_backup":        backup,
            "device_protection":    device,
            "tech_support":         tech,
            "streaming_tv":         tv,
            "streaming_movies":     movies,
            "contract":             contract,
            "paperless_billing":    paperless,
            "payment_method":       payment,
            "tenure_segment":       tseg,
            "monthly_charges_tier": ctier,
        }

        all_features = feature_meta["all_features"]
        X = pd.DataFrame([features])[all_features]

        # ── Run prediction ────────────────────────────────────
        proba      = pipeline.predict_proba(X)[0]
        churn_prob = float(proba[1])
        churn_pred = int(pipeline.predict(X)[0])

        # ── Display result ────────────────────────────────────
        st.markdown("---")
        st.markdown("### Prediction Result")

        if churn_prob >= 0.70:
            tier, colour, icon = "Critical", RED,   "🚨"
        elif churn_prob >= 0.50:
            tier, colour, icon = "High",     RED,   "⚠️"
        elif churn_prob >= 0.30:
            tier, colour, icon = "Medium",   AMBER, "📋"
        else:
            tier, colour, icon = "Low",      GREEN, "✅"

        col1, col2, col3 = st.columns(3)
        with col1:
            kpi_card("Churn Probability",
                     f"{churn_prob:.1%}", tier, colour)
        with col2:
            kpi_card("Prediction",
                     "Will Churn" if churn_pred else "Will Stay",
                     f"{icon} Risk tier: {tier}", colour)
        with col3:
            kpi_card("Retention Probability",
                     f"{1-churn_prob:.1%}",
                     "Model confidence customer stays", GREEN)

        # ── Gauge chart ───────────────────────────────────────
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = churn_prob * 100,
            title = {"text": "Churn Risk Score", "font": {"size": 18}},
            delta = {"reference": 26.5, "suffix": "%",
                     "decreasing": {"color": GREEN},
                     "increasing": {"color": RED}},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": colour},
                "steps": [
                    {"range": [0,  30], "color": "#D1FAE5"},
                    {"range": [30, 50], "color": "#FEF3C7"},
                    {"range": [50, 70], "color": "#FEE2E2"},
                    {"range": [70,100], "color": "#FECACA"},
                ],
                "threshold": {
                    "line": {"color": "darkred", "width": 4},
                    "thickness": 0.75, "value": 50,
                },
            },
            number = {"suffix": "%"},
        ))
        fig.update_layout(height=300, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # ── Recommendation ────────────────────────────────────
        st.markdown("#### 💡 Recommended Action")
        from api.model_loader import get_recommendation
        rec = get_recommendation(
            churn_probability=churn_prob,
            contract=contract,
            tenure=tenure,
            internet_service=internet,
            monthly_charges=monthly_charges,
            support_score=support_score_val,
            num_services=num_services_val,
        )
        if churn_prob >= 0.5:
            st.error(f"**{rec}**")
        elif churn_prob >= 0.3:
            st.warning(f"**{rec}**")
        else:
            st.success(f"**{rec}**")

        # ── Feature summary ───────────────────────────────────
        with st.expander("View input features used for this prediction"):
            st.json(features)


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────

def main():
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview()
    elif page == "📊 Churn Analysis":
        page_churn_analysis()
    elif page == "🔍 Churn Drivers (SHAP)":
        page_shap()
    elif page == "💰 Revenue Risk":
        page_revenue_risk()
    elif page == "👥 Customer Segments":
        page_segments()
    elif page == "🤖 Live Prediction":
        page_live_prediction()


if __name__ == "__main__":
    main()