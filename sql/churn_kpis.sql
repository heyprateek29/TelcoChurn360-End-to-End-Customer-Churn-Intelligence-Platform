-- sql/churn_kpis.sql
-- ─────────────────────────────────────────────────────────────
-- TOP-LEVEL CHURN KPIs
-- Business question: "What is our overall churn health?"
-- Used in: Streamlit dashboard KPI cards
-- ─────────────────────────────────────────────────────────────

SELECT
    COUNT(*)                                        AS total_customers,
    SUM(churn_flag)                                 AS total_churned,
    COUNT(*) - SUM(churn_flag)                      AS total_retained,

    -- Churn rate as a percentage
    ROUND(AVG(churn_flag) * 100, 2)                 AS churn_rate_pct,

    -- Monthly revenue metrics
    ROUND(SUM(monthly_charges), 2)                  AS total_mrr,
    ROUND(AVG(monthly_charges), 2)                  AS avg_monthly_charges,

    -- Revenue lost to churn (monthly)
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END), 2)  AS monthly_revenue_lost,

    -- Revenue at risk as % of total MRR
    ROUND(
        SUM(CASE WHEN churn_flag = 1
            THEN monthly_charges ELSE 0 END)
        / SUM(monthly_charges) * 100, 2
    )                                               AS revenue_at_risk_pct,

    -- Average tenure of churned vs retained (months)
    ROUND(AVG(CASE WHEN churn_flag = 1
              THEN tenure END), 1)                  AS avg_tenure_churned_months,
    ROUND(AVG(CASE WHEN churn_flag = 0
              THEN tenure END), 1)                  AS avg_tenure_retained_months

FROM telco;