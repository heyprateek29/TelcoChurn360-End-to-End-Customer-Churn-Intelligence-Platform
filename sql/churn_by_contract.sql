-- sql/churn_by_contract.sql
-- ─────────────────────────────────────────────────────────────
-- CHURN BY CONTRACT TYPE
-- Business question: "Does contract length protect us from churn?"
-- Insight: Month-to-month = highest churn. Two-year = lowest.
-- Used in: Streamlit contract analysis chart
-- ─────────────────────────────────────────────────────────────

SELECT
    contract,
    contract_risk_score,

    COUNT(*)                                            AS total_customers,
    SUM(churn_flag)                                     AS churned,
    COUNT(*) - SUM(churn_flag)                          AS retained,
    ROUND(AVG(churn_flag) * 100, 1)                     AS churn_rate_pct,

    -- Revenue metrics per contract type
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2)                      AS total_mrr,
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END), 2)      AS revenue_at_risk,

    -- Average tenure (longer tenure = contract lock-in working)
    ROUND(AVG(tenure), 1)                               AS avg_tenure_months,
    ROUND(AVG(CASE WHEN churn_flag = 1
              THEN tenure END), 1)                      AS avg_tenure_churned

FROM telco
GROUP BY contract, contract_risk_score
ORDER BY contract_risk_score DESC;