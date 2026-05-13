-- sql/revenue_at_risk.sql
-- ─────────────────────────────────────────────────────────────
-- REVENUE AT RISK BY SEGMENT
-- Business question: "Which customer segments are causing the
--                    most financial damage when they churn?"
-- Used in: Streamlit revenue risk chart
-- ─────────────────────────────────────────────────────────────

SELECT
    tenure_segment,
    contract,
    monthly_charges_tier,

    COUNT(*)                                                AS customers,
    SUM(churn_flag)                                         AS churned,
    ROUND(AVG(churn_flag) * 100, 1)                         AS churn_rate_pct,

    -- Total monthly revenue at risk from this segment
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END), 2)          AS revenue_lost_monthly,

    -- Avg monthly charge of churned customers in segment
    ROUND(AVG(CASE WHEN churn_flag = 1
              THEN monthly_charges END), 2)                 AS avg_churned_charge,

    -- Annualised revenue at risk
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END) * 12, 2)    AS revenue_lost_annual

FROM telco
GROUP BY tenure_segment, contract, monthly_charges_tier
HAVING SUM(churn_flag) > 0
ORDER BY revenue_lost_monthly DESC
LIMIT 20;