-- sql/churn_by_tenure.sql
-- ─────────────────────────────────────────────────────────────
-- CHURN BY TENURE SEGMENT
-- Business question: "When in the customer lifecycle do we
--                    lose the most customers?"
-- Insight: First 6 months are critical — fix onboarding.
-- Used in: Streamlit lifecycle funnel chart
-- ─────────────────────────────────────────────────────────────

SELECT
    tenure_segment,

    -- Ordering column so dashboard can sort correctly
    CASE tenure_segment
        WHEN 'New'         THEN 1
        WHEN 'Growing'     THEN 2
        WHEN 'Established' THEN 3
        WHEN 'Loyal'       THEN 4
        WHEN 'Champion'    THEN 5
    END                                                 AS segment_order,

    COUNT(*)                                            AS total_customers,
    SUM(churn_flag)                                     AS churned,
    COUNT(*) - SUM(churn_flag)                          AS retained,
    ROUND(AVG(churn_flag) * 100, 1)                     AS churn_rate_pct,

    -- Tenure stats within segment
    MIN(tenure)                                         AS min_tenure_months,
    MAX(tenure)                                         AS max_tenure_months,
    ROUND(AVG(tenure), 1)                               AS avg_tenure_months,

    -- Revenue profile
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2)                      AS segment_mrr,
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END), 2)      AS segment_revenue_at_risk

FROM telco
GROUP BY tenure_segment
ORDER BY segment_order;