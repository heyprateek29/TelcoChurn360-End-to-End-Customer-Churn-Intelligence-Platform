-- sql/customer_segmentation.sql
-- ─────────────────────────────────────────────────────────────
-- CUSTOMER SEGMENTATION (RFM-style)
-- Business question: "How do we group customers by value
--                    and risk for targeted retention campaigns?"
--
-- Segments:
--   High Value + Low Risk   → Champions  (protect & reward)
--   High Value + High Risk  → At Risk    (urgent intervention)
--   Low Value  + High Risk  → Lost Cause (low effort retention)
--   Low Value  + Low Risk   → Sleepers   (upsell opportunity)
-- ─────────────────────────────────────────────────────────────

WITH scored AS (
    SELECT
        customer_id,
        churn_flag,
        tenure,
        monthly_charges,
        total_charges,
        num_services,
        contract,
        internet_service,

        -- Value score: high monthly spend + long tenure = high value
        CASE
            WHEN monthly_charges > 65 AND tenure > 24 THEN 'High Value'
            ELSE 'Low Value'
        END AS value_tier,

        -- Risk score: month-to-month + new tenure = high risk
        CASE
            WHEN contract = 'Month-to-month' AND tenure <= 12 THEN 'High Risk'
            WHEN contract = 'Month-to-month' AND tenure <= 24 THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS risk_tier

    FROM telco
),

segmented AS (
    SELECT *,
        CASE
            WHEN value_tier = 'High Value' AND risk_tier = 'Low Risk'    THEN 'Champions'
            WHEN value_tier = 'High Value' AND risk_tier = 'High Risk'   THEN 'At Risk'
            WHEN value_tier = 'High Value' AND risk_tier = 'Medium Risk' THEN 'Needs Attention'
            WHEN value_tier = 'Low Value'  AND risk_tier = 'High Risk'   THEN 'Lost Cause'
            WHEN value_tier = 'Low Value'  AND risk_tier = 'Medium Risk' THEN 'Promising'
            ELSE 'Sleepers'
        END AS customer_segment
    FROM scored
)

SELECT
    customer_segment,
    value_tier,
    risk_tier,

    COUNT(*)                                            AS customers,
    ROUND(AVG(churn_flag) * 100, 1)                     AS actual_churn_rate_pct,
    SUM(churn_flag)                                     AS actually_churned,

    -- Revenue profile
    ROUND(AVG(monthly_charges), 2)                      AS avg_monthly_charges,
    ROUND(SUM(monthly_charges), 2)                      AS total_segment_mrr,
    ROUND(SUM(CASE WHEN churn_flag = 1
              THEN monthly_charges ELSE 0 END), 2)      AS revenue_lost,

    -- Engagement
    ROUND(AVG(num_services), 1)                         AS avg_services,
    ROUND(AVG(tenure), 1)                               AS avg_tenure_months

FROM segmented
GROUP BY customer_segment, value_tier, risk_tier
ORDER BY total_segment_mrr DESC;