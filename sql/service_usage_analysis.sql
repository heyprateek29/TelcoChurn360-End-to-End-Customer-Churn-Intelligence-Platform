-- sql/service_usage_analysis.sql
-- ─────────────────────────────────────────────────────────────
-- SERVICE USAGE & CHURN CORRELATION
-- Business question: "Which services reduce churn the most?
--                    Where should we push upsells?"
-- Insight: Customers with tech support / security churn less.
-- Used in: Streamlit service impact chart
-- ─────────────────────────────────────────────────────────────

-- Part A: Adoption and churn rate per service
WITH service_flags AS (
    SELECT
        customer_id,
        churn_flag,
        monthly_charges,

        -- Flag each service as 1 if subscribed
        CASE WHEN phone_service    = 'Yes' THEN 1 ELSE 0 END AS has_phone,
        CASE WHEN multiple_lines   = 'Yes' THEN 1 ELSE 0 END AS has_multi_lines,
        CASE WHEN internet_service != 'No'  THEN 1 ELSE 0 END AS has_internet,
        CASE WHEN internet_service = 'Fiber optic' THEN 1 ELSE 0 END AS has_fiber,
        CASE WHEN online_security  = 'Yes' THEN 1 ELSE 0 END AS has_security,
        CASE WHEN online_backup    = 'Yes' THEN 1 ELSE 0 END AS has_backup,
        CASE WHEN device_protection= 'Yes' THEN 1 ELSE 0 END AS has_device_protect,
        CASE WHEN tech_support     = 'Yes' THEN 1 ELSE 0 END AS has_tech_support,
        CASE WHEN streaming_tv     = 'Yes' THEN 1 ELSE 0 END AS has_streaming_tv,
        CASE WHEN streaming_movies = 'Yes' THEN 1 ELSE 0 END AS has_streaming_movies

    FROM telco
),

-- Unpivot services into rows for easy aggregation
services_long AS (
    SELECT 'Phone service'     AS service, has_phone            AS subscribed, churn_flag FROM service_flags
    UNION ALL
    SELECT 'Multiple lines',              has_multi_lines,                   churn_flag FROM service_flags
    UNION ALL
    SELECT 'Internet (any)',              has_internet,                      churn_flag FROM service_flags
    UNION ALL
    SELECT 'Fiber optic',                has_fiber,                         churn_flag FROM service_flags
    UNION ALL
    SELECT 'Online security',            has_security,                      churn_flag FROM service_flags
    UNION ALL
    SELECT 'Online backup',              has_backup,                        churn_flag FROM service_flags
    UNION ALL
    SELECT 'Device protection',          has_device_protect,                churn_flag FROM service_flags
    UNION ALL
    SELECT 'Tech support',               has_tech_support,                  churn_flag FROM service_flags
    UNION ALL
    SELECT 'Streaming TV',               has_streaming_tv,                  churn_flag FROM service_flags
    UNION ALL
    SELECT 'Streaming movies',           has_streaming_movies,              churn_flag FROM service_flags
)

SELECT
    service,
    SUM(subscribed)                                         AS subscribers,
    ROUND(AVG(subscribed) * 100, 1)                         AS adoption_rate_pct,

    -- Churn rate AMONG subscribers vs non-subscribers
    ROUND(AVG(CASE WHEN subscribed = 1 THEN churn_flag END) * 100, 1)
                                                            AS churn_rate_with_service,
    ROUND(AVG(CASE WHEN subscribed = 0 THEN churn_flag END) * 100, 1)
                                                            AS churn_rate_without_service,

    -- Churn lift: positive = service increases churn, negative = reduces it
    ROUND(
        AVG(CASE WHEN subscribed = 1 THEN churn_flag END) * 100
      - AVG(CASE WHEN subscribed = 0 THEN churn_flag END) * 100
    , 1)                                                    AS churn_lift_pct

FROM services_long
GROUP BY service
ORDER BY churn_lift_pct ASC;   -- Most protective services first