-- =========================================================
-- RETENTION ANALYSIS
-- =========================================================

SELECT
    'payment_method' AS dimension,
    payment_method AS dimension_value,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(AVG(churn_flag) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges

FROM telco

GROUP BY payment_method


UNION ALL


SELECT
    'paperless_billing' AS dimension,
    paperless_billing AS dimension_value,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(AVG(churn_flag) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges

FROM telco

GROUP BY paperless_billing


UNION ALL


SELECT
    'senior_citizen' AS dimension,
    senior_citizen AS dimension_value,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(AVG(churn_flag) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges

FROM telco

GROUP BY senior_citizen


UNION ALL


SELECT
    'gender' AS dimension,
    gender AS dimension_value,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(AVG(churn_flag) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges

FROM telco

GROUP BY gender


UNION ALL


SELECT
    'dependents' AS dimension,
    dependents AS dimension_value,
    COUNT(*) AS customers,
    SUM(churn_flag) AS churned,
    ROUND(AVG(churn_flag) * 100, 1) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges

FROM telco

GROUP BY dependents


ORDER BY churn_rate_pct DESC;