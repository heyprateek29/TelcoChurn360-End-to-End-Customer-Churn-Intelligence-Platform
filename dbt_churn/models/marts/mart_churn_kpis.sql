-- dbt_churn/models/marts/mart_churn_kpis.sql
-- ─────────────────────────────────────────────────────────────
-- MART: mart_churn_kpis
--
-- Purpose: Top-level business KPIs consumed by the Streamlit
--          dashboard KPI cards section.
--
-- Materialised as: TABLE (fast reads for dashboard)
-- Depends on:      stg_customers
-- ─────────────────────────────────────────────────────────────

with base as (

    select * from {{ ref('stg_customers') }}

),

kpis as (

    select
        -- ── Volume ────────────────────────────────────────────
        count(*)                                        as total_customers,
        sum(churn_flag)                                 as total_churned,
        count(*) - sum(churn_flag)                      as total_retained,

        -- ── Churn rate ────────────────────────────────────────
        round(avg(churn_flag) * 100, 2)                 as churn_rate_pct,
        round((1 - avg(churn_flag)) * 100, 2)           as retention_rate_pct,

        -- ── Revenue ───────────────────────────────────────────
        round(sum(monthly_charges), 2)                  as total_mrr,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,

        -- Monthly revenue lost to churn
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as monthly_revenue_lost,

        -- Annualised churn cost
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ) * 12, 2)                                      as annual_revenue_lost,

        -- Revenue at risk as share of MRR
        round(
            sum(case when churn_flag = 1 then monthly_charges else 0 end)
            / nullif(sum(monthly_charges), 0) * 100
        , 2)                                            as revenue_at_risk_pct,

        -- ── Tenure ────────────────────────────────────────────
        round(avg(tenure), 1)                           as avg_tenure_months,
        round(avg(
            case when churn_flag = 1 then tenure end
        ), 1)                                           as avg_tenure_churned,
        round(avg(
            case when churn_flag = 0 then tenure end
        ), 1)                                           as avg_tenure_retained,

        -- ── Services ──────────────────────────────────────────
        round(avg(num_services), 2)                     as avg_num_services,
        round(avg(
            case when churn_flag = 1 then num_services end
        ), 2)                                           as avg_services_churned,
        round(avg(
            case when churn_flag = 0 then num_services end
        ), 2)                                           as avg_services_retained

    from base

)

select * from kpis