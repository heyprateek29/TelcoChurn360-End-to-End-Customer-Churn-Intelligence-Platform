-- dbt_churn/models/marts/mart_churn_by_segment.sql
-- ─────────────────────────────────────────────────────────────
-- MART: mart_churn_by_segment
--
-- Purpose: Churn metrics broken out by every major business
--          dimension. Powers the segment analysis charts in
--          the Streamlit dashboard.
--
-- Materialised as: TABLE
-- Depends on:      stg_customers
-- ─────────────────────────────────────────────────────────────

with base as (

    select * from {{ ref('stg_customers') }}

),

-- ── By tenure segment ─────────────────────────────────────────
tenure_segments as (

    select
        'tenure_segment'                                as dimension,
        tenure_segment                                  as dimension_value,

        -- Sort order for charts
        case tenure_segment
            when 'New'         then 1
            when 'Growing'     then 2
            when 'Established' then 3
            when 'Loyal'       then 4
            when 'Champion'    then 5
        end                                             as sort_order,

        count(*)                                        as customers,
        sum(churn_flag)                                 as churned,
        round(avg(churn_flag) * 100, 1)                 as churn_rate_pct,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as revenue_at_risk

    from base
    group by tenure_segment

),

-- ── By contract type ─────────────────────────────────────────
contract_segments as (

    select
        'contract'                                      as dimension,
        contract                                        as dimension_value,
        contract_risk_score                             as sort_order,
        count(*)                                        as customers,
        sum(churn_flag)                                 as churned,
        round(avg(churn_flag) * 100, 1)                 as churn_rate_pct,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as revenue_at_risk

    from base
    group by contract, contract_risk_score

),

-- ── By internet service ───────────────────────────────────────
internet_segments as (

    select
        'internet_service'                              as dimension,
        internet_service                                as dimension_value,
        row_number() over (order by avg(churn_flag) desc) as sort_order,
        count(*)                                        as customers,
        sum(churn_flag)                                 as churned,
        round(avg(churn_flag) * 100, 1)                 as churn_rate_pct,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as revenue_at_risk

    from base
    group by internet_service

),

-- ── By payment method ─────────────────────────────────────────
payment_segments as (

    select
        'payment_method'                                as dimension,
        payment_method                                  as dimension_value,
        row_number() over (order by avg(churn_flag) desc) as sort_order,
        count(*)                                        as customers,
        sum(churn_flag)                                 as churned,
        round(avg(churn_flag) * 100, 1)                 as churn_rate_pct,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as revenue_at_risk

    from base
    group by payment_method

),

-- ── By monthly charges tier ───────────────────────────────────
charges_segments as (

    select
        'monthly_charges_tier'                          as dimension,
        monthly_charges_tier                            as dimension_value,
        case monthly_charges_tier
            when 'Low'     then 1
            when 'Mid'     then 2
            when 'High'    then 3
            when 'Premium' then 4
        end                                             as sort_order,
        count(*)                                        as customers,
        sum(churn_flag)                                 as churned,
        round(avg(churn_flag) * 100, 1)                 as churn_rate_pct,
        round(avg(monthly_charges), 2)                  as avg_monthly_charges,
        round(sum(
            case when churn_flag = 1 then monthly_charges else 0 end
        ), 2)                                           as revenue_at_risk

    from base
    group by monthly_charges_tier

),

-- Union all dimensions into one tidy table
all_segments as (

    select * from tenure_segments
    union all
    select * from contract_segments
    union all
    select * from internet_segments
    union all
    select * from payment_segments
    union all
    select * from charges_segments

)

select * from all_segments
order by dimension, sort_order