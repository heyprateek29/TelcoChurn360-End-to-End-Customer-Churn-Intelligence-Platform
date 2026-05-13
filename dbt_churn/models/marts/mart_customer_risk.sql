-- dbt_churn/models/marts/mart_customer_risk.sql
-- ─────────────────────────────────────────────────────────────
-- MART: mart_customer_risk
--
-- Purpose: One row per customer with a rule-based risk score
--          and recommended retention action.
--          Powers the "Customer Risk Table" in the dashboard
--          and can be exported for CRM campaigns.
--
-- Materialised as: TABLE
-- Depends on:      stg_customers
-- ─────────────────────────────────────────────────────────────

with base as (

    select * from {{ ref('stg_customers') }}

),

risk_scored as (

    select
        customer_id,
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure,
        tenure_segment,
        contract,
        contract_risk_score,
        internet_service,
        monthly_charges,
        total_charges,
        num_services,
        support_score,
        has_fiber,
        payment_method,
        churn_flag,
        churn_label,

        -- ── Rule-based risk score (0–100) ─────────────────────
        -- Combines the strongest predictors from the ML model
        -- and the SQL analysis. Useful for CRM segmentation
        -- without needing to run the ML model.
        least(100, greatest(0,
              -- Contract risk (0, 15, or 30 points)
              (contract_risk_score - 1) * 15

              -- Tenure risk: new customers score higher
            + case
                when tenure <= 6  then 30
                when tenure <= 12 then 20
                when tenure <= 24 then 10
                else 0
              end

              -- Fiber internet adds risk
            + has_fiber * 10

              -- Monthly charge tier adds risk
            + case
                when monthly_charges > 89 then 15
                when monthly_charges > 65 then 8
                else 0
              end

              -- Support services reduce risk
            - support_score * 5

              -- More bundled services reduce risk
            - least(num_services * 3, 15)
        ))                                              as risk_score,

        -- ── Customer value tier ───────────────────────────────
        case
            when monthly_charges > 89 and tenure > 24 then 'High Value'
            when monthly_charges > 65                  then 'Mid Value'
            else                                            'Low Value'
        end                                             as value_tier,

        -- ── Risk category ─────────────────────────────────────
        case
            when contract_risk_score = 3 and tenure <= 12 then 'Critical'
            when contract_risk_score = 3 and tenure <= 24 then 'High'
            when contract_risk_score >= 2 and tenure <= 12 then 'High'
            when contract_risk_score = 3                   then 'Medium'
            else                                                'Low'
        end                                             as risk_category,

        -- ── Recommended retention action ──────────────────────
        case
            when contract_risk_score = 3 and tenure <= 6
                then 'URGENT: Offer annual contract discount + onboarding call'
            when contract_risk_score = 3 and tenure <= 12
                then 'Offer annual contract upgrade with 15% discount'
            when contract_risk_score = 3 and has_fiber = 1
                then 'Proactive tech support outreach — fiber satisfaction check'
            when contract_risk_score = 3 and monthly_charges > 89
                then 'Pricing review — offer loyalty rate or bundle discount'
            when contract_risk_score = 3
                then 'Nurture: send contract upgrade incentive email'
            when support_score = 0 and num_services <= 2
                then 'Upsell: offer tech support + security bundle'
            else
                'Monitor: low risk — standard engagement'
        end                                             as recommended_action

    from base

)

select
    *,
    -- Segment label combining value and risk for campaign targeting
    value_tier || ' / ' || risk_category               as campaign_segment
from risk_scored
order by risk_score desc