-- dbt_churn/models/staging/stg_customers.sql
-- ─────────────────────────────────────────────────────────────
-- STAGING LAYER: stg_customers
--
-- Purpose: Clean, rename, and type-cast the raw source table.
-- The staging layer is the contract between raw data and
-- business logic. Nothing computed here — just shape the data.
--
-- Materialised as: VIEW (always fresh, no storage cost)
-- ─────────────────────────────────────────────────────────────

with source as (

    -- Reference the raw source table declared in _sources.yml
    -- dbt resolves this to the actual table at runtime
    select * from {{ source('telco_raw', 'customers') }}

),

staged as (

    select
        -- ── Identity ──────────────────────────────────────────
        customer_id,

        -- ── Demographics ──────────────────────────────────────
        gender,
        senior_citizen,
        partner,
        dependents,

        -- ── Service subscriptions ─────────────────────────────
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,

        -- ── Contract & billing ────────────────────────────────
        contract,
        paperless_billing,
        payment_method,

        -- ── Financial ─────────────────────────────────────────
        cast(monthly_charges as double) as monthly_charges,
        cast(total_charges   as double) as total_charges,
        cast(tenure          as integer) as tenure,

        -- ── Target variable ───────────────────────────────────
        churn                           as churn_label,   -- 'Yes'/'No'
        cast(churn_flag as integer)     as churn_flag,    -- 1/0

        -- ── Engineered features (from Phase 2) ────────────────
        tenure_segment,
        monthly_charges_tier,
        cast(num_services        as integer) as num_services,
        cast(has_fiber           as integer) as has_fiber,
        cast(contract_risk_score as integer) as contract_risk_score,
        cast(support_score       as integer) as support_score,
        cast(avg_monthly_spend   as double)  as avg_monthly_spend,
        revenue_at_risk

    from source

)

select * from staged