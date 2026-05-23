-- CreditPulse Database Schema
-- Run once on first startup (auto-executed via docker-compose init)

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS audit;

-- Raw layer
CREATE TABLE IF NOT EXISTS raw.accounts (
    account_id TEXT PRIMARY KEY,
    account_type TEXT,
    age_days INTEGER,
    avg_monthly_spend FLOAT,
    region TEXT,
    is_high_risk BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.merchants (
    merchant_id TEXT PRIMARY KEY,
    category TEXT,
    base_fraud_rate FLOAT,
    is_foreign BOOLEAN,
    avg_txn_amount FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw.transactions (
    txn_id TEXT PRIMARY KEY,
    account_id TEXT REFERENCES raw.accounts(account_id),
    merchant_id TEXT REFERENCES raw.merchants(merchant_id),
    amount FLOAT NOT NULL,
    merchant_category TEXT,
    is_foreign_merchant BOOLEAN,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    txn_velocity_1h INTEGER,
    amount_vs_avg_ratio FLOAT,
    is_fraud BOOLEAN NOT NULL DEFAULT FALSE,
    fraud_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_txn_account ON raw.transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_created ON raw.transactions(created_at DESC);

-- Mart layer: risk scores
CREATE TABLE IF NOT EXISTS mart.risk_scores (
    id SERIAL PRIMARY KEY,
    txn_id TEXT UNIQUE NOT NULL,
    account_id TEXT,
    merchant_id TEXT,
    fraud_probability FLOAT,
    credit_risk_score FLOAT,
    composite_risk_score FLOAT,
    decision TEXT CHECK (decision IN ('FRAUD', 'REVIEW', 'CLEAR')),
    model_version TEXT DEFAULT 'v1.0',
    scored_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_risk_scores_account ON mart.risk_scores(account_id);
CREATE INDEX IF NOT EXISTS idx_risk_scores_scored_at ON mart.risk_scores(scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_decision ON mart.risk_scores(decision);

-- Mart layer: streaming feature tables
CREATE TABLE IF NOT EXISTS mart.features_account_txn_counts (
    account_id TEXT,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    txn_count_1h INTEGER,
    total_amount_1h FLOAT,
    avg_amount_1h FLOAT,
    PRIMARY KEY (account_id, window_start)
);

CREATE TABLE IF NOT EXISTS mart.features_velocity (
    account_id TEXT,
    window_start TIMESTAMPTZ,
    txn_count_5m INTEGER,
    velocity_score FLOAT,
    PRIMARY KEY (account_id, window_start)
);

CREATE TABLE IF NOT EXISTS mart.features_amount_stats (
    account_id TEXT,
    window_start TIMESTAMPTZ,
    amount_mean_24h FLOAT,
    amount_std_24h FLOAT,
    txn_count_24h INTEGER,
    PRIMARY KEY (account_id, window_start)
);

CREATE TABLE IF NOT EXISTS mart.features_merchant_stats (
    merchant_id TEXT,
    window_start TIMESTAMPTZ,
    merchant_txn_count INTEGER,
    merchant_avg_amount FLOAT,
    PRIMARY KEY (merchant_id, window_start)
);

-- Audit layer (append-only, never UPDATE or DELETE)
CREATE TABLE IF NOT EXISTS audit.model_decisions (
    id SERIAL PRIMARY KEY,
    txn_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    score FLOAT,
    decision TEXT,
    top_features JSONB,
    decided_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decisions_txn ON audit.model_decisions(txn_id);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON audit.model_decisions(decided_at DESC);

CREATE TABLE IF NOT EXISTS audit.explanations (
    id SERIAL PRIMARY KEY,
    txn_id TEXT NOT NULL,
    shap_values JSONB,
    top_features JSONB,
    counterfactuals JSONB,
    anchor JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_explanations_txn ON audit.explanations(txn_id);

CREATE TABLE IF NOT EXISTS audit.drift_reports (
    id SERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    psi_score FLOAT,
    drift_detected BOOLEAN,
    report JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.fairness_reports (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT,
    gate_passed BOOLEAN,
    report JSONB,
    computed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.fairness_overrides (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT,
    overrider TEXT,
    justification TEXT,
    overridden_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit.hitl_overrides (
    id SERIAL PRIMARY KEY,
    txn_id TEXT NOT NULL,
    original_decision TEXT,
    override_decision TEXT,
    analyst_id TEXT,
    justification TEXT,
    overridden_at TIMESTAMPTZ DEFAULT NOW()
);
