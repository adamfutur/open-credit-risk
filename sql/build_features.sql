-- build_features.sql
-- Runs against the raw table loaded into SQLite (see src/load_to_sql.py).
-- Produces a base feature table: renames raw columns to their business meaning
-- and aggregates the three delinquency-count columns into one payment-history
-- signal, before the row set is handed to PySpark for the ratio engineering
-- (DTI, utilization) and the SMOTE step in pandas.

DROP TABLE IF EXISTS credit_base;

CREATE TABLE credit_base AS
SELECT
    id                                          AS client_id,
    "SeriousDlqin2yrs"                          AS defaulted,               -- target
    "RevolvingUtilizationOfUnsecuredLines"      AS raw_credit_utilization,  -- pre-existing ratio in source data
    age                                         AS age,
    "DebtRatio"                                 AS raw_debt_ratio,          -- pre-existing DTI-like ratio in source data
    "MonthlyIncome"                             AS monthly_income,
    "NumberOfOpenCreditLinesAndLoans"           AS open_credit_lines,
    "NumberRealEstateLoansOrLines"              AS real_estate_lines,
    "NumberOfDependents"                        AS dependents,
    -- Aggregated payment-history signal: total late-payment incidents across
    -- all three severity buckets (30-59, 60-89, 90+ days past due).
    ("NumberOfTime30-59DaysPastDueNotWorse"
        + "NumberOfTime60-89DaysPastDueNotWorse"
        + "NumberOfTimes90DaysLate")            AS total_late_payment_incidents,
    "NumberOfTimes90DaysLate"                   AS severe_delinquencies_90d
FROM credit_raw;

-- Sanity check query (run manually): class balance should be ~6-7% default.
-- SELECT defaulted, COUNT(*), COUNT(*) * 1.0 / (SELECT COUNT(*) FROM credit_base)
-- FROM credit_base GROUP BY defaulted;
