-- Cleaned customer view
CREATE VIEW IF NOT EXISTS customers_clean AS
SELECT
    customerID,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    CASE 
        WHEN TRIM(TotalCharges) = '' THEN NULL
        ELSE CAST(TotalCharges AS REAL)
    END AS TotalCharges,
    Churn
FROM customers_raw;

-- Feature-ready table
CREATE TABLE IF NOT EXISTS customer_features AS
SELECT
    customerID,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS churn,
    tenure,
    MonthlyCharges,
    TotalCharges,
    (TotalCharges / (tenure + 1)) AS charges_per_tenure,
    Contract,
    PaymentMethod,
    InternetService,
    PaperlessBilling,
    SeniorCitizen
FROM customers_clean;
