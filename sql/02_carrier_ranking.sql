-- Carrier on-time ranking for the whole period (min 10k flights)
SELECT
    carrier,
    SUM(flights) AS flights,
    ROUND(100.0 * (1 - SUM(arr_delayed_15) / SUM(operated)), 2) AS on_time_pct,
    ROUND(100.0 * SUM(cancelled) / SUM(flights), 2) AS cancel_pct,
    RANK() OVER (ORDER BY 1 - SUM(arr_delayed_15) / SUM(operated) DESC) AS otp_rank
FROM gold.carrier_monthly
GROUP BY carrier
HAVING SUM(flights) >= 10000
ORDER BY otp_rank;
