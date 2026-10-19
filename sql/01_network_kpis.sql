-- Network-level KPIs by month (operated = not cancelled and not diverted)
SELECT
    month,
    SUM(flights)                                              AS flights,
    ROUND(100.0 * SUM(cancelled) / SUM(flights), 2)           AS cancel_pct,
    ROUND(100.0 * (1 - SUM(arr_delayed_15) / SUM(operated)), 2) AS on_time_pct,
    ROUND(SUM(avg_arr_delay_min * operated) / SUM(operated), 2) AS avg_arr_delay_min
FROM gold.carrier_monthly
GROUP BY month
ORDER BY month;
