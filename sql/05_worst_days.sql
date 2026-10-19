-- Five worst operating days (lowest on-time %) with a 7-day rolling on-time trend
WITH d AS (
  SELECT flight_date, flights, otp_pct, cancel_pct,
         ROUND(AVG(otp_pct) OVER (ORDER BY flight_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS otp_7d_avg
  FROM gold.daily_kpis
)
SELECT * FROM d ORDER BY otp_pct ASC LIMIT 5;
