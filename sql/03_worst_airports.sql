-- Busiest-50 origin airports ranked by departure-delay share
WITH totals AS (
    SELECT airport, state, SUM(flights) AS flights, SUM(operated) AS operated,
           SUM(arr_delayed_15) AS delayed, SUM(cancelled) AS cancelled
    FROM gold.airport_daily GROUP BY airport, state
), busiest AS (SELECT * FROM totals ORDER BY flights DESC LIMIT 50)
SELECT airport, state, flights,
       ROUND(100.0 * delayed / operated, 2) AS delayed_pct,
       ROUND(100.0 * cancelled / flights, 2) AS cancel_pct
FROM busiest ORDER BY delayed_pct DESC LIMIT 10;
