-- Share of delay minutes by root cause (BTS cause codes), whole period
WITH s AS (
  SELECT SUM(carrier_delay_min) c, SUM(weather_delay_min) w, SUM(nas_delay_min) n,
         SUM(security_delay_min) sec, SUM(late_aircraft_delay_min) la FROM gold.delay_causes
)
SELECT cause, ROUND(100.0 * minutes / (c + w + n + sec + la), 2) AS pct_of_delay_minutes
FROM s, (VALUES ('carrier', 1), ('weather', 2), ('nas', 3), ('security', 4), ('late_aircraft', 5)) v(cause, k),
LATERAL (SELECT CASE k WHEN 1 THEN c WHEN 2 THEN w WHEN 3 THEN n WHEN 4 THEN sec ELSE la END AS minutes)
ORDER BY pct_of_delay_minutes DESC;
