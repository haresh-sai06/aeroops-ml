| Model | Features | CV RMSE (5-fold by engine) | Test RMSE | Test RMSE (uncapped truth) | NASA score | Fit (s) |
|---|---|---|---|---|---|---|
| Baseline (mean RUL) | - | 41.68 ± 2.27 | 41.94 | 43.07 | 33354.5 | 0.0 |
| Ridge regression | rolling | 20.18 ± 1.7 | 19.5 | 20.56 | 756.5 | 0.0 |
| Random Forest | rolling | 17.6 ± 3.25 | 16.74 | 17.98 | 532.7 | 30.1 |
| XGBoost | raw | 17.95 ± 3.35 | 17.57 | 18.77 | 655.1 | 1.1 |
| XGBoost | rolling | 17.31 ± 3.18 | 17.12 | 18.35 | 522.8 | 2.0 |
| LSTM (PyTorch) | 30-cycle windows | - | 14.25 | 15.45 | 389.5 | 81.1 |
