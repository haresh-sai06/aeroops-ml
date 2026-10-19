| Model | CV ROC-AUC (2 folds) | Test ROC-AUC | Test PR-AUC | Test Brier | Test F1 @thr |
|---|---|---|---|---|---|
| Prior (base rate) | 0.5 ± 0.0 | 0.5 | 0.2085 | 0.1651 | - |
| LogReg: schedule + history (numeric) | 0.5902 ± 0.0159 | 0.6539 | 0.3205 | 0.1581 | 0.3797 |
| XGB: schedule only | 0.5773 ± 0.0272 | 0.6431 | 0.318 | 0.1609 | 0.3674 |
| XGB: schedule + history | 0.5874 ± 0.0136 | 0.6449 | 0.3117 | 0.1626 | 0.3825 |
| XGB: schedule + history + operational | 0.7279 ± 0.015 | 0.7736 | 0.5975 | 0.124 | 0.5254 |
