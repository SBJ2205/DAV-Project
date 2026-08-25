# Research Summary: Small Business Insolvency Early-Warning Analytics

## 1. Research Objectives & Answers

### RQ1: Minority Class Detection in Imbalanced SME Data
- In a dataset with **3.23% minority class prevalence** (220 bankrupt vs 6,599 healthy firms), standard accuracy (~96.8%) is misleading.
- Using class-balanced tree ensembles and PR-AUC, the system successfully identifies **65.9% of distressed firms** while maintaining a high specificity of **96.3%**.

### RQ2: Key Financial Variables Contributing to Bankruptcy
Global SHAP analysis across all models identifies the following top 5 financial predictors:
1. **Continuous interest rate (after tax)** — Reflects heavy financing burdens relative to operating returns.
2. **Borrowing dependency** — High reliance on external debt elevates liquidity risk.
3. **Persistent EPS in the Last Four Seasons** — Sustained earnings deterioration strongly signals distress.
4. **Total debt / Total net worth** — Excessive financial leverage reduces solvency buffers.
5. **Equity to Liability** — Low capitalization increases vulnerability to cash shocks.

### RQ3: Model Comparison (Linear vs Nonlinear Ensembles)

| Model | ROC-AUC | PR-AUC (Key) | Recall (Distressed) | Precision | F1-Score | Specificity |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.9171 | 0.3190 | **0.8182** | 0.1856 | 0.3025 | 0.8803 |
| **Random Forest (Balanced)** | **0.9486** | **0.4999** | 0.6591 | 0.3718 | 0.4754 | **0.9629** |
| **XGBoost (Weighted)** | **0.9531** | 0.4779 | 0.6136 | **0.3913** | **0.4779** | **0.9682** |

- **Conclusion**: Random Forest and XGBoost achieve significantly superior PR-AUC (~0.48–0.50 vs 0.32) and far fewer false alarms compared to Logistic Regression.

### RQ4: Actionability of What-If Sensitivity Trajectories
- Single-variable sensitivity trajectories allow decision-makers to quantitatively simulate the impact of targeted managerial interventions (e.g., debt restructuring or liquidity buffers) on model-estimated insolvency probability without black-box opacity.

---

## 2. Limitations & Transparency
1. **Cross-Sectional Observations**: The dataset contains single-period snapshots per company; trajectory analysis models mathematical sensitivity rather than true historical progression.
2. **Associative, Not Causal**: SHAP indicates feature importance in the model's decision surface; it does not prove economic causation.
