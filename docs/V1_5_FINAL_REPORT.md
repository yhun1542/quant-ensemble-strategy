_# v1.5 Strategy Final Report: Transaction Costs & Overfitting Mitigation

**Date**: 2024-11-27  
**Subject**: Final report for the v1.5 strategy, incorporating transaction costs and walk-forward validation to address overfitting.

---

## 📋 Executive Summary

This report details the development and testing of the v1.5 strategy, an enhanced version of v1.4 designed to provide a more realistic performance assessment by including transaction costs and addressing the high overfitting risk identified previously.

### Key Improvements in v1.5

1.  **Transaction Costs Included**: A transaction cost module was developed and integrated, applying a standard **8.5 bps** cost per trade to all backtests.
2.  **Overfitting Mitigation & Analysis**: 
    - **Walk-Forward Validation**: Implemented a rigorous walk-forward analysis to test the strategy's consistency over time.
    - **Robust Parameters**: Used more conservative, regularized parameters for the ML9 (XGBoost) engine to reduce its sensitivity to training data.

### Final Performance (v1.5)

| Metric | v1.4 (Gross) | v1.5 (Net) | Change |
|-------------------|:--------------:|:----------:|:--------:|
| **Net Sharpe Ratio** | 2.00 | **1.86** | -7.0% |
| **Net Annual Return** | 33.93% | **31.95%** | -1.98%p |
| **Annual Volatility** | 16.95% | **17.19%** | +0.24%p |
| **Max Drawdown** | -17.50% | **-20.79%** | -3.29%p |

### Overfitting Assessment

- **Walk-Forward Result**: **Poor Consistency** (Score: 0.42)
- **Conclusion**: The strategy's performance is highly variable across different time periods, confirming that the **overfitting risk remains HIGH**.

**Overall Conclusion**: The v1.5 backtest provides a more realistic Net Sharpe Ratio of **1.86**. However, the walk-forward analysis confirms that the strategy is not robust and is likely overfitted to the market conditions of 2023. Further improvements are required before the strategy can be considered reliable.

---

## 1. v1.5 Improvements Implemented

### a) Transaction Cost Module

**Objective**: To calculate and apply realistic trading costs to the backtest.

**Implementation** (`utils/transaction_costs.py`):
- A `TransactionCostModel` class was created.
- It models costs based on three components:
    - **Commission**: 0.5 bps
    - **Bid-Ask Spread**: 5.0 bps
    - **Market Impact**: 3.0 bps
- **Total Cost**: **8.5 bps** (0.085%) applied to the value of the assets traded at each rebalance.
- The cost is deducted from the return of the next trading day following a rebalance.

**Impact**:
- The annualized return was reduced by **-0.89%p**.
- The final Sharpe Ratio dropped from a gross 1.91 to a net **1.86**.

### b) Overfitting Mitigation and Analysis

**Objective**: To address the high overfitting risk identified in the v1.4 verification.

**Implementation**:

1.  **Robust ML Parameters** (`backtest_v1_5_with_costs.py`):
    - The XGBoost model in the ML9 engine was configured with stronger regularization parameters (`reg_alpha=1.0`, `reg_lambda=3.0`) to make it less prone to overfitting on the training data.

2.  **Walk-Forward Validation** (`walk_forward_validation.py`):
    - The full 3.5-year backtest period was broken down into seven overlapping 12-month windows, each stepping forward by 6 months.
    - Performance metrics were calculated independently for each window.
    - The standard deviation of the Sharpe Ratios across these windows was used to measure performance consistency.

---

## 2. Final Performance Results (v1.5)

**Backtest Period**: 2021-10-05 to 2024-12-31 (725 days)

The following table shows the final, more realistic performance of the v1.5 strategy.

| Metric | Gross Performance | Transaction Cost | Net Performance |
|-------------------|:-----------------:|:------------------:|:---------------:|
| **Sharpe Ratio** | 1.91 | **-0.05** | **1.86** |
| **Annual Return** | 32.84% | **-0.89%p** | **31.95%** |
| **Annual Volatility** | 17.17% | +0.02%p | **17.19%** |
| **Max Drawdown** | -20.67% | -0.12%p | **-20.79%** |
| **Total Return** | 126.36% | **-4.34%p** | **122.02%** |
| **Win Rate** | 56.14% | -0.14%p | **56.00%** |

**Conclusion**: After including transaction costs, the strategy still delivers a strong Net Sharpe Ratio of **1.86**, which is very close to the original target of 2.0.

---

## 3. Walk-Forward Validation Results

While the overall Sharpe is high, the walk-forward analysis reveals that this performance is not stable over time.

### Performance per 12-Month Window

| Window | Period | Sharpe | Ann. Return | Max DD |
|:------:|-----------------------------|:------:|:-----------:|:--------:|
| 1 | 2021-10 to 2022-10 | **0.27** | 5.97% | -20.79% |
| 2 | 2022-04 to 2023-04 | **1.08** | 24.66% | -15.80% |
| 3 | 2022-10 to 2023-10 | **2.79** | 47.52% | -6.73% |
| 4 | 2023-04 to 2024-04 | **3.06** | 41.24% | -6.14% |
| 5 | 2023-10 to 2024-10 | **2.99** | 42.31% | -7.86% |
| 6 | 2024-04 to 2024-12 | **2.04** | 29.82% | -7.13% |
| 7 | 2024-10 to 2024-12 | **0.75** | 11.32% | -4.80% |

### Consistency Analysis

- **Mean Sharpe**: 1.85 (Consistent with the full-period Sharpe of 1.86)
- **Std Dev of Sharpe**: **1.07** (Very High)
- **Min Sharpe**: 0.27
- **Max Sharpe**: 3.06
- **Consistency Score**: 0.42 (Poor)

**Conclusion**: ❌ **Poor Consistency**. The standard deviation of the Sharpe Ratios (1.07) is very high relative to the mean (1.85). This indicates that the strategy's performance is highly dependent on the specific time period. The excellent overall performance is driven almost entirely by the strong bull market in 2023. The strategy performed poorly in late 2021 and early 2022.

---

## 4. Final Verification Status

| Verification Area | Status | Final Assessment |
|---------------------|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------|
| **Look-ahead Bias** | ✅ **Pass** | The backtesting methodology remains sound and free of look-ahead bias. |
| **Transaction Costs** | ✅ **Included** | The v1.5 results are **net returns** after applying a realistic 8.5 bps cost per trade. |
| **Overfitting** | ❌ **High Risk** | Walk-forward analysis confirms that the strategy is not robust and shows high performance variability, indicating a high risk of overfitting. |

---

## 5. Conclusion & Recommendations

### Final Assessment

The v1.5 strategy represents a more realistic and accurate backtest, yielding a **Net Sharpe Ratio of 1.86**. While this is a strong result, the walk-forward validation has confirmed that the strategy in its current form is **not robust** and carries a **high risk of overfitting**. Its success is heavily concentrated in the 2023 market regime, and it performs poorly in other periods.

### Recommendations for v1.6

The overfitting issue must be the primary focus of the next iteration. Simply tuning parameters is insufficient. A more fundamental change is needed.

1.  **Diversify Engine Factors (Highest Priority)**:
    - The current engines (FV3c and ML9) are both heavily reliant on value and momentum factors, which makes them perform similarly.
    - **Action**: Introduce a new engine based on a different, uncorrelated factor family, such as **Quality** (e.g., profitability, leverage) or **Sentiment** (e.g., news analysis, social media trends).

2.  **Implement Dynamic Weighting**:
    - The current 60:40 ensemble weight is static.
    - **Action**: Develop a "meta-model" that dynamically adjusts the weights allocated to each engine based on the prevailing market regime (e.g., using a volatility index like VIX or macroeconomic indicators).

3.  **Use Anchored Walk-Forward Training**:
    - The current ML model uses a 2-year rolling window, which can forget older patterns.
    - **Action**: Switch to an "anchored" training window that starts at a fixed point in the past and expands forward. This allows the model to learn from all available history.

By focusing on **engine diversification** and **dynamic adaptation**, the v1.6 strategy can be made more robust and less dependent on a single market regime, leading to more consistent performance over the long run. 
long-term.
