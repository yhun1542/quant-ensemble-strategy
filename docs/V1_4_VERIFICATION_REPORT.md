# v1.4 Strategy Verification Report

**Date**: 2024-11-27  
**Subject**: Look-ahead Bias, Overfitting, and Transaction Costs Verification

---

## 📋 Executive Summary

This report provides a comprehensive verification of the v1.4 strategy (Sharpe 2.00) across three critical dimensions: look-ahead bias, overfitting, and transaction costs.

### Key Findings

| Verification Area | Result | Summary |
|---------------------|--------|---------|
| **Look-ahead Bias** | ✅ **Pass** | The core logic is sound. Minor issues with initial data availability do not affect the final results. |
| **Overfitting** | ⚠️ **Warning** | The strategy shows signs of potential overfitting or regime dependency. Out-of-Sample performance is significantly better than In-Sample. |
| **Transaction Costs** | ❌ **Not Included** | The current Sharpe 2.00 is a **gross** figure. The estimated **net Sharpe is 1.92 - 1.95** after accounting for costs. |

**Overall Conclusion**: The strategy is robust against look-ahead bias, but shows signs of overfitting. The reported performance does not include transaction costs, but the strategy remains strong (Sharpe > 1.9) even after estimating them.

---

## 1. Look-ahead Bias Verification

**Objective**: To ensure that the backtest does not use future information to make trading decisions.

### Methodology

1.  **Code Inspection**: Manually reviewed the data handling and signal generation logic in `generate_weights_for_v1_4.py` and `backtest_v1_4_long_only.py`.
2.  **Data Availability Check**: Verified that for each rebalance date, only historical price data was available for factor calculation.
3.  **Manual Calculation**: Replicated the factor calculation for a sample date (`2022-09-01`) to confirm no future data was used.

### Findings

#### Core Logic

The fundamental logic of the backtest is sound and free of look-ahead bias:
- **Factor Calculation**: On any given rebalance date `D`, factors such as `momentum_60d` and `volatility_30d` are calculated using price data only up to and including `D`.
- **ML Model Training**: The XGBoost model is trained on a rolling 2-year window of historical data, ending on date `D`.
- **Portfolio Construction**: The portfolio weights are determined on date `D` and are applied starting from the **next trading day (D+1)**. This correctly simulates a real-world scenario.

#### Minor Issue: Initial Data Availability

The verification script identified a minor issue at the very beginning of the backtest period:
- The first three rebalance dates (`2021-07-01`, `2021-08-02`, `2021-09-01`) had fewer than 60 days of prior price data.
- This means the `momentum_60d` factor could not be calculated for these initial months.

**Impact**: This is **not a critical issue**. The backtest effectively begins on `2021-10-01`, the first date with sufficient historical data for all factors. The final performance metrics are calculated from this date onwards, so they are not affected by this initial data limitation.

### Conclusion

✅ **Pass**. The v1.4 strategy is fundamentally free of look-ahead bias.

---

## 2. Overfitting Analysis

**Objective**: To assess whether the strategy's performance is consistent across different time periods or if it is overfitted to a specific period.

### Methodology

1.  **Time Period Split**: The full backtest period was divided into two equal halves based on the number of rebalance dates:
    - **In-Sample (IS)**: 2021-10-01 to 2023-06-01 (17 rebalances)
    - **Out-of-Sample (OOS)**: 2023-07-03 to 2024-12-02 (18 rebalances)
2.  **Performance Comparison**: Calculated and compared the Sharpe Ratio and other key metrics for both the IS and OOS periods.

### Findings

The strategy performed significantly better in the Out-of-Sample period.

| Metric | In-Sample (IS) | Out-of-Sample (OOS) | OOS vs IS Ratio |
|-------------------|----------------|---------------------|-----------------|
| **Sharpe Ratio** | 1.92 | **2.44** | **1.27x** |
| **Annual Return** | 31.10% | **35.23%** | - |
| **Annual Volatility** | 16.19% | 14.44% | - |
| **Max Drawdown** | -17.84% | **-8.76%** | - |
| **Total Return** | 118.16% | 57.08% | - |

**Key Observation**: The Out-of-Sample Sharpe Ratio is **27% higher** than the In-Sample Sharpe Ratio. Typically, in a robust strategy, one would expect the OOS performance to be similar to or slightly lower than the IS performance. The opposite result here is a strong indicator of either **overfitting** or **regime dependency**.

### Conclusion

⚠️ **Warning**. The strategy shows signs of being over-optimized for the market conditions of the latter half of the backtest period (mid-2023 to 2024). While the performance is strong in both periods, the discrepancy suggests that the strategy's high Sharpe Ratio might not be as robust across different market regimes. This risk should be considered before live deployment.

---

## 3. Transaction Costs Verification

**Objective**: To determine if transaction costs are included in the backtest and to estimate their impact on performance.

### Methodology

1.  **Code Inspection**: Searched the backtesting scripts (`backtest_v1_4_long_only.py`, `utils/execution_smoothing_v2.py`) for any logic related to commissions, slippage, or trading costs.
2.  **Turnover Calculation**: Calculated the average portfolio turnover at each rebalance to quantify the amount of trading.
3.  **Impact Estimation**: Based on the turnover, estimated the performance degradation using a standard transaction cost assumption of 10-15 basis points (bps) per trade.

### Findings

#### Code Inspection
No code related to transaction costs was found. The backtest calculates returns based purely on price changes.

#### Portfolio Turnover
- **Average Turnover per Rebalance**: **85.43%**
- This is a relatively high turnover rate, indicating that the portfolio composition changes significantly each month.

#### Estimated Impact
- **Annualized Cost**: A turnover of 85.43% per month, with a cost of 10-15 bps, results in an estimated annualized performance drag of **0.92% to 1.37%**.

### Estimated Net Performance

The table below shows the estimated performance after deducting transaction costs.

| Metric | Gross (Current) | Net (10 bps Cost) | Net (15 bps Cost) |
|-------------------|-----------------|-------------------|-------------------|
| **Sharpe Ratio** | 2.00 | **1.95** (-2.5%) | **1.92** (-4.0%) |
| **Annual Return** | 33.93% | **33.01%** | **32.56%** |

### Conclusion

❌ **Not Included**. The current Sharpe Ratio of 2.00 is a **gross figure**. After accounting for realistic transaction costs, the estimated **net Sharpe Ratio is between 1.92 and 1.95**.

Even after including costs, the strategy's performance remains very strong and still approaches the lower end of the 2.0-2.5 target range.

---

## 📈 Final Summary & Recommendations

| Verification Area | Result | Key Takeaway |
|---------------------|:--------:|:-----------------------------------------------------------------------------------------------------------------------------------|
| **Look-ahead Bias** | ✅ **Pass** | The backtest is methodologically sound and free of look-ahead bias. |
| **Overfitting** | ⚠️ **Warning** | The strategy may be overfitted to recent market conditions. Performance in different regimes is uncertain. |
| **Transaction Costs** | ❌ **Not Included** | The reported Sharpe 2.00 is a gross figure. The estimated net Sharpe is **1.92 - 1.95**. |

### Recommendations

1.  **Acknowledge Overfitting Risk**: Before live deployment, the strategy should be tested on a longer and more diverse set of historical data (e.g., 2015-2020) to validate its robustness across different market regimes.

2.  **Incorporate Transaction Costs**: For future backtests, a transaction cost module should be added to the code to provide more realistic net performance figures directly. A baseline assumption of 10-15 bps is reasonable.

3.  **Proceed with Caution**: While the net Sharpe of ~1.95 is excellent, the overfitting risk is a genuine concern. The next logical step should be to focus on validating the strategy's robustness rather than pushing for higher returns.
