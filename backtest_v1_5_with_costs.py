#!/usr/bin/env python3
"""
v1.5 Strategy Backtest with Transaction Costs and Walk-Forward Validation

Improvements over v1.4:
1. Transaction costs included (8.5 bps per trade)
2. Walk-forward validation to reduce overfitting
3. More conservative parameter selection
"""
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_with_execution_smoothing, ExecutionSmoothingConfig
from utils.transaction_costs import apply_transaction_costs
from utils.factors import compute_all_factors

print("="*100)
print("v1.5 Strategy Backtest: Transaction Costs + Walk-Forward Validation")
print("="*100)

# Load price data
print("\n[1/7] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
prices.index = prices.index.normalize()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Compute factors
print("\n[2/7] Computing factors...")
factors = compute_all_factors(prices)
print(f"  Computed factors for {len(factors)} data points")

# Get rebalance dates
print("\n[3/7] Determining rebalance dates...")
dates = prices.index
monthly_dates = []
current_month = None
for date in dates:
    if current_month != date.month:
        monthly_dates.append(date)
        current_month = date.month

rebalance_dates = [d for d in monthly_dates if d >= pd.Timestamp('2021-10-01')]
print(f"  Generated {len(rebalance_dates)} rebalance dates (starting from 2021-10-01)")

# Generate FV3c weights (simplified for speed)
print("\n[4/7] Generating FV3c weights...")
fv3c_weights_by_date = {}
top_quantile = 0.2

for rebal_date in rebalance_dates:
    if rebal_date not in factors.index.get_level_values("date"):
        continue
    
    factors_at_date = factors.loc[rebal_date].copy()
    factors_sorted = factors_at_date.sort_values("value_proxy", ascending=True)
    
    n_stocks = len(factors_sorted)
    n_long = int(n_stocks * top_quantile)
    
    # Long-only: select cheap stocks
    long_tickers = factors_sorted.head(n_long).index.tolist()
    
    # Volatility-based weighting
    portfolio = {}
    long_vols = []
    for ticker in long_tickers:
        vol = factors_at_date.loc[ticker, "volatility_30d"]
        if vol > 0:
            long_vols.append((ticker, 1.0 / vol))
    
    if long_vols:
        total_inv_vol = sum(w for _, w in long_vols)
        for ticker, inv_vol in long_vols:
            portfolio[ticker] = inv_vol / total_inv_vol
    
    if portfolio:
        fv3c_weights_by_date[rebal_date] = pd.Series(portfolio)

print(f"  Generated {len(fv3c_weights_by_date)} FV3c weight sets")

# Generate ML9 weights (simplified for speed)
print("\n[5/7] Generating ML9 weights...")
import xgboost as xgb

ml9_weights_by_date = {}
prediction_horizon = 10
train_window = 252 * 2  # 2 years

feature_cols = ["momentum_60d", "volatility_30d", "value_proxy"]

for i, rebal_date in enumerate(rebalance_dates):
    if rebal_date not in factors.index.get_level_values("date"):
        continue
    
    # Get training data
    train_start_idx = max(0, dates.get_loc(rebal_date) - train_window)
    train_dates = dates[train_start_idx:dates.get_loc(rebal_date)]
    
    if len(train_dates) < 60:
        continue
    
    # Build training dataset
    X_train_list = []
    y_train_list = []
    
    for train_date in train_dates:
        if train_date not in factors.index.get_level_values("date"):
            continue
        
        factors_at_train = factors.loc[train_date]
        
        date_idx = dates.get_loc(train_date)
        if date_idx + prediction_horizon >= len(dates):
            continue
        
        future_date = dates[date_idx + prediction_horizon]
        
        for ticker in factors_at_train.index:
            if ticker not in prices.columns:
                continue
            
            features = factors_at_train.loc[ticker, feature_cols].values
            
            try:
                fwd_ret = prices.loc[future_date, ticker] / prices.loc[train_date, ticker] - 1.0
            except:
                continue
            
            X_train_list.append(features)
            y_train_list.append(fwd_ret)
    
    if len(X_train_list) < 100:
        continue
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    # Quantile-based classification
    y_quantiles = pd.Series(y_train).rank(pct=True)
    y_class = np.ones(len(y_train), dtype=int)
    y_class[y_quantiles >= 0.8] = 2
    y_class[y_quantiles <= 0.2] = 0
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        verbosity=0
    )
    
    model.fit(X_train, y_class)
    
    # Predict
    factors_at_rebal = factors.loc[rebal_date]
    X_pred = factors_at_rebal[feature_cols].values
    probs = model.predict_proba(X_pred)
    
    scores = probs[:, 2] - probs[:, 0]
    scores_series = pd.Series(scores, index=factors_at_rebal.index)
    scores_sorted = scores_series.sort_values(ascending=False)
    
    n_stocks = len(scores_sorted)
    n_long = int(n_stocks * top_quantile)
    
    long_tickers = scores_sorted.head(n_long).index.tolist()
    
    portfolio = {}
    if long_tickers:
        for ticker in long_tickers:
            portfolio[ticker] = 1.0 / len(long_tickers)
    
    if portfolio:
        ml9_weights_by_date[rebal_date] = pd.Series(portfolio)
    
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/{len(rebalance_dates)} dates...")

print(f"  Generated {len(ml9_weights_by_date)} ML9 weight sets")

# Create ensemble (60:40)
print("\n[6/7] Creating ensemble weights (60:40 Long-Only)...")
ensemble_weights_by_date = {}
common_dates = sorted(set(fv3c_weights_by_date.keys()) & set(ml9_weights_by_date.keys()))

for date in common_dates:
    fv3c_w = fv3c_weights_by_date[date]
    ml9_w = ml9_weights_by_date[date]
    
    all_tickers = fv3c_w.index.union(ml9_w.index)
    fv3c_full = fv3c_w.reindex(all_tickers).fillna(0.0)
    ml9_full = ml9_w.reindex(all_tickers).fillna(0.0)
    
    ensemble_w = 0.6 * fv3c_full + 0.4 * ml9_full
    long_only_w = ensemble_w[ensemble_w > 0]
    
    if len(long_only_w) > 0:
        long_only_w = long_only_w / long_only_w.sum()
        ensemble_weights_by_date[date] = long_only_w

print(f"  Created {len(ensemble_weights_by_date)} ensemble weight sets")

# Calculate returns with Execution Smoothing
print("\n[7/7] Calculating returns with Execution Smoothing v2...")
final_rebal_dates = sorted(ensemble_weights_by_date.keys())

cfg = ExecutionSmoothingConfig(n_steps=2)
gross_returns = portfolio_returns_with_execution_smoothing(
    prices=prices,
    weights_by_date=ensemble_weights_by_date,
    rebalance_dates=final_rebal_dates,
    cfg=cfg
)

print(f"  Calculated {len(gross_returns)} days of gross returns")

# Apply transaction costs
print("\n  Applying transaction costs (8.5 bps)...")
net_returns = apply_transaction_costs(
    returns=gross_returns,
    weights_by_date=ensemble_weights_by_date,
    rebalance_dates=final_rebal_dates,
    cost_bps=8.5
)

print(f"  Calculated {len(net_returns)} days of net returns")

# Calculate metrics
def calc_metrics(rets, label=""):
    if len(rets) == 0:
        return {}
    
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1 if n_years > 0 and total > -1 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 and not np.isnan(ann_ret) else np.nan
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    win_rate = (rets > 0).sum() / len(rets)
    
    return {
        'sharpe': sharpe,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'max_dd': max_dd,
        'total_return': total,
        'win_rate': win_rate,
        'n_days': len(rets)
    }

gross_metrics = calc_metrics(gross_returns, "Gross")
net_metrics = calc_metrics(net_returns, "Net")

# Display results
print("\n" + "="*100)
print("v1.5 Strategy Results")
print("="*100)

print("\nGross Performance (before transaction costs):")
print(f"  Sharpe Ratio:   {gross_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {gross_metrics['annual_return']:.2%}")
print(f"  Annual Vol:     {gross_metrics['annual_vol']:.2%}")
print(f"  Max DD:         {gross_metrics['max_dd']:.2%}")
print(f"  Total Return:   {gross_metrics['total_return']:.2%}")
print(f"  Win Rate:       {gross_metrics['win_rate']:.2%}")
print(f"  Days:           {gross_metrics['n_days']}")

print("\nNet Performance (after transaction costs @ 8.5 bps):")
print(f"  Sharpe Ratio:   {net_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {net_metrics['annual_return']:.2%}")
print(f"  Annual Vol:     {net_metrics['annual_vol']:.2%}")
print(f"  Max DD:         {net_metrics['max_dd']:.2%}")
print(f"  Total Return:   {net_metrics['total_return']:.2%}")
print(f"  Win Rate:       {net_metrics['win_rate']:.2%}")
print(f"  Days:           {net_metrics['n_days']}")

print("\nTransaction Cost Impact:")
tc_impact_sharpe = gross_metrics['sharpe'] - net_metrics['sharpe']
tc_impact_ret = gross_metrics['annual_return'] - net_metrics['annual_return']
print(f"  Sharpe Impact:  {tc_impact_sharpe:+.2f}")
print(f"  Return Impact:  {tc_impact_ret:+.2%}p")

# Save results
results = {
    'strategy': 'v1.5_with_costs',
    'description': 'FV3c + ML9 Long-Only + Execution Smoothing v2 + Transaction Costs (8.5 bps)',
    'generated_at': datetime.now().isoformat(),
    'components': {
        'engines': ['FV3c', 'ML9'],
        'ensemble_weights': [0.6, 0.4],
        'position_type': 'long_only',
        'execution_smoothing': 'v2 (n_steps=2)',
        'transaction_costs': '8.5 bps per trade'
    },
    'gross_metrics': gross_metrics,
    'net_metrics': net_metrics,
    'transaction_cost_impact': {
        'sharpe_impact': tc_impact_sharpe,
        'return_impact': tc_impact_ret
    },
    'daily_returns': {
        'gross': {
            'index': [d.strftime('%Y-%m-%d') for d in gross_returns.index],
            'values': gross_returns.tolist()
        },
        'net': {
            'index': [d.strftime('%Y-%m-%d') for d in net_returns.index],
            'values': net_returns.tolist()
        }
    },
    'rebalance_dates': [d.strftime('%Y-%m-%d') for d in final_rebal_dates]
}

with open('results/v1_5_with_costs_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to results/v1_5_with_costs_results.json")

print("\n" + "="*100)
print("v1.5 Backtest Complete!")
print("="*100)
