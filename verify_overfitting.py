#!/usr/bin/env python3
"""
Overfitting Verification for v1.4

This script performs in-sample vs out-of-sample analysis to detect overfitting.
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_simple

print("="*100)
print("Overfitting Verification for v1.4")
print("="*100)

# Load data
print("\n[1/4] Loading data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
prices.index = prices.index.normalize()

with open('results/ensemble_fv3c_ml9.json', 'r') as f:
    data = json.load(f)

with open('results/v1_4_long_only_results.json', 'r') as f:
    v1_4_results = json.load(f)

print(f"  Price data: {prices.index[0]} to {prices.index[-1]}")
print(f"  Total days: {len(prices)}")

# Create ensemble weights
print("\n[2/4] Creating ensemble weights...")
fv3c_weights_dict = data['fv3c_weights']
ml9_weights_dict = data['ml9_weights']

ensemble_weights_by_date = {}
all_dates = sorted(set(fv3c_weights_dict.keys()) & set(ml9_weights_dict.keys()))

for date_str in all_dates:
    date = pd.Timestamp(date_str).normalize()
    
    fv3c_w = pd.Series(fv3c_weights_dict[date_str])
    ml9_w = pd.Series(ml9_weights_dict[date_str])
    
    all_tickers = fv3c_w.index.union(ml9_w.index)
    fv3c_full = fv3c_w.reindex(all_tickers).fillna(0.0)
    ml9_full = ml9_w.reindex(all_tickers).fillna(0.0)
    
    ensemble_w = 0.6 * fv3c_full + 0.4 * ml9_full
    long_only_w = ensemble_w[ensemble_w > 0]
    
    if len(long_only_w) > 0:
        long_only_w = long_only_w / long_only_w.sum()
        ensemble_weights_by_date[date] = long_only_w

print(f"  Created {len(ensemble_weights_by_date)} weight sets")

# Split into in-sample and out-of-sample
print("\n[3/4] Splitting into in-sample and out-of-sample...")

all_rebal_dates = sorted(ensemble_weights_by_date.keys())
n_dates = len(all_rebal_dates)
split_idx = int(n_dates * 0.5)  # 50/50 split

is_dates = all_rebal_dates[:split_idx]
oos_dates = all_rebal_dates[split_idx:]

print(f"  Total rebalance dates: {n_dates}")
print(f"  In-Sample: {len(is_dates)} dates ({is_dates[0]} to {is_dates[-1]})")
print(f"  Out-of-Sample: {len(oos_dates)} dates ({oos_dates[0]} to {oos_dates[-1]})")

# Calculate IS and OOS returns
def calc_returns(rebal_dates, weights_dict, prices):
    weights_subset = {d: weights_dict[d] for d in rebal_dates if d in weights_dict}
    returns = portfolio_returns_simple(
        prices=prices,
        weights_by_date=weights_subset,
        rebalance_dates=rebal_dates
    )
    return returns

is_returns = calc_returns(is_dates, ensemble_weights_by_date, prices)
oos_returns = calc_returns(oos_dates, ensemble_weights_by_date, prices)

print(f"\n  In-Sample returns: {len(is_returns)} days")
print(f"  Out-of-Sample returns: {len(oos_returns)} days")

# Calculate metrics
def calc_metrics(rets):
    if len(rets) == 0:
        return {'sharpe': 0, 'ann_ret': 0, 'ann_vol': 0, 'max_dd': 0}
    
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1 if n_years > 0 and total > -1 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 and not np.isnan(ann_ret) else np.nan
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    return {
        'sharpe': sharpe,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'max_dd': max_dd,
        'total_ret': total,
        'n_days': len(rets)
    }

is_metrics = calc_metrics(is_returns)
oos_metrics = calc_metrics(oos_returns)

# Display results
print("\n[4/4] Overfitting Analysis Results")
print("="*100)

print("\nIn-Sample Performance:")
print(f"  Sharpe Ratio:   {is_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {is_metrics['ann_ret']:.2%}")
print(f"  Annual Vol:     {is_metrics['ann_vol']:.2%}")
print(f"  Max DD:         {is_metrics['max_dd']:.2%}")
print(f"  Total Return:   {is_metrics['total_ret']:.2%}")
print(f"  Days:           {is_metrics['n_days']}")

print("\nOut-of-Sample Performance:")
print(f"  Sharpe Ratio:   {oos_metrics['sharpe']:.2f}")
print(f"  Annual Return:  {oos_metrics['ann_ret']:.2%}")
print(f"  Annual Vol:     {oos_metrics['ann_vol']:.2%}")
print(f"  Max DD:         {oos_metrics['max_dd']:.2%}")
print(f"  Total Return:   {oos_metrics['total_ret']:.2%}")
print(f"  Days:           {oos_metrics['n_days']}")

print("\nComparison (OOS vs IS):")
sharpe_ratio = oos_metrics['sharpe'] / is_metrics['sharpe'] if is_metrics['sharpe'] != 0 else np.nan
print(f"  Sharpe Ratio:   {sharpe_ratio:.2f}x")
print(f"  Annual Return:  {oos_metrics['ann_ret'] - is_metrics['ann_ret']:+.2%}p")
print(f"  Annual Vol:     {oos_metrics['ann_vol'] - is_metrics['ann_vol']:+.2%}p")

# Overfitting assessment
print("\n" + "="*100)
print("Overfitting Assessment")
print("="*100)

if np.isnan(sharpe_ratio):
    print("\n⚠️ CANNOT ASSESS - In-Sample Sharpe is zero or negative")
elif sharpe_ratio > 1.2:
    print(f"\n⚠️ POTENTIAL OVERFITTING DETECTED")
    print(f"  OOS Sharpe ({oos_metrics['sharpe']:.2f}) is {sharpe_ratio:.2f}x higher than IS Sharpe ({is_metrics['sharpe']:.2f})")
    print(f"  This suggests the strategy may be overfitted to the IS period.")
elif sharpe_ratio < 0.8:
    print(f"\n⚠️ POOR GENERALIZATION")
    print(f"  OOS Sharpe ({oos_metrics['sharpe']:.2f}) is only {sharpe_ratio:.2f}x of IS Sharpe ({is_metrics['sharpe']:.2f})")
    print(f"  Strategy performance degraded significantly out-of-sample.")
else:
    print(f"\n✅ GOOD GENERALIZATION")
    print(f"  OOS Sharpe ({oos_metrics['sharpe']:.2f}) is {sharpe_ratio:.2f}x of IS Sharpe ({is_metrics['sharpe']:.2f})")
    print(f"  Strategy shows consistent performance across periods.")

print("\n" + "="*100)
print("Verification Complete")
print("="*100)
