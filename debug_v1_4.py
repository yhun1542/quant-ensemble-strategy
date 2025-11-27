#!/usr/bin/env python3
"""
Debug v1.4 Implementation
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_simple

print("="*100)
print("Debugging v1.4 Implementation")
print("="*100)

# Load weights
print("\n[1/4] Loading weights...")
with open('results/ensemble_fv3c_ml9.json', 'r') as f:
    data = json.load(f)

fv3c_weights_dict = data['fv3c_weights']
ml9_weights_dict = data['ml9_weights']

# Convert to Series
fv3c_weights_by_date = {}
for date_str, weights_dict in fv3c_weights_dict.items():
    date = pd.Timestamp(date_str)
    fv3c_weights_by_date[date] = pd.Series(weights_dict)

ml9_weights_by_date = {}
for date_str, weights_dict in ml9_weights_dict.items():
    date = pd.Timestamp(date_str)
    ml9_weights_by_date[date] = pd.Series(weights_dict)

# Check first few weights
print("\nFirst FV3c weights:")
first_date = sorted(fv3c_weights_by_date.keys())[0]
print(f"  Date: {first_date}")
print(f"  Weights:\n{fv3c_weights_by_date[first_date]}")
print(f"  Sum: {fv3c_weights_by_date[first_date].sum():.4f}")
print(f"  Long sum: {fv3c_weights_by_date[first_date][fv3c_weights_by_date[first_date] > 0].sum():.4f}")
print(f"  Short sum: {fv3c_weights_by_date[first_date][fv3c_weights_by_date[first_date] < 0].sum():.4f}")

print("\nFirst ML9 weights:")
first_date_ml9 = sorted(ml9_weights_by_date.keys())[0]
print(f"  Date: {first_date_ml9}")
print(f"  Weights:\n{ml9_weights_by_date[first_date_ml9]}")
print(f"  Sum: {ml9_weights_by_date[first_date_ml9].sum():.4f}")
print(f"  Long sum: {ml9_weights_by_date[first_date_ml9][ml9_weights_by_date[first_date_ml9] > 0].sum():.4f}")
print(f"  Short sum: {ml9_weights_by_date[first_date_ml9][ml9_weights_by_date[first_date_ml9] < 0].sum():.4f}")

# Create ensemble
print("\n[2/4] Creating ensemble weights...")
all_dates = sorted(set(fv3c_weights_by_date.keys()) & set(ml9_weights_by_date.keys()))
ensemble_weights_by_date = {}

for date in all_dates:
    fv3c_w = fv3c_weights_by_date[date]
    ml9_w = ml9_weights_by_date[date]
    
    all_tickers = fv3c_w.index.union(ml9_w.index)
    fv3c_full = fv3c_w.reindex(all_tickers).fillna(0.0)
    ml9_full = ml9_w.reindex(all_tickers).fillna(0.0)
    
    ensemble_w = 0.6 * fv3c_full + 0.4 * ml9_full
    ensemble_w = ensemble_w[ensemble_w.abs() > 1e-8]
    
    ensemble_weights_by_date[date] = ensemble_w

print(f"  Created {len(ensemble_weights_by_date)} ensemble weight sets")

# Check first ensemble weights
first_ensemble_date = sorted(ensemble_weights_by_date.keys())[0]
print(f"\nFirst ensemble weights (Date: {first_ensemble_date}):")
print(f"  Weights:\n{ensemble_weights_by_date[first_ensemble_date]}")
print(f"  Sum: {ensemble_weights_by_date[first_ensemble_date].sum():.4f}")
print(f"  Long sum: {ensemble_weights_by_date[first_ensemble_date][ensemble_weights_by_date[first_ensemble_date] > 0].sum():.4f}")
print(f"  Short sum: {ensemble_weights_by_date[first_ensemble_date][ensemble_weights_by_date[first_ensemble_date] < 0].sum():.4f}")

# Load prices
print("\n[3/4] Loading prices...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")

# Test simple returns (no execution smoothing)
print("\n[4/4] Testing simple returns (no execution smoothing)...")
rebalance_dates = sorted(ensemble_weights_by_date.keys())

simple_returns = portfolio_returns_simple(
    prices=prices,
    weights_by_date=ensemble_weights_by_date,
    rebalance_dates=rebalance_dates
)

print(f"  Calculated {len(simple_returns)} days of returns")

# Calculate metrics
def calc_metrics(rets):
    total = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1/n_years) - 1 if total > -1 else np.nan
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 and not np.isnan(ann_ret) else np.nan
    
    cum = (1 + rets).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    
    return {
        'sharpe': sharpe,
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'max_dd': max_dd,
        'total_return': total,
        'n_days': len(rets)
    }

metrics = calc_metrics(simple_returns)

print("\n" + "="*100)
print("Simple Returns (No Execution Smoothing)")
print("="*100)
print(f"  Sharpe Ratio:   {metrics['sharpe']:.2f}")
print(f"  Annual Return:  {metrics['annual_return']:.2%}")
print(f"  Annual Vol:     {metrics['annual_vol']:.2%}")
print(f"  Max DD:         {metrics['max_dd']:.2%}")
print(f"  Total Return:   {metrics['total_return']:.2%}")
print(f"  Days:           {metrics['n_days']}")

# Check first few returns
print("\nFirst 10 returns:")
print(simple_returns.head(10))

print("\nLast 10 returns:")
print(simple_returns.tail(10))

print("\nReturn statistics:")
print(f"  Mean: {simple_returns.mean():.6f}")
print(f"  Std:  {simple_returns.std():.6f}")
print(f"  Min:  {simple_returns.min():.6f}")
print(f"  Max:  {simple_returns.max():.6f}")
