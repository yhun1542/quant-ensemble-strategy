#!/usr/bin/env python3
"""
Generate Engine Weights for v1.4

This script runs FV3c and ML9 engines from scratch to generate weights_by_date
for each rebalance date, which will be used in v1.4 with Execution Smoothing v2.
"""
import json
import pandas as pd
import numpy as np
import sys
from datetime import datetime

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from engines.factor_value_v3c import run_factor_value_v3c_backtest
from engines.ml_xgboost_v9 import run_ml_xgboost_v9_backtest

print("="*100)
print("Generating Engine Weights for v1.4")
print("="*100)

# Load price data
print("\n[1/5] Loading price data...")
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
print(f"  Loaded {len(prices)} days of prices for {len(prices.columns)} tickers")
print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

# Run FV3c engine
print("\n[2/5] Running FV3c engine...")
fv3c_results = run_factor_value_v3c_backtest(
    prices=prices,
    long_short=True,
    top_n=10,
    bottom_n=10
)

print(f"  FV3c Sharpe: {fv3c_results['sharpe']:.2f}")
print(f"  FV3c Annual Return: {fv3c_results['annual_return']:.2%}")
print(f"  FV3c weights generated for {len(fv3c_results['weights_by_date'])} dates")

# Run ML9 engine
print("\n[3/5] Running ML9 engine...")
ml9_results = run_ml_xgboost_v9_backtest(
    prices=prices,
    long_short=True,
    top_n=10,
    bottom_n=10
)

print(f"  ML9 Sharpe: {ml9_results['sharpe']:.2f}")
print(f"  ML9 Annual Return: {ml9_results['annual_return']:.2%}")
print(f"  ML9 weights generated for {len(ml9_results['weights_by_date'])} dates")

# Convert weights to serializable format
print("\n[4/5] Converting weights to JSON format...")
fv3c_weights_json = {}
for date, weights in fv3c_results['weights_by_date'].items():
    date_str = date.strftime('%Y-%m-%d')
    fv3c_weights_json[date_str] = weights.to_dict()

ml9_weights_json = {}
for date, weights in ml9_results['weights_by_date'].items():
    date_str = date.strftime('%Y-%m-%d')
    ml9_weights_json[date_str] = weights.to_dict()

# Save results
print("\n[5/5] Saving results...")
output = {
    'generated_at': datetime.now().isoformat(),
    'description': 'FV3c and ML9 engine weights for v1.4 implementation',
    'fv3c_metrics': {
        'sharpe': fv3c_results['sharpe'],
        'annual_return': fv3c_results['annual_return'],
        'annual_vol': fv3c_results['annual_vol'],
        'max_dd': fv3c_results['max_dd']
    },
    'ml9_metrics': {
        'sharpe': ml9_results['sharpe'],
        'annual_return': ml9_results['annual_return'],
        'annual_vol': ml9_results['annual_vol'],
        'max_dd': ml9_results['max_dd']
    },
    'fv3c_weights': fv3c_weights_json,
    'ml9_weights': ml9_weights_json,
    'rebalance_dates': sorted(fv3c_weights_json.keys())
}

with open('results/ensemble_fv3c_ml9.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"  ✅ Saved to results/ensemble_fv3c_ml9.json")
print(f"  Total rebalance dates: {len(output['rebalance_dates'])}")

print("\n" + "="*100)
print("Engine Weights Generation Complete!")
print("="*100)
print("\nNext step: Run backtest_v1_4_complete.py to apply Execution Smoothing v2")
