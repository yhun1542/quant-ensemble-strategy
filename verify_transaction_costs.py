#!/usr/bin/env python3
"""
Transaction Costs Verification for v1.4

This script checks if transaction costs are included in the backtest.
"""
import json
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.execution_smoothing_v2 import portfolio_returns_simple

print("="*100)
print("Transaction Costs Verification for v1.4")
print("="*100)

# Check 1: Code inspection
print("\n[1/3] Code Inspection")
print("-" * 100)

print("\nChecking backtest_v1_4_long_only.py...")
with open('backtest_v1_4_long_only.py', 'r') as f:
    code = f.read()

has_tc = False
if 'transaction_cost' in code.lower() or 'trading_cost' in code.lower() or 'commission' in code.lower():
    has_tc = True
    print("  ✅ Transaction cost keywords found in code")
else:
    print("  ❌ No transaction cost keywords found in code")

print("\nChecking utils/execution_smoothing_v2.py...")
with open('utils/execution_smoothing_v2.py', 'r') as f:
    code = f.read()

if 'transaction_cost' in code.lower() or 'trading_cost' in code.lower() or 'commission' in code.lower():
    has_tc = True
    print("  ✅ Transaction cost keywords found in code")
else:
    print("  ❌ No transaction cost keywords found in code")

# Check 2: Calculate transaction costs
print("\n[2/3] Calculating Transaction Costs Impact")
print("-" * 100)

# Load data
prices = pd.read_csv('data/price_data_sp500.csv', index_col=0, parse_dates=True)
prices = prices.sort_index()
prices.index = prices.index.normalize()

with open('results/ensemble_fv3c_ml9.json', 'r') as f:
    data = json.load(f)

# Create ensemble weights
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

# Calculate turnover
print("\nCalculating portfolio turnover...")
rebal_dates = sorted(ensemble_weights_by_date.keys())
turnovers = []

for i in range(1, len(rebal_dates)):
    prev_date = rebal_dates[i-1]
    curr_date = rebal_dates[i]
    
    prev_w = ensemble_weights_by_date[prev_date]
    curr_w = ensemble_weights_by_date[curr_date]
    
    # Align weights
    all_tickers = prev_w.index.union(curr_w.index)
    prev_w_aligned = prev_w.reindex(all_tickers).fillna(0.0)
    curr_w_aligned = curr_w.reindex(all_tickers).fillna(0.0)
    
    # Turnover = sum of absolute weight changes
    turnover = (prev_w_aligned - curr_w_aligned).abs().sum()
    turnovers.append(turnover)

avg_turnover = np.mean(turnovers)
print(f"  Average turnover per rebalance: {avg_turnover:.2%}")
print(f"  Number of rebalances: {len(turnovers)}")
print(f"  Total turnover: {sum(turnovers):.2f}")

# Estimate transaction costs
print("\nEstimating transaction costs...")

# Typical costs for US equities:
# - Commission: ~$0.005 per share (or 0.5 bps)
# - Spread: ~5 bps for liquid stocks
# - Market impact: ~2-5 bps for small orders
# Total: ~10-15 bps per trade

tc_bps_low = 10  # 10 bps (0.10%)
tc_bps_high = 15  # 15 bps (0.15%)

total_tc_low = sum(turnovers) * (tc_bps_low / 10000)
total_tc_high = sum(turnovers) * (tc_bps_high / 10000)

# Annualize
n_years = (rebal_dates[-1] - rebal_dates[0]).days / 365.25
ann_tc_low = total_tc_low / n_years
ann_tc_high = total_tc_high / n_years

print(f"  Transaction cost assumption: {tc_bps_low}-{tc_bps_high} bps per trade")
print(f"  Total transaction costs (3.5y): {total_tc_low:.2%} - {total_tc_high:.2%}")
print(f"  Annualized transaction costs: {ann_tc_low:.2%} - {ann_tc_high:.2%}")

# Impact on performance
with open('results/v1_4_long_only_results.json', 'r') as f:
    v1_4_results = json.load(f)

gross_sharpe = v1_4_results['metrics']['sharpe']
gross_ann_ret = v1_4_results['metrics']['annual_return']

net_ann_ret_low = gross_ann_ret - ann_tc_low
net_ann_ret_high = gross_ann_ret - ann_tc_high

# Estimate net Sharpe (assuming vol unchanged)
gross_ann_vol = v1_4_results['metrics']['annual_vol']
net_sharpe_low = net_ann_ret_low / gross_ann_vol
net_sharpe_high = net_ann_ret_high / gross_ann_vol

print("\n[3/3] Performance Impact")
print("-" * 100)

print(f"\nGross Performance (current v1.4):")
print(f"  Sharpe Ratio:   {gross_sharpe:.2f}")
print(f"  Annual Return:  {gross_ann_ret:.2%}")
print(f"  Annual Vol:     {gross_ann_vol:.2%}")

print(f"\nNet Performance (after TC, low estimate):")
print(f"  Sharpe Ratio:   {net_sharpe_low:.2f} ({net_sharpe_low - gross_sharpe:+.2f})")
print(f"  Annual Return:  {net_ann_ret_low:.2%} ({net_ann_ret_low - gross_ann_ret:+.2%}p)")

print(f"\nNet Performance (after TC, high estimate):")
print(f"  Sharpe Ratio:   {net_sharpe_high:.2f} ({net_sharpe_high - gross_sharpe:+.2f})")
print(f"  Annual Return:  {net_ann_ret_high:.2%} ({net_ann_ret_high - gross_ann_ret:+.2%}p)")

# Summary
print("\n" + "="*100)
print("Transaction Costs Summary")
print("="*100)

if not has_tc:
    print("\n❌ TRANSACTION COSTS NOT INCLUDED IN BACKTEST")
    print("\nCurrent v1.4 results are GROSS returns (before costs).")
    print("\nEstimated NET performance after transaction costs:")
    print(f"  Sharpe: {net_sharpe_low:.2f} - {net_sharpe_high:.2f}")
    print(f"  Annual Return: {net_ann_ret_low:.2%} - {net_ann_ret_high:.2%}")
    print("\nRecommendation:")
    print("  1. Add transaction cost module to backtest")
    print("  2. Use 10-15 bps per trade as baseline")
    print("  3. Re-run backtest with costs included")
else:
    print("\n✅ TRANSACTION COSTS INCLUDED IN BACKTEST")
    print("\nCurrent v1.4 results are NET returns (after costs).")

print("\n" + "="*100)
print("Verification Complete")
print("="*100)
