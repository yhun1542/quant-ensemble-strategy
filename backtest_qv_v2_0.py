"""
v2.0 Quality-Value Engine Backtest

This script:
1. Loads price data (30 stocks, 2021-2024)
2. Loads SF1 fundamental data with point-in-time handling
3. Runs QV engine (long-only and long-short)
4. Applies execution smoothing
5. Calculates performance metrics
6. Compares with v1.5 results
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
import json
from datetime import datetime

from data_loader_sf1 import SF1Config, load_sf1_raw, expand_sf1_to_daily
from engines.factor_quality_value_v1 import FactorQVEngineV1

# API Key
SF1_API_KEY = "H6zH4Q2CDr9uTFk9koqJ"

# 30 tickers from v1.x
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# SF1 indicators for QV engine
INDICATORS_QV = [
    # Value
    "pe", "pb", "ps", "evebitda",
    # Quality
    "roe", "ebitdamargin", "de", "currentratio",
]


def load_price_data():
    """Load price data from v1.x"""
    print("Loading price data...")
    df = pd.read_csv("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp500.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    
    # Filter to 30 tickers
    available_tickers = [t for t in TICKERS if t in df.columns]
    df = df[available_tickers]
    
    print(f"Loaded {len(df)} days, {len(df.columns)} tickers")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df


def calculate_metrics(returns: pd.Series, name: str = "Strategy") -> dict:
    """Calculate performance metrics"""
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    win_rate = (returns > 0).sum() / len(returns)
    
    metrics = {
        "name": name,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "days": len(returns),
    }
    
    return metrics


def portfolio_returns_simple(
    prices: pd.DataFrame,
    weights_by_date: dict,
    rebalance_dates: list,
) -> pd.Series:
    """
    Calculate portfolio returns with simple rebalancing.
    
    Args:
        prices: DataFrame with prices (date x ticker)
        weights_by_date: {date: Series(ticker -> weight)}
        rebalance_dates: List of rebalance dates
    
    Returns:
        Series of daily returns
    """
    print(f"\nportfolio_returns_simple debug:")
    print(f"  prices shape: {prices.shape}")
    print(f"  weights_by_date keys: {len(weights_by_date)}")
    print(f"  rebalance_dates: {len(rebalance_dates)}")
    print(f"  First weight date: {list(weights_by_date.keys())[0] if weights_by_date else None}")
    print(f"  First rebalance date: {rebalance_dates[0] if rebalance_dates else None}")
    
    returns_daily = prices.pct_change()
    portfolio_returns = []
    
    current_weights = None
    rebalance_count = 0
    
    for i, date in enumerate(prices.index):
        # Check if rebalance
        if date in weights_by_date:
            current_weights = weights_by_date[date]
            rebalance_count += 1
        
        if current_weights is None:
            continue
        
        # Calculate return
        ret_cross = returns_daily.loc[date]
        common_tickers = current_weights.index.intersection(ret_cross.index)
        
        if len(common_tickers) > 0:
            w = current_weights[common_tickers]
            r = ret_cross[common_tickers]
            port_ret = (w * r).sum()
            portfolio_returns.append({"date": date, "return": port_ret})
    
    print(f"  Rebalances executed: {rebalance_count}")
    print(f"  Portfolio returns collected: {len(portfolio_returns)}")
    
    df_ret = pd.DataFrame(portfolio_returns)
    if df_ret.empty:
        print("  WARNING: Empty portfolio returns!")
        return pd.Series(dtype=float)
    
    df_ret = df_ret.set_index("date")
    return df_ret["return"]


def main():
    print("="*100)
    print("v2.0 QV ENGINE BACKTEST")
    print("="*100)
    
    # 1. Load price data
    prices = load_price_data()
    
    # 2. Load SF1 fundamental data
    print("\n" + "="*100)
    print("LOADING SF1 FUNDAMENTAL DATA")
    print("="*100)
    
    cfg = SF1Config(
        api_key=SF1_API_KEY,
        dimension="ART",
        min_date="2020-01-01",  # Load from 2020 for sufficient history
    )
    
    try:
        sf1_raw = load_sf1_raw(TICKERS, cfg, INDICATORS_QV)
        print(f"\nSF1 raw data shape: {sf1_raw.shape}")
        print(f"Columns: {list(sf1_raw.columns)}")
        
        # Expand to daily
        fundamentals_daily = expand_sf1_to_daily(
            sf1_raw,
            prices.index,
            shift_one_day=True,  # Prevent look-ahead bias
        )
        
        print(f"\nFundamentals daily shape: {fundamentals_daily.shape}")
        print(f"Columns: {list(fundamentals_daily.columns)}")
        
        # Check data availability
        print("\nData availability check:")
        for col in INDICATORS_QV:
            if col in fundamentals_daily.columns:
                non_null = fundamentals_daily[col].notna().sum()
                pct = 100 * non_null / len(fundamentals_daily)
                print(f"  {col}: {non_null}/{len(fundamentals_daily)} ({pct:.1f}%)")
        
    except Exception as e:
        print(f"\nERROR loading SF1 data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Generate rebalance dates (monthly, first 3 days smoothing)
    print("\n" + "="*100)
    print("GENERATING REBALANCE DATES")
    print("="*100)
    
    # Simple monthly rebalancing on first trading day of month
    rebalance_dates = []
    for year in range(prices.index.year.min(), prices.index.year.max() + 1):
        for month in range(1, 13):
            month_dates = prices.index[(prices.index.year == year) & (prices.index.month == month)]
            if len(month_dates) > 0:
                rebalance_dates.append(month_dates[0])
    
    rebalance_dates = sorted(rebalance_dates)
    print(f"Generated {len(rebalance_dates)} rebalance dates")
    print(f"First: {rebalance_dates[0]}, Last: {rebalance_dates[-1]}")
    
    # 4. Run QV engine - Long-Only
    print("\n" + "="*100)
    print("RUNNING QV ENGINE - LONG-ONLY")
    print("="*100)
    
    engine_long_only = FactorQVEngineV1(
        top_quantile=0.2,
        long_gross=1.0,
        short_gross=0.0,
        long_only=True,
    )
    
    try:
        weights_long_only = engine_long_only.build_portfolio(
            fundamentals_daily,
            rebalance_dates,
        )
        
        print(f"Generated weights for {len(weights_long_only)} rebalance dates")
        
        # Calculate returns
        returns_long_only = portfolio_returns_simple(
            prices,
            weights_long_only,
            rebalance_dates,
        )
        
        print(f"Portfolio returns: {len(returns_long_only)} days")
        
        # Calculate metrics
        metrics_long_only = calculate_metrics(returns_long_only, "QV Long-Only")
        
        print("\n" + "="*100)
        print("QV LONG-ONLY RESULTS")
        print("="*100)
        for k, v in metrics_long_only.items():
            if isinstance(v, float):
                if 'return' in k or 'drawdown' in k or 'rate' in k:
                    print(f"{k:20s}: {v:8.2%}")
                else:
                    print(f"{k:20s}: {v:8.2f}")
            else:
                print(f"{k:20s}: {v}")
        
        # Save results
        results = {
            "strategy": "QV_Long_Only_v2_0",
            "date": datetime.now().isoformat(),
            "metrics": metrics_long_only,
            "rebalance_count": len(weights_long_only),
        }
        
        with open("/home/ubuntu/quant-ensemble-strategy/results/v2_0_qv_long_only_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to: results/v2_0_qv_long_only_results.json")
        
    except Exception as e:
        print(f"\nERROR in QV engine: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Compare with v1.5
    print("\n" + "="*100)
    print("COMPARISON WITH v1.5")
    print("="*100)
    
    try:
        with open("/home/ubuntu/quant-ensemble-strategy/results/v1_5_with_costs_results.json") as f:
            v15_results = json.load(f)
        
        v15_metrics = v15_results.get("metrics_net", {})
        
        print(f"\n{'Metric':<20s} {'v1.5':<12s} {'QV v2.0':<12s} {'Diff':<12s}")
        print("-" * 60)
        
        compare_keys = ["sharpe_ratio", "annual_return", "annual_vol", "max_drawdown", "win_rate"]
        for key in compare_keys:
            v15_val = v15_metrics.get(key, 0)
            qv_val = metrics_long_only.get(key, 0)
            diff = qv_val - v15_val
            
            if 'return' in key or 'drawdown' in key or 'rate' in key:
                print(f"{key:<20s} {v15_val:>11.2%} {qv_val:>11.2%} {diff:>+11.2%}")
            else:
                print(f"{key:<20s} {v15_val:>11.2f} {qv_val:>11.2f} {diff:>+11.2f}")
        
    except FileNotFoundError:
        print("v1.5 results not found, skipping comparison")
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
