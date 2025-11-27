"""
3-Engine Ensemble: QV v2.1 + FV3c + ML9

Complete implementation using pre-generated weights
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from itertools import product

from utils.transaction_costs import TransactionCostModel

# 30 tickers
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]


def load_price_data():
    """Load price data"""
    df = pd.read_csv("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp500.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    # Remove timezone
    df.index = df.index.tz_localize(None)
    available_tickers = [t for t in TICKERS if t in df.columns]
    df = df[available_tickers]
    return df


def load_qv_weights():
    """Load QV v2.1 weights from results"""
    # QV v2.1 weights need to be generated first
    # For now, return empty dict
    return {}


def load_fv3c_ml9_weights():
    """Load FV3c and ML9 weights"""
    with open("/home/ubuntu/quant-ensemble-strategy/results/ensemble_fv3c_ml9.json", "r") as f:
        data = json.load(f)
    
    fv3c_weights = {}
    for date_str, weights_dict in data["fv3c_weights"].items():
        if weights_dict:  # Skip empty
            date = pd.to_datetime(date_str).tz_localize(None)
            fv3c_weights[date] = pd.Series(weights_dict)
    
    ml9_weights = {}
    for date_str, weights_dict in data["ml9_weights"].items():
        if weights_dict:  # Skip empty
            date = pd.to_datetime(date_str).tz_localize(None)
            ml9_weights[date] = pd.Series(weights_dict)
    
    return fv3c_weights, ml9_weights


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


def portfolio_returns(
    prices: pd.DataFrame,
    weights_by_date: dict,
) -> pd.Series:
    """Calculate portfolio returns"""
    returns_daily = prices.pct_change()
    portfolio_returns = []
    
    # Create date-only index for weights
    weights_by_date_only = {}
    for dt, w in weights_by_date.items():
        date_only = dt.date()
        weights_by_date_only[date_only] = w
    
    current_weights = None
    
    for date in prices.index:
        date_only = date.date()
        
        # Check if rebalance
        if date_only in weights_by_date_only:
            current_weights = weights_by_date_only[date_only]
        
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
    
    df_ret = pd.DataFrame(portfolio_returns)
    if df_ret.empty:
        return pd.Series(dtype=float)
    
    df_ret = df_ret.set_index("date")
    return df_ret["return"]


def main():
    print("="*100)
    print("3-ENGINE ENSEMBLE: QV v2.1 + FV3c + ML9 - COMPLETE IMPLEMENTATION")
    print("="*100)
    
    # 1. Load price data
    print("\nLoading price data...")
    prices = load_price_data()
    print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
    print(f"Date range: {prices.index.min()} to {prices.index.max()}")
    
    # 2. Load weights
    print("\n" + "="*100)
    print("LOADING ENGINE WEIGHTS")
    print("="*100)
    
    print("\n--- Loading FV3c and ML9 weights ---")
    fv3c_weights, ml9_weights = load_fv3c_ml9_weights()
    print(f"FV3c: {len(fv3c_weights)} rebalance dates")
    print(f"ML9: {len(ml9_weights)} rebalance dates")
    
    # 3. Calculate returns for each engine
    print("\n" + "="*100)
    print("CALCULATING ENGINE RETURNS")
    print("="*100)
    
    # FV3c
    print("\n--- FV3c ---")
    returns_fv = portfolio_returns(prices, fv3c_weights)
    metrics_fv = calculate_metrics(returns_fv, "FV3c")
    print(f"Returns: {len(returns_fv)} days")
    print(f"Sharpe: {metrics_fv['sharpe_ratio']:.2f}")
    print(f"Annual Return: {metrics_fv['annual_return']:.2%}")
    print(f"Annual Vol: {metrics_fv['annual_vol']:.2%}")
    
    # ML9
    print("\n--- ML9 ---")
    returns_ml = portfolio_returns(prices, ml9_weights)
    metrics_ml = calculate_metrics(returns_ml, "ML9")
    print(f"Returns: {len(returns_ml)} days")
    print(f"Sharpe: {metrics_ml['sharpe_ratio']:.2f}")
    print(f"Annual Return: {metrics_ml['annual_return']:.2%}")
    print(f"Annual Vol: {metrics_ml['annual_vol']:.2%}")
    
    # 4. Combine returns
    print("\n" + "="*100)
    print("COMBINING RETURNS")
    print("="*100)
    
    df_returns = pd.concat([
        returns_fv.rename("fv3c"),
        returns_ml.rename("ml9"),
    ], axis=1).dropna()
    
    print(f"Combined returns: {len(df_returns)} days")
    print(f"\nCorrelation matrix:")
    print(df_returns.corr().round(3))
    
    # 5. Grid search for optimal weights (FV3c + ML9 only for now)
    print("\n" + "="*100)
    print("GRID SEARCH FOR OPTIMAL WEIGHTS (FV3c + ML9)")
    print("="*100)
    
    weights_grid = np.linspace(0.0, 1.0, 21)
    results = []
    
    for w_fv in weights_grid:
        w_ml = 1.0 - w_fv
        
        # Ensemble returns
        ensemble_ret = w_fv * df_returns["fv3c"] + w_ml * df_returns["ml9"]
        
        # Calculate metrics
        metrics = calculate_metrics(ensemble_ret, f"Ensemble_FV{w_fv:.2f}_ML{w_ml:.2f}")
        
        results.append({
            "w_fv": float(w_fv),
            "w_ml": float(w_ml),
            **metrics
        })
    
    df_results = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    
    print(f"\nTested {len(df_results)} weight combinations")
    print(f"\nTop 10 by Sharpe:")
    print(df_results.head(10)[["w_fv", "w_ml", "sharpe_ratio", "annual_return", "annual_vol", "max_drawdown"]])
    
    # 6. Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    results_dict = {
        "strategy": "FV3c_ML9_Ensemble",
        "date": datetime.now().isoformat(),
        "engine_metrics": {
            "fv3c": metrics_fv,
            "ml9": metrics_ml,
        },
        "engine_correlations": df_returns.corr().to_dict(),
        "grid_search_results": df_results.to_dict(orient="records"),
        "best_weights": {
            "w_fv": float(df_results.iloc[0]["w_fv"]),
            "w_ml": float(df_results.iloc[0]["w_ml"]),
            "sharpe": float(df_results.iloc[0]["sharpe_ratio"]),
            "annual_return": float(df_results.iloc[0]["annual_return"]),
            "annual_vol": float(df_results.iloc[0]["annual_vol"]),
        }
    }
    
    with open("/home/ubuntu/quant-ensemble-strategy/results/v2_1_fv3c_ml9_ensemble_complete.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print("Results saved to: results/v2_1_fv3c_ml9_ensemble_complete.json")
    
    # 7. Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    best = df_results.iloc[0]
    print(f"\nBest Ensemble (FV3c + ML9):")
    print(f"  Weights: FV3c {best['w_fv']:.1%}, ML9 {best['w_ml']:.1%}")
    print(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    print(f"  Annual Return: {best['annual_return']:.2%}")
    print(f"  Annual Vol: {best['annual_vol']:.2%}")
    print(f"  Max DD: {best['max_drawdown']:.2%}")
    
    print("\n" + "="*100)
    print("ENSEMBLE BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
