"""
3-Engine Ensemble: QV + FV3c + ML9

Goal: Find optimal weights to achieve Sharpe 1.5-2.0
"""

import sys
sys.path.append('/home/ubuntu/quant-ensemble-strategy')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from itertools import product

from data_loader_sf1 import SF1Config, load_sf1_raw, expand_sf1_to_daily
from engines.factor_quality_value_v2_1 import FactorQVEngineV21
from engines.factor_value_v3c_dynamic import FactorValueV3cDynamic
from engines.ml_xgboost_v9_ranking import MLXGBoostV9Ranking
from utils.transaction_costs import TransactionCostModel

# API Key
SF1_API_KEY = "H6zH4Q2CDr9uTFk9koqJ"

# 30 tickers
TICKERS = [
    "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
    "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM"
]

# SF1 indicators
INDICATORS_QV = [
    "pe", "pb", "ps", "evebitda",
    "roe", "ebitdamargin", "de", "currentratio",
]


def load_price_data():
    """Load price data"""
    df = pd.read_csv("/home/ubuntu/quant-ensemble-strategy/data/price_data_sp500.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    available_tickers = [t for t in TICKERS if t in df.columns]
    df = df[available_tickers]
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


def portfolio_returns(
    prices: pd.DataFrame,
    weights_by_date: dict,
) -> pd.Series:
    """Calculate portfolio returns"""
    returns_daily = prices.pct_change()
    portfolio_returns = []
    
    current_weights = None
    
    for date in prices.index:
        # Check if rebalance
        if date in weights_by_date:
            current_weights = weights_by_date[date]
        
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
    print("3-ENGINE ENSEMBLE: QV + FV3c + ML9")
    print("="*100)
    
    # 1. Load price data
    print("\nLoading price data...")
    prices = load_price_data()
    print(f"Loaded {len(prices)} days, {len(prices.columns)} tickers")
    
    # 2. Generate rebalance dates
    rebalance_dates = []
    for year in range(prices.index.year.min(), prices.index.year.max() + 1):
        for month in range(1, 13):
            month_dates = prices.index[(prices.index.year == year) & (prices.index.month == month)]
            if len(month_dates) > 0:
                rebalance_dates.append(month_dates[0])
    
    rebalance_dates = sorted(rebalance_dates)
    print(f"Generated {len(rebalance_dates)} rebalance dates")
    
    # 3. Generate returns for each engine
    print("\n" + "="*100)
    print("GENERATING ENGINE RETURNS")
    print("="*100)
    
    # ===== QV Engine =====
    print("\n--- QV v2.1 (Inverse-Vol Top40) ---")
    
    cfg = SF1Config(
        api_key=SF1_API_KEY,
        dimension="ART",
        min_date="2020-01-01",
    )
    
    sf1_raw = load_sf1_raw(TICKERS, cfg, INDICATORS_QV)
    fundamentals_daily = expand_sf1_to_daily(
        sf1_raw,
        prices.index,
        shift_one_day=True,
    )
    
    engine_qv = FactorQVEngineV21(
        top_quantile=0.4,
        long_gross=1.0,
        short_gross=0.0,
        long_only=True,
        use_inverse_vol=True,
        vol_lookback=63,
    )
    
    weights_qv = engine_qv.build_portfolio(fundamentals_daily, prices, rebalance_dates)
    returns_qv_gross = portfolio_returns(prices, weights_qv)
    
    tc_model = TransactionCostModel(commission_bps=0.5, spread_bps=5.0, impact_bps=3.0)
    returns_qv = tc_model.apply_costs_to_returns(returns_qv_gross, weights_qv, list(weights_qv.keys()))
    
    print(f"QV returns: {len(returns_qv)} days")
    
    # ===== FV3c Engine =====
    print("\n--- FV3c (Factor Value v3c Dynamic) ---")
    
    engine_fv = FactorValueV3cDynamic(
        top_quantile=0.2,
        long_gross=1.0,
        short_gross=0.0,
        long_only=True,
    )
    
    # FV3c uses price-based factors
    weights_fv = engine_fv.build_portfolio(prices, rebalance_dates)
    returns_fv_gross = portfolio_returns(prices, weights_fv)
    returns_fv = tc_model.apply_costs_to_returns(returns_fv_gross, weights_fv, list(weights_fv.keys()))
    
    print(f"FV3c returns: {len(returns_fv)} days")
    
    # ===== ML9 Engine =====
    print("\n--- ML9 (XGBoost v9 Ranking) ---")
    
    engine_ml = MLXGBoostV9Ranking(
        top_quantile=0.2,
        long_gross=1.0,
        short_gross=0.0,
        long_only=True,
        lookback_years=2,
    )
    
    weights_ml = engine_ml.build_portfolio(prices, rebalance_dates)
    returns_ml_gross = portfolio_returns(prices, weights_ml)
    returns_ml = tc_model.apply_costs_to_returns(returns_ml_gross, weights_ml, list(weights_ml.keys()))
    
    print(f"ML9 returns: {len(returns_ml)} days")
    
    # 4. Combine returns
    print("\n" + "="*100)
    print("COMBINING RETURNS")
    print("="*100)
    
    df_returns = pd.concat([
        returns_qv.rename("qv"),
        returns_fv.rename("fv3c"),
        returns_ml.rename("ml9"),
    ], axis=1).dropna()
    
    print(f"Combined returns: {len(df_returns)} days")
    print(f"\nCorrelation matrix:")
    print(df_returns.corr().round(3))
    
    # 5. Grid search for optimal weights
    print("\n" + "="*100)
    print("GRID SEARCH FOR OPTIMAL WEIGHTS")
    print("="*100)
    
    weights_grid = np.linspace(0.0, 1.0, 11)
    results = []
    
    for w_qv, w_fv in product(weights_grid, weights_grid):
        w_ml = 1.0 - w_qv - w_fv
        
        if w_ml < 0 or w_ml > 1:
            continue
        
        # Ensemble returns
        ensemble_ret = (
            w_qv * df_returns["qv"] +
            w_fv * df_returns["fv3c"] +
            w_ml * df_returns["ml9"]
        )
        
        # Calculate metrics
        metrics = calculate_metrics(ensemble_ret, f"Ensemble_{w_qv:.1f}_{w_fv:.1f}_{w_ml:.1f}")
        
        results.append({
            "w_qv": float(w_qv),
            "w_fv": float(w_fv),
            "w_ml": float(w_ml),
            **metrics
        })
    
    df_results = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)
    
    print(f"\nTested {len(df_results)} weight combinations")
    print(f"\nTop 10 by Sharpe:")
    print(df_results.head(10)[["w_qv", "w_fv", "w_ml", "sharpe_ratio", "annual_return", "annual_vol", "max_drawdown"]])
    
    # 6. Save results
    print("\n" + "="*100)
    print("SAVING RESULTS")
    print("="*100)
    
    results_dict = {
        "strategy": "QV_FV3c_ML9_Ensemble_v2_1",
        "date": datetime.now().isoformat(),
        "engine_correlations": df_returns.corr().to_dict(),
        "grid_search_results": df_results.to_dict(orient="records"),
        "best_weights": {
            "w_qv": float(df_results.iloc[0]["w_qv"]),
            "w_fv": float(df_results.iloc[0]["w_fv"]),
            "w_ml": float(df_results.iloc[0]["w_ml"]),
            "sharpe": float(df_results.iloc[0]["sharpe_ratio"]),
        }
    }
    
    with open("/home/ubuntu/quant-ensemble-strategy/results/v2_1_qv_fv3c_ml9_ensemble.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print("Results saved to: results/v2_1_qv_fv3c_ml9_ensemble.json")
    
    print("\n" + "="*100)
    print("ENSEMBLE BACKTEST COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
