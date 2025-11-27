#!/usr/bin/env python3
"""
v1.4 전략 백테스트
Signal Smoothing + Execution Smoothing + FV4 + ML10 앙상블
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils.signal_prices import (
    SignalSmoothingConfig,
    build_signal_price_df,
    expand_signal_prices,
    get_rebalance_dates_from_signal_df,
)
from utils.factors import compute_all_factors
from engines.factor_value_v4_signal_smoothing import FactorValueV4, FV4Config
from engines.ml_xgboost_v10_signal_smoothing import MLXGBoostV10, ML10Config
from utils.execution_smoothing import (
    portfolio_returns_simple,
    portfolio_returns_with_execution_smoothing,
    ExecutionSmoothingConfig,
)
from utils.risk_overlay import (
    apply_risk_overlays,
    VolTargetConfig,
    DrawdownConfig,
    RegimeConfig,
    RegimeExposureConfig,
)


def load_price_data(data_dir: Path) -> pd.DataFrame:
    """가격 데이터 로드"""
    prices_df = pd.read_csv(data_dir / "prices_30stocks.csv", index_col=0, parse_dates=True)
    return prices_df


def load_spx_close(data_dir: Path) -> pd.Series:
    """S&P 500 종가 로드"""
    spx_df = pd.read_csv(data_dir / "spx_close.csv", index_col=0, parse_dates=True)
    return spx_df["SPX"]


def calc_metrics(ret_daily: pd.Series) -> dict:
    """성과 지표 계산"""
    ret_daily = ret_daily.fillna(0.0)
    
    if len(ret_daily) < 2:
        return {
            "sharpe": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "n_days": 0,
        }
    
    mean_ret = ret_daily.mean()
    std_ret = ret_daily.std()
    
    sharpe = (mean_ret * 252) / (std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
    annual_return = mean_ret * 252
    annual_vol = std_ret * np.sqrt(252)
    
    cum_ret = (1.0 + ret_daily).cumprod()
    peak = cum_ret.cummax()
    dd = cum_ret / peak - 1.0
    max_dd = dd.min()
    
    win_rate = (ret_daily > 0).sum() / len(ret_daily) if len(ret_daily) > 0 else 0.0
    
    return {
        "sharpe": float(sharpe),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "n_days": len(ret_daily),
    }


def main():
    print("="*100)
    print("v1.4 전략 백테스트")
    print("Signal Smoothing + Execution Smoothing + FV4 + ML10 앙상블")
    print("="*100)
    
    # 설정
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    signal_cfg = SignalSmoothingConfig(window=3)
    fv4_cfg = FV4Config(top_quantile=0.2)
    ml10_cfg = ML10Config(top_quantile=0.2, prediction_horizon=10)
    exec_cfg = ExecutionSmoothingConfig(n_steps=2)
    
    # 1) 가격 데이터 로드
    print("\n[1/8] 가격 데이터 로딩...")
    prices = load_price_data(data_dir)
    spx_close = load_spx_close(data_dir)
    print(f"  Prices: {len(prices)} days, {len(prices.columns)} stocks")
    print(f"  Period: {prices.index[0]} ~ {prices.index[-1]}")
    
    # 2) Signal prices 생성
    print("\n[2/8] Signal prices 생성...")
    signal_df_raw = build_signal_price_df(prices, signal_cfg)
    signal_df = expand_signal_prices(signal_df_raw, prices.index)
    rebalance_dates = get_rebalance_dates_from_signal_df(signal_df_raw)
    print(f"  Signal prices: {len(signal_df_raw)} months")
    print(f"  Rebalance dates: {len(rebalance_dates)}")
    
    # 3) Factors 계산
    print("\n[3/8] Factors 계산 (signal prices 기준)...")
    factors = compute_all_factors(prices, signal_df)
    print(f"  Factors shape: {factors.shape}")
    print(f"  Columns: {list(factors.columns)}")
    
    # 4) FV4 엔진 실행
    print("\n[4/8] FV4 엔진 실행...")
    fv4_engine = FactorValueV4(fv4_cfg)
    weights_fv4 = fv4_engine.build_portfolio(prices, factors, rebalance_dates)
    print(f"  FV4 weights: {len(weights_fv4)} dates")
    
    # 5) ML10 엔진 실행
    print("\n[5/8] ML10 엔진 실행...")
    # Train/Test split
    train_start = prices.index[0]
    train_end = pd.Timestamp("2023-06-01")
    test_start = pd.Timestamp("2023-06-01")
    test_end = prices.index[-1]
    
    print(f"  Train: {train_start} ~ {train_end}")
    print(f"  Test: {test_start} ~ {test_end}")
    
    ml10_engine = MLXGBoostV10(ml10_cfg)
    ml10_engine.train(prices, factors, train_start, train_end)
    print("  ML10 model trained")
    
    weights_ml10 = ml10_engine.build_portfolio(prices, factors, rebalance_dates)
    print(f"  ML10 weights: {len(weights_ml10)} dates")
    
    # 6) 수익률 계산 (Execution Smoothing 적용)
    print("\n[6/8] 수익률 계산...")
    
    # FV4 수익률
    ret_fv4 = portfolio_returns_with_execution_smoothing(
        prices, weights_fv4, rebalance_dates, exec_cfg
    )
    print(f"  FV4 returns: {len(ret_fv4)} days")
    
    # ML10 수익률
    ret_ml10 = portfolio_returns_with_execution_smoothing(
        prices, weights_ml10, rebalance_dates, exec_cfg
    )
    print(f"  ML10 returns: {len(ret_ml10)} days")
    
    # 7) 앙상블 (60:40)
    print("\n[7/8] 앙상블 (FV4 60% + ML10 40%)...")
    ret_ensemble = 0.6 * ret_fv4 + 0.4 * ret_ml10
    ret_ensemble = ret_ensemble.fillna(0.0)
    print(f"  Ensemble returns: {len(ret_ensemble)} days")
    
    # 8) 리스크 레이어 적용
    print("\n[8/8] 리스크 레이어 적용...")
    
    regime_cfg = RegimeConfig(ma_window=200)
    regime_exp_cfg = RegimeExposureConfig(bull=1.0, sideways=0.5, bear=0.25)
    vol_cfg = VolTargetConfig(target_vol=0.10, lookback=60)
    dd_cfg = DrawdownConfig(dd_threshold=-0.10, reduce_to=0.5)
    
    result = apply_risk_overlays(
        ret_ensemble,
        spx_close,
        regime_cfg,
        regime_exp_cfg,
        vol_cfg,
        dd_cfg,
    )
    
    ret_final = result["ret_final"]
    print(f"  Final returns: {len(ret_final)} days")
    
    # 성과 계산
    print("\n" + "="*100)
    print("성과 지표")
    print("="*100)
    
    metrics_fv4 = calc_metrics(ret_fv4)
    metrics_ml10 = calc_metrics(ret_ml10)
    metrics_ensemble = calc_metrics(ret_ensemble)
    metrics_final = calc_metrics(ret_final)
    
    print("\nFV4 엔진:")
    print(f"  Sharpe: {metrics_fv4['sharpe']:.4f}")
    print(f"  Annual Return: {metrics_fv4['annual_return']:.2%}")
    print(f"  Annual Vol: {metrics_fv4['annual_volatility']:.2%}")
    print(f"  Max DD: {metrics_fv4['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics_fv4['win_rate']:.2%}")
    
    print("\nML10 엔진:")
    print(f"  Sharpe: {metrics_ml10['sharpe']:.4f}")
    print(f"  Annual Return: {metrics_ml10['annual_return']:.2%}")
    print(f"  Annual Vol: {metrics_ml10['annual_volatility']:.2%}")
    print(f"  Max DD: {metrics_ml10['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics_ml10['win_rate']:.2%}")
    
    print("\n앙상블 (60:40):")
    print(f"  Sharpe: {metrics_ensemble['sharpe']:.4f}")
    print(f"  Annual Return: {metrics_ensemble['annual_return']:.2%}")
    print(f"  Annual Vol: {metrics_ensemble['annual_volatility']:.2%}")
    print(f"  Max DD: {metrics_ensemble['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics_ensemble['win_rate']:.2%}")
    
    print("\n최종 (리스크 레이어 적용):")
    print(f"  Sharpe: {metrics_final['sharpe']:.4f}")
    print(f"  Annual Return: {metrics_final['annual_return']:.2%}")
    print(f"  Annual Vol: {metrics_final['annual_volatility']:.2%}")
    print(f"  Max DD: {metrics_final['max_drawdown']:.2%}")
    print(f"  Win Rate: {metrics_final['win_rate']:.2%}")
    
    # 결과 저장
    print("\n" + "="*100)
    print("결과 저장")
    print("="*100)
    
    output = {
        "config": {
            "signal_smoothing": {"window": signal_cfg.window},
            "execution_smoothing": {"n_steps": exec_cfg.n_steps},
            "fv4": {"top_quantile": fv4_cfg.top_quantile},
            "ml10": {
                "top_quantile": ml10_cfg.top_quantile,
                "prediction_horizon": ml10_cfg.prediction_horizon,
            },
            "ensemble": {"fv4_weight": 0.6, "ml10_weight": 0.4},
        },
        "metrics": {
            "fv4": metrics_fv4,
            "ml10": metrics_ml10,
            "ensemble": metrics_ensemble,
            "final": metrics_final,
        },
        "daily_returns": {
            "fv4": ret_fv4.to_dict(),
            "ml10": ret_ml10.to_dict(),
            "ensemble": ret_ensemble.to_dict(),
            "final": ret_final.to_dict(),
        },
    }
    
    output_path = results_dir / "v1_4_backtest.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"✅ 결과 저장: {output_path}")
    print("\n" + "="*100)
    print("v1.4 백테스트 완료!")
    print("="*100)


if __name__ == "__main__":
    main()
