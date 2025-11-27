#!/usr/bin/env python3
"""
ML XGBoost v7 - Simplified version
- Value proxy inverted (1 / value_proxy)
- Long-only (no short)
- Equal weight (no volatility weighting)
- Leverage limited
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    sharpe: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    win_rate: float
    num_trades: int


class MLXGBoostV7Simple:
    """
    ML XGBoost v7 - Simplified version
    """
    
    def __init__(self, price_data: pd.DataFrame, factor_data: pd.DataFrame,
                 top_quantile: float = 0.2,
                 prediction_horizon: int = 10):
        self.prices = price_data
        self.factors = factor_data.copy()
        
        # Value proxy 반전
        self.factors["value_proxy_inv"] = 1.0 / self.factors["value_proxy"]
        
        self.top_quantile = top_quantile
        self.prediction_horizon = prediction_horizon
        
        # XGBoost 파라미터
        self.xgb_params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 50,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
            "random_state": 42,
        }
        
    def _prepare_ml_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """ML 학습용 데이터셋 준비"""
        dates = sorted(set(self.factors.index.get_level_values("date")))
        dates_in_range = [d for d in dates if start_date <= d <= end_date]
        
        X_list = []
        y_list = []
        
        for date in dates_in_range:
            if date not in self.factors.index.get_level_values("date"):
                continue
            
            factors_at_date = self.factors.loc[date]
            
            # Forward return 계산
            future_date_idx = dates.index(date) + self.prediction_horizon
            if future_date_idx >= len(dates):
                continue
            
            future_date = dates[future_date_idx]
            
            for ticker in factors_at_date.index:
                if ticker not in self.prices.columns:
                    continue
                
                # Features (value_proxy_inv 사용)
                features = {
                    "momentum_60d": factors_at_date.loc[ticker, "momentum_60d"],
                    "value_proxy_inv": factors_at_date.loc[ticker, "value_proxy_inv"],
                    "volatility_30d": factors_at_date.loc[ticker, "volatility_30d"],
                }
                
                # Target (forward return)
                if date in self.prices.index and future_date in self.prices.index:
                    fwd_ret = self.prices.loc[future_date, ticker] / self.prices.loc[date, ticker] - 1.0
                    
                    X_list.append(features)
                    y_list.append(fwd_ret)
        
        if not X_list:
            raise RuntimeError(f"No valid samples in range {start_date} to {end_date}")
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        # NaN/Inf 제거
        valid_mask = ~(X.isna().any(axis=1) | y.isna() | np.isinf(y))
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        if len(X) == 0:
            raise RuntimeError(f"No valid samples after cleaning in range {start_date} to {end_date}")
        
        return X, y
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[xgb.XGBRegressor, StandardScaler]:
        """모델 학습"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBRegressor(**self.xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        return model, scaler
    
    def _predict_scores(self, model: xgb.XGBRegressor, scaler: StandardScaler,
                        factors_at_date: pd.DataFrame) -> pd.Series:
        """특정 날짜의 예측 점수 계산"""
        X = factors_at_date[["momentum_60d", "value_proxy_inv", "volatility_30d"]]
        X_scaled = scaler.transform(X)
        
        predictions = model.predict(X_scaled)
        
        return pd.Series(predictions, index=factors_at_date.index)
    
    def _construct_portfolio(self, predictions: pd.Series) -> Dict[str, float]:
        """
        예측 점수 기반 포트폴리오 구성 (Long-only, 균등 가중)
        - High prediction → Long
        """
        predictions_sorted = predictions.sort_values(ascending=False)
        
        n_stocks = len(predictions_sorted)
        n_long = max(5, int(n_stocks * self.top_quantile))
        
        long_tickers = predictions_sorted.head(n_long).index.tolist()
        
        portfolio = {}
        
        # Long positions (균등 가중)
        if long_tickers:
            weight = 1.0 / len(long_tickers)
            for ticker in long_tickers:
                portfolio[ticker] = weight
        
        return portfolio
    
    def _get_monthly_rebalance_dates(self, start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        """월간 리밸런싱 날짜 생성"""
        dates = self.prices.loc[start:end].index
        monthly_dates = []
        
        current_month = None
        for date in dates:
            if current_month != date.month:
                monthly_dates.append(date)
                current_month = date.month
        
        return monthly_dates
    
    def _calc_metrics(self, returns: pd.Series) -> PerformanceMetrics:
        """성과 지표 계산"""
        returns = returns.fillna(0.0)
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        sharpe = (mean_ret * TRADING_DAYS) / (std_ret * np.sqrt(TRADING_DAYS)) if std_ret > 0 else 0.0
        annual_return = mean_ret * TRADING_DAYS
        annual_vol = std_ret * np.sqrt(TRADING_DAYS)
        
        cum_ret = (1.0 + returns).cumprod()
        peak = cum_ret.cummax()
        dd = cum_ret / peak - 1.0
        max_dd = dd.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return PerformanceMetrics(
            sharpe=float(sharpe),
            annual_return=float(annual_return),
            annual_volatility=float(annual_vol),
            max_drawdown=float(max_dd),
            win_rate=float(win_rate),
            num_trades=len(returns)
        )
    
    def _backtest_period(self, train_start: pd.Timestamp, train_end: pd.Timestamp,
                         test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.Series:
        """Walk-forward 백테스트 (1개 윈도우)"""
        # 학습
        print(f"  학습: {train_start.date()} ~ {train_end.date()}")
        X_train, y_train = self._prepare_ml_dataset(train_start, train_end)
        model, scaler = self._train_model(X_train, y_train)
        
        # Feature importance 출력
        feature_names = ["momentum_60d", "value_proxy_inv", "volatility_30d"]
        importances = model.feature_importances_
        print(f"  Feature importance: {dict(zip(feature_names, importances))}")
        
        # 테스트
        print(f"  테스트: {test_start.date()} ~ {test_end.date()}")
        rebal_dates = self._get_monthly_rebalance_dates(test_start, test_end)
        
        daily_returns = []
        current_portfolio = {}
        
        test_dates = self.prices.loc[test_start:test_end].index
        
        for i, date in enumerate(test_dates):
            # 월간 리밸런싱
            if i > 0 and date in rebal_dates:
                prev_date = test_dates[i-1]
                
                if prev_date in self.factors.index.get_level_values("date"):
                    factors_at_prev = self.factors.loc[prev_date]
                    predictions = self._predict_scores(model, scaler, factors_at_prev)
                    current_portfolio = self._construct_portfolio(predictions)
            
            # 수익률 계산
            if current_portfolio and i > 0:
                prev_date = test_dates[i-1]
                
                daily_ret = 0.0
                for ticker, weight in current_portfolio.items():
                    if ticker in self.prices.columns:
                        ret = self.prices.loc[date, ticker] / self.prices.loc[prev_date, ticker] - 1.0
                        daily_ret += weight * ret
                
                daily_returns.append({"date": date, "ret": daily_ret})
        
        if daily_returns:
            return pd.Series({r["date"]: r["ret"] for r in daily_returns})
        else:
            return pd.Series(dtype=float)
    
    def run_walkforward_backtest(self) -> Dict[str, Any]:
        """Walk-forward backtest"""
        dates = sorted(set(self.factors.index.get_level_values("date")))
        
        train_years = 3
        test_years = 1
        
        start_year = dates[0].year
        end_year = dates[-1].year - test_years
        
        windows = []
        for y in range(start_year + train_years, end_year + 1):
            train_start = pd.Timestamp(year=y-train_years, month=1, day=1)
            train_end = pd.Timestamp(year=y-1, month=12, day=31)
            test_start = pd.Timestamp(year=y, month=1, day=1)
            test_end = pd.Timestamp(year=y+test_years-1, month=12, day=31)
            windows.append((train_start, train_end, test_start, test_end))
        
        all_daily_ret = []
        
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
            print(f"\n[Window {i}/{len(windows)}]")
            daily_ret = self._backtest_period(tr_start, tr_end, te_start, te_end)
            
            if len(daily_ret) > 0:
                all_daily_ret.append(daily_ret)
        
        if not all_daily_ret:
            raise RuntimeError("No valid walk-forward windows.")
        
        daily_ret_all = pd.concat(all_daily_ret).sort_index()
        overall_metrics = self._calc_metrics(daily_ret_all)
        
        return {
            "overall": asdict(overall_metrics),
            "daily_returns": [
                {"date": d.strftime("%Y-%m-%d"), "ret": float(r)}
                for d, r in daily_ret_all.items()
            ],
        }


def main():
    print("=" * 100)
    print("ML XGBoost v7 - Simplified (Value Proxy Inverted + Long-only + Equal Weight)")
    print("=" * 100)
    
    # 데이터 로드
    price_data = pd.read_parquet("data/price_data_sp500.parquet")
    factor_data = pd.read_parquet("data/factors_price_based.parquet")
    
    engine = MLXGBoostV7Simple(price_data, factor_data)
    result = engine.run_walkforward_backtest()
    
    # 저장
    output_path = Path("engine_results/ml_xgboost_v7_simple_oos.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 100)
    print("Overall Performance (Out-of-Sample)")
    print("=" * 100)
    print(f"Sharpe Ratio: {result['overall']['sharpe']:.4f}")
    print(f"Annual Return: {result['overall']['annual_return']*100:.2f}%")
    print(f"Annual Volatility: {result['overall']['annual_volatility']*100:.2f}%")
    print(f"Max Drawdown: {result['overall']['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {result['overall']['win_rate']*100:.2f}%")
    
    print(f"\n✅ 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
