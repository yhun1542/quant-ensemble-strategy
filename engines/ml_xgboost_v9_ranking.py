#!/usr/bin/env python3
"""
ML XGBoost v9 - Cross-sectional ranking version
- Cross-sectional z-score normalization (relative ranking within each date)
- Quantile-based target (top/bottom 20% classification)
- Long-only strategy
- Enhanced XGBoost parameters
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


class MLXGBoostV9Ranking:
    """
    ML XGBoost v9 - Cross-sectional ranking version
    """
    
    def __init__(self, price_data: pd.DataFrame, factor_data: pd.DataFrame,
                 top_quantile: float = 0.2,
                 prediction_horizon: int = 10):
        self.prices = price_data
        self.factors = factor_data.copy()
        
        # Value proxy 반전
        self.factors["value_proxy_inv"] = 1.0 / self.factors["value_proxy"]
        
        # Cross-sectional ranking (z-score normalization)
        self._apply_cross_sectional_ranking()
        
        self.top_quantile = top_quantile
        self.prediction_horizon = prediction_horizon
        
        # XGBoost 파라미터 - 분류 문제로 변경
        self.xgb_params = {
            "objective": "multi:softprob",  # 다중 분류
            "num_class": 3,  # Top(2), Middle(1), Bottom(0)
            "max_depth": 5,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 1.0,
            "reg_lambda": 3.0,
            "random_state": 42,
        }
        
    def _apply_cross_sectional_ranking(self):
        """날짜별 cross-sectional z-score 정규화"""
        dates = sorted(set(self.factors.index.get_level_values("date")))
        
        for col in ["momentum_60d", "value_proxy_inv", "volatility_30d"]:
            ranked_values = []
            
            for date in dates:
                if date not in self.factors.index.get_level_values("date"):
                    continue
                
                factors_at_date = self.factors.loc[date, col]
                
                # Z-score 정규화 (날짜 내)
                mean = factors_at_date.mean()
                std = factors_at_date.std()
                
                if std > 0:
                    z_scores = (factors_at_date - mean) / std
                else:
                    z_scores = factors_at_date * 0
                
                for ticker in factors_at_date.index:
                    ranked_values.append({
                        "date": date,
                        "ticker": ticker,
                        f"{col}_rank": z_scores.loc[ticker]
                    })
            
            if ranked_values:
                rank_df = pd.DataFrame(ranked_values).set_index(["date", "ticker"])
                self.factors = self.factors.join(rank_df, how="left")
        
        print("✅ Cross-sectional ranking applied")
        
    def _prepare_ml_dataset(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[pd.DataFrame, pd.Series]:
        """ML 학습용 데이터셋 준비 (quantile-based target)"""
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
            
            # 날짜별 forward return 수집
            fwd_rets_at_date = {}
            for ticker in factors_at_date.index:
                if ticker not in self.prices.columns:
                    continue
                
                if date in self.prices.index and future_date in self.prices.index:
                    fwd_ret = self.prices.loc[future_date, ticker] / self.prices.loc[date, ticker] - 1.0
                    fwd_rets_at_date[ticker] = fwd_ret
            
            if not fwd_rets_at_date:
                continue
            
            # Quantile 계산 (날짜 내)
            fwd_rets_series = pd.Series(fwd_rets_at_date)
            q_low = fwd_rets_series.quantile(self.top_quantile)
            q_high = fwd_rets_series.quantile(1 - self.top_quantile)
            
            # 각 종목에 대해 target 할당
            for ticker in factors_at_date.index:
                if ticker not in fwd_rets_at_date:
                    continue
                
                fwd_ret = fwd_rets_at_date[ticker]
                
                # Target: 0 (Bottom 20%), 1 (Middle 60%), 2 (Top 20%)
                if fwd_ret <= q_low:
                    target = 0  # Bottom
                elif fwd_ret >= q_high:
                    target = 2  # Top
                else:
                    target = 1  # Middle
                
                # Features (ranked)
                features = {
                    "momentum_60d_rank": factors_at_date.loc[ticker, "momentum_60d_rank"],
                    "value_proxy_inv_rank": factors_at_date.loc[ticker, "value_proxy_inv_rank"],
                    "volatility_30d_rank": factors_at_date.loc[ticker, "volatility_30d_rank"],
                }
                
                X_list.append(features)
                y_list.append(target)
        
        if not X_list:
            raise RuntimeError(f"No valid samples in range {start_date} to {end_date}")
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        # NaN/Inf 제거
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)
        
        if len(X) == 0:
            raise RuntimeError(f"No valid samples after cleaning in range {start_date} to {end_date}")
        
        print(f"  샘플 수: {len(X)}, Target 분포: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[xgb.XGBClassifier, StandardScaler]:
        """모델 학습 (분류)"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = xgb.XGBClassifier(**self.xgb_params)
        model.fit(X_train_scaled, y_train, verbose=False)
        
        return model, scaler
    
    def _predict_scores(self, model: xgb.XGBClassifier, scaler: StandardScaler,
                        factors_at_date: pd.DataFrame) -> pd.Series:
        """특정 날짜의 예측 점수 계산"""
        feature_cols = ["momentum_60d_rank", "value_proxy_inv_rank", "volatility_30d_rank"]
        
        X = factors_at_date[feature_cols]
        X_scaled = scaler.transform(X)
        
        # 확률 예측 (Top class 확률 사용)
        proba = model.predict_proba(X_scaled)
        top_proba = proba[:, 2]  # Class 2 (Top) 확률
        
        return pd.Series(top_proba, index=factors_at_date.index)
    
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
        print(f"  Feature importance: {dict(zip(X_train.columns, model.feature_importances_))}")
        
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
    print("ML XGBoost v9 - Cross-sectional Ranking + Quantile Target")
    print("=" * 100)
    
    # 데이터 로드
    price_data = pd.read_parquet("data/price_data_sp500.parquet")
    factor_data = pd.read_parquet("data/factors_price_based.parquet")
    
    engine = MLXGBoostV9Ranking(price_data, factor_data)
    result = engine.run_walkforward_backtest()
    
    # 저장
    output_path = Path("engine_results/ml_xgboost_v9_ranking_oos.json")
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
