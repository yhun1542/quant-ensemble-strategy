#!/usr/bin/env python3
"""
ML XGBoost v10 - Signal Smoothing 지원
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import xgboost as xgb


@dataclass
class ML10Config:
    """ML10 설정"""
    top_quantile: float = 0.2
    prediction_horizon: int = 10
    xgb_params: dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 5,
                "learning_rate": 0.05,
                "n_estimators": 200,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 1.0,
                "reg_lambda": 3.0,
                "random_state": 42,
            }


class MLXGBoostV10:
    """
    ML XGBoost v10 - Signal Smoothing 지원
    
    - Features는 signal prices로 계산
    - Labels는 실제 prices로 계산
    - Cross-sectional ranking
    - Quantile-based classification
    """
    
    def __init__(self, cfg: ML10Config | None = None):
        self.cfg = cfg or ML10Config()
        self.model = None
    
    def _apply_cross_sectional_ranking(
        self,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """날짜별 cross-sectional z-score 정규화"""
        factors = factors.copy()
        dates = sorted(set(factors.index.get_level_values("date")))
        
        for col in ["momentum_60d", "value_proxy_inv", "volatility_30d"]:
            if col not in factors.columns:
                continue
            
            ranked_values = []
            
            for date in dates:
                if date not in factors.index.get_level_values("date"):
                    continue
                
                factors_at_date = factors.loc[date, col]
                
                # Z-score 정규화
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
                factors = factors.join(rank_df, how="left")
        
        return factors
    
    def _prepare_ml_dataset(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        ML 학습용 데이터셋 준비
        
        Parameters
        ----------
        prices : pd.DataFrame
            실제 가격 (labels 계산용)
        factors : pd.DataFrame
            팩터 데이터 (features, signal prices 기반)
        start_date, end_date : pd.Timestamp
            데이터 범위
        
        Returns
        -------
        tuple
            (X, y) - Features와 Labels
        """
        dates = sorted(set(factors.index.get_level_values("date")))
        dates_in_range = [d for d in dates if start_date <= d <= end_date]
        
        X_list = []
        y_list = []
        
        for date in dates_in_range:
            if date not in factors.index.get_level_values("date"):
                continue
            
            factors_at_date = factors.loc[date]
            
            # Forward return 계산 (실제 prices 사용)
            future_date_idx = dates.index(date) + self.cfg.prediction_horizon
            if future_date_idx >= len(dates):
                continue
            
            future_date = dates[future_date_idx]
            
            # 날짜별 forward return 수집
            fwd_rets_at_date = {}
            for ticker in factors_at_date.index:
                if ticker not in prices.columns:
                    continue
                
                if date in prices.index and future_date in prices.index:
                    fwd_ret = prices.loc[future_date, ticker] / prices.loc[date, ticker] - 1.0
                    fwd_rets_at_date[ticker] = fwd_ret
            
            if not fwd_rets_at_date:
                continue
            
            # Quantile 계산
            fwd_rets_series = pd.Series(fwd_rets_at_date)
            q_low = fwd_rets_series.quantile(self.cfg.top_quantile)
            q_high = fwd_rets_series.quantile(1 - self.cfg.top_quantile)
            
            # 각 종목에 대해 target 할당
            for ticker in factors_at_date.index:
                if ticker not in fwd_rets_at_date:
                    continue
                
                fwd_ret = fwd_rets_at_date[ticker]
                
                # Target: 0 (Bottom 20%), 1 (Middle 60%), 2 (Top 20%)
                if fwd_ret <= q_low:
                    target = 0
                elif fwd_ret >= q_high:
                    target = 2
                else:
                    target = 1
                
                # Features (ranking 사용)
                feature_cols = ["momentum_60d_rank", "value_proxy_inv_rank", "volatility_30d_rank"]
                features = factors_at_date.loc[ticker, feature_cols].values
                
                if not np.any(np.isnan(features)):
                    X_list.append(features)
                    y_list.append(target)
        
        if not X_list:
            return pd.DataFrame(), pd.Series(dtype=int)
        
        X = pd.DataFrame(X_list, columns=feature_cols)
        y = pd.Series(y_list)
        
        return X, y
    
    def train(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ):
        """
        모델 학습
        
        Parameters
        ----------
        prices : pd.DataFrame
            실제 가격
        factors : pd.DataFrame
            팩터 데이터 (signal prices 기반)
        train_start, train_end : pd.Timestamp
            학습 기간
        """
        # Ranking 적용
        factors_ranked = self._apply_cross_sectional_ranking(factors)
        
        # 데이터셋 준비
        X_train, y_train = self._prepare_ml_dataset(
            prices, factors_ranked, train_start, train_end
        )
        
        if len(X_train) == 0:
            raise ValueError("No training data available")
        
        # 모델 학습
        self.model = xgb.XGBClassifier(**self.cfg.xgb_params)
        self.model.fit(X_train, y_train)
        
        self.factors_ranked = factors_ranked
    
    def build_portfolio(
        self,
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        rebalance_dates: list[pd.Timestamp],
    ) -> Dict[pd.Timestamp, pd.Series]:
        """
        포트폴리오 구성
        
        Parameters
        ----------
        prices : pd.DataFrame
            실제 가격
        factors : pd.DataFrame
            팩터 데이터 (signal prices 기반)
        rebalance_dates : list
            리밸런싱 날짜
        
        Returns
        -------
        dict
            {date: Series(ticker -> weight)}
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Ranking 적용
        factors_ranked = self._apply_cross_sectional_ranking(factors)
        
        weights_by_date = {}
        
        for date in rebalance_dates:
            if date not in factors_ranked.index.get_level_values("date"):
                continue
            
            factors_at_date = factors_ranked.loc[date]
            
            # Features 준비
            feature_cols = ["momentum_60d_rank", "value_proxy_inv_rank", "volatility_30d_rank"]
            X = factors_at_date[feature_cols]
            
            # 예측
            proba = self.model.predict_proba(X)  # (n_stocks, 3)
            top_proba = proba[:, 2]  # Top class 확률
            
            # Top 20% 선택
            n_stocks = len(factors_at_date)
            n_long = int(n_stocks * self.cfg.top_quantile)
            
            top_indices = np.argsort(top_proba)[-n_long:]
            selected_tickers = factors_at_date.index[top_indices]
            
            # Equal weight
            if len(selected_tickers) > 0:
                weights = pd.Series(
                    1.0 / len(selected_tickers),
                    index=selected_tickers
                )
                weights_by_date[date] = weights
        
        return weights_by_date


if __name__ == "__main__":
    print("="*100)
    print("ML XGBoost v10 - Signal Smoothing 지원")
    print("="*100)
    print("\n사용 예시:")
    print("  from engines.ml_xgboost_v10_signal_smoothing import MLXGBoostV10, ML10Config")
    print("  from utils.factors import compute_all_factors")
    print("  from utils.signal_prices import build_signal_price_df")
    print("")
    print("  # Signal prices 생성")
    print("  signal_df = build_signal_price_df(prices, cfg)")
    print("  signal_df = signal_df.reindex(prices.index).ffill()")
    print("")
    print("  # Factors 계산 (signal prices 사용)")
    print("  factors = compute_all_factors(prices, signal_df)")
    print("")
    print("  # ML10 엔진 학습 및 실행")
    print("  engine = MLXGBoostV10(ML10Config())")
    print("  engine.train(prices, factors, train_start, train_end)")
    print("  weights = engine.build_portfolio(prices, factors, rebalance_dates)")
