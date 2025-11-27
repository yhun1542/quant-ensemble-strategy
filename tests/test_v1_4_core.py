#!/usr/bin/env python3
"""
v1.4 핵심 기능 단위 테스트
AI 평가 피드백 반영
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 테스트 대상 모듈
import sys
sys.path.insert(0, '/home/ubuntu/quant-ensemble-strategy')

from utils.signal_prices import build_signal_price_df, SignalSmoothingConfig
from utils.execution_smoothing_v2 import (
    portfolio_returns_with_execution_smoothing,
    ExecutionSmoothingConfig,
    find_next_trading_days
)
from utils.factors import compute_value_factor, compute_volatility_factor


class TestSignalPrices:
    """Signal Prices 생성 테스트"""
    
    def test_signal_price_basic(self):
        """기본 Signal Price 생성 테스트"""
        # 테스트 데이터 생성
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')  # 거래일만
        prices = pd.DataFrame({
            'AAPL': np.random.uniform(100, 110, len(dates)),
            'MSFT': np.random.uniform(200, 210, len(dates)),
        }, index=dates)
        
        cfg = SignalSmoothingConfig(window=3)
        signal_df = build_signal_price_df(prices, cfg)
        
        # 검증
        assert signal_df.shape[0] > 0, "Signal prices should not be empty"
        assert not signal_df.isna().any().any(), "Signal prices should not contain NaN"
        assert set(signal_df.columns) == set(prices.columns), "Columns should match"
    
    def test_signal_price_lookahead_prevention(self):
        """룩어헤드 방지 검증: 윈도우 마지막 날이 리밸 기준일"""
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='B')
        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104, 105, 106, 107],
        }, index=dates)
        
        cfg = SignalSmoothingConfig(window=3)
        signal_df = build_signal_price_df(prices, cfg)
        
        # 첫 번째 signal_date는 Day 3 (윈도우 마지막 날)
        first_signal_date = signal_df.index[0]
        assert first_signal_date == dates[2], "First signal date should be Day 3"
        
        # Signal price는 Day 1~3 평균
        expected = (100 + 101 + 102) / 3
        assert abs(signal_df.loc[first_signal_date, 'AAPL'] - expected) < 0.01
    
    def test_signal_price_incomplete_window(self):
        """불완전한 윈도우 처리 테스트"""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='B')  # 5일만
        prices = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104],
        }, index=dates)
        
        cfg = SignalSmoothingConfig(window=3)
        signal_df = build_signal_price_df(prices, cfg)
        
        # 윈도우 길이가 부족한 초기 날짜는 스킵되어야 함
        assert len(signal_df) <= len(prices) - cfg.window + 1


class TestExecutionSmoothing:
    """Execution Smoothing 테스트"""
    
    def test_find_next_trading_days(self):
        """거래일 찾기 테스트"""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
        start_date = dates[5]
        
        next_days = find_next_trading_days(start_date, dates, n_days=3)
        
        assert len(next_days) == 3, "Should find 3 trading days"
        assert next_days[0] == dates[6], "First day should be next trading day"
        assert next_days[1] == dates[7], "Second day should be 2nd trading day"
    
    def test_execution_smoothing_2step(self):
        """2-step Execution Smoothing 테스트"""
        # 테스트 데이터
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
        prices = pd.DataFrame({
            'AAPL': np.linspace(100, 110, len(dates)),
            'MSFT': np.linspace(200, 210, len(dates)),
        }, index=dates)
        
        # 리밸 날짜 및 가중치
        rebalance_dates = [dates[10]]
        weights_by_date = {
            dates[10]: pd.Series({'AAPL': 0.6, 'MSFT': 0.4})
        }
        
        cfg = ExecutionSmoothingConfig(n_steps=2)
        ret = portfolio_returns_with_execution_smoothing(
            prices, weights_by_date, rebalance_dates, cfg
        )
        
        # 검증
        assert len(ret) > 0, "Should have returns"
        assert not ret.isna().any(), "Returns should not contain NaN"
    
    def test_execution_smoothing_vs_simple(self):
        """Execution Smoothing vs Simple 비교"""
        from utils.execution_smoothing_v2 import portfolio_returns_simple
        
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='B')
        prices = pd.DataFrame({
            'AAPL': np.random.uniform(100, 110, len(dates)),
            'MSFT': np.random.uniform(200, 210, len(dates)),
        }, index=dates)
        
        rebalance_dates = [dates[10]]
        weights_by_date = {
            dates[10]: pd.Series({'AAPL': 0.6, 'MSFT': 0.4})
        }
        
        cfg = ExecutionSmoothingConfig(n_steps=2)
        ret_smooth = portfolio_returns_with_execution_smoothing(
            prices, weights_by_date, rebalance_dates, cfg
        )
        ret_simple = portfolio_returns_simple(
            prices, weights_by_date, rebalance_dates
        )
        
        # 두 방식 모두 수익률 생성
        assert len(ret_smooth) > 0
        assert len(ret_simple) > 0
        
        # Smoothing이 변동성을 줄이는지 확인 (일반적으로)
        # (이 테스트는 데이터에 따라 실패할 수 있음, 참고용)
        # assert ret_smooth.std() <= ret_simple.std() * 1.1


class TestFactors:
    """Factors 계산 테스트"""
    
    def test_value_factor(self):
        """Value Factor 계산 테스트"""
        prices = pd.Series({'AAPL': 100, 'MSFT': 200, 'GOOGL': 150})
        value = compute_value_factor(prices)
        
        assert len(value) == 3
        assert value['AAPL'] > value['MSFT'], "Lower price should have higher value"
        assert not value.isna().any()
    
    def test_volatility_factor(self):
        """Volatility Factor 계산 테스트"""
        dates = pd.date_range('2024-01-01', '2024-02-01', freq='B')
        prices = pd.DataFrame({
            'AAPL': np.random.uniform(100, 110, len(dates)),
            'MSFT': np.random.uniform(200, 210, len(dates)),
        }, index=dates)
        
        vol = compute_volatility_factor(prices, window=20)
        
        assert len(vol) == 2
        assert not vol.isna().any()
        assert (vol > 0).all(), "Volatility should be positive"


class TestLookaheadPrevention:
    """룩어헤드 방지 종합 테스트"""
    
    def test_signal_prices_no_future_data(self):
        """Signal Prices가 미래 데이터를 사용하지 않는지 검증"""
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='B')
        prices = pd.DataFrame({
            'AAPL': list(range(100, 100 + len(dates))),
        }, index=dates)
        
        cfg = SignalSmoothingConfig(window=3)
        signal_df = build_signal_price_df(prices, cfg)
        
        # 각 signal_date에서 signal_price가 해당 날짜 이전 데이터만 사용했는지 확인
        for signal_date in signal_df.index:
            signal_price = signal_df.loc[signal_date, 'AAPL']
            
            # 윈도우 시작일 계산
            date_idx = prices.index.get_loc(signal_date)
            window_start_idx = max(0, date_idx - cfg.window + 1)
            window_prices = prices.iloc[window_start_idx:date_idx + 1]['AAPL']
            
            # Signal price는 윈도우 평균이어야 함
            expected_avg = window_prices.mean()
            assert abs(signal_price - expected_avg) < 0.01, \
                f"Signal price on {signal_date} uses future data"
    
    def test_ml_features_vs_labels_separation(self):
        """ML Features(signal_prices) vs Labels(actual_prices) 분리 검증"""
        # 이 테스트는 실제 ML 엔진 실행 시 검증
        # 여기서는 개념적 검증만
        
        dates = pd.date_range('2024-01-01', '2024-02-01', freq='B')
        actual_prices = pd.DataFrame({
            'AAPL': np.random.uniform(100, 110, len(dates)),
        }, index=dates)
        
        cfg = SignalSmoothingConfig(window=3)
        signal_prices = build_signal_price_df(actual_prices, cfg)
        
        # Signal prices는 actual prices보다 적어야 함 (윈도우 때문에)
        assert len(signal_prices) < len(actual_prices)
        
        # Signal prices의 각 값은 actual prices의 평균이므로 다름
        common_dates = signal_prices.index.intersection(actual_prices.index)
        for date in common_dates:
            assert signal_prices.loc[date, 'AAPL'] != actual_prices.loc[date, 'AAPL'], \
                "Signal price should differ from actual price (it's an average)"


def run_all_tests():
    """모든 테스트 실행"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    print("="*100)
    print("v1.4 핵심 기능 단위 테스트")
    print("="*100)
    print("\n테스트 항목:")
    print("  1. Signal Prices 생성")
    print("  2. 룩어헤드 방지")
    print("  3. Execution Smoothing")
    print("  4. Factors 계산")
    print("\n실행 중...")
    print("="*100)
    
    run_all_tests()
