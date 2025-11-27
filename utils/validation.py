#!/usr/bin/env python3
"""
구조적 룩어헤드 및 과적합 방지 메커니즘
- 데이터 검증 레이어
- 시계열 데이터 처리 가드레일
- 자동 검증 함수
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """검증 결과"""
    passed: bool
    message: str
    details: Optional[Dict] = None


class TimeSeriesValidator:
    """시계열 데이터 검증기 - 룩어헤드 방지"""
    
    @staticmethod
    def validate_factor_computation(
        prices: pd.DataFrame,
        factors: pd.DataFrame,
        sample_dates: List[pd.Timestamp],
        sample_tickers: List[str],
        factor_name: str,
        compute_func: callable
    ) -> ValidationResult:
        """
        팩터 계산 검증 - 수동 재계산과 비교
        
        Args:
            prices: 가격 데이터
            factors: 계산된 팩터 (MultiIndex: date, ticker)
            sample_dates: 검증할 날짜 샘플
            sample_tickers: 검증할 티커 샘플
            factor_name: 팩터 이름
            compute_func: 수동 재계산 함수 (date, ticker, prices) -> float
        
        Returns:
            ValidationResult
        """
        mismatches = []
        
        for date in sample_dates:
            for ticker in sample_tickers:
                try:
                    # 엔진에서 계산된 값
                    engine_value = factors.loc[(date, ticker), factor_name]
                    
                    # 수동 재계산
                    manual_value = compute_func(date, ticker, prices)
                    
                    # 비교
                    diff = abs(engine_value - manual_value)
                    if diff > 1e-6:
                        mismatches.append({
                            'date': date,
                            'ticker': ticker,
                            'engine_value': engine_value,
                            'manual_value': manual_value,
                            'diff': diff
                        })
                except Exception as e:
                    mismatches.append({
                        'date': date,
                        'ticker': ticker,
                        'error': str(e)
                    })
        
        if not mismatches:
            return ValidationResult(
                passed=True,
                message=f"✅ {factor_name} 팩터 계산 검증 통과",
                details={'n_samples': len(sample_dates) * len(sample_tickers)}
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"❌ {factor_name} 팩터 계산 불일치 발견",
                details={'mismatches': mismatches}
            )
    
    @staticmethod
    def validate_no_future_data(
        data: pd.DataFrame,
        date_col: str = None,
        check_date: pd.Timestamp = None
    ) -> ValidationResult:
        """
        미래 데이터 사용 여부 검증
        
        Args:
            data: 검증할 데이터
            date_col: 날짜 컬럼 이름 (None이면 index 사용)
            check_date: 기준 날짜 (None이면 현재 날짜)
        
        Returns:
            ValidationResult
        """
        if date_col is None:
            dates = data.index
        else:
            dates = data[date_col]
        
        if check_date is None:
            check_date = pd.Timestamp.now()
        
        future_dates = dates[dates > check_date]
        
        if len(future_dates) == 0:
            return ValidationResult(
                passed=True,
                message="✅ 미래 데이터 없음",
                details={'check_date': check_date}
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"❌ 미래 데이터 {len(future_dates)}개 발견",
                details={
                    'check_date': check_date,
                    'future_dates': future_dates.tolist()
                }
            )
    
    @staticmethod
    def validate_train_test_split(
        train_dates: pd.DatetimeIndex,
        test_dates: pd.DatetimeIndex
    ) -> ValidationResult:
        """
        Train/Test Split 검증 - 시계열 순서 확인
        
        Args:
            train_dates: 학습 데이터 날짜
            test_dates: 테스트 데이터 날짜
        
        Returns:
            ValidationResult
        """
        train_max = train_dates.max()
        test_min = test_dates.min()
        
        if train_max < test_min:
            return ValidationResult(
                passed=True,
                message="✅ Train/Test Split 순서 정상",
                details={
                    'train_max': train_max,
                    'test_min': test_min
                }
            )
        else:
            overlap = train_dates[train_dates >= test_min]
            return ValidationResult(
                passed=False,
                message=f"❌ Train/Test 날짜 겹침 발견 ({len(overlap)}개)",
                details={
                    'train_max': train_max,
                    'test_min': test_min,
                    'overlap_dates': overlap.tolist()
                }
            )


class OverfittingValidator:
    """과적합 검증기"""
    
    @staticmethod
    def validate_is_oos_consistency(
        is_sharpe: float,
        oos_sharpe: float,
        threshold: float = 0.5
    ) -> ValidationResult:
        """
        IS vs OOS 일관성 검증
        
        Args:
            is_sharpe: In-Sample Sharpe
            oos_sharpe: Out-of-Sample Sharpe
            threshold: 허용 비율 (OOS/IS >= threshold)
        
        Returns:
            ValidationResult
        """
        if is_sharpe <= 0:
            return ValidationResult(
                passed=False,
                message="⚠️  IS Sharpe가 0 이하 - 전략 자체에 문제",
                details={'is_sharpe': is_sharpe, 'oos_sharpe': oos_sharpe}
            )
        
        ratio = oos_sharpe / is_sharpe
        
        if ratio >= threshold:
            return ValidationResult(
                passed=True,
                message=f"✅ IS/OOS 일관성 양호 (비율: {ratio:.2f})",
                details={'is_sharpe': is_sharpe, 'oos_sharpe': oos_sharpe, 'ratio': ratio}
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"❌ OOS 성과 크게 저하 (비율: {ratio:.2f}) - 과적합 의심",
                details={'is_sharpe': is_sharpe, 'oos_sharpe': oos_sharpe, 'ratio': ratio}
            )
    
    @staticmethod
    def validate_parameter_stability(
        param_values: List[float],
        cv_threshold: float = 0.3
    ) -> ValidationResult:
        """
        파라미터 안정성 검증 (윈도우별 변동)
        
        Args:
            param_values: 윈도우별 파라미터 값
            cv_threshold: 변동계수 임계값
        
        Returns:
            ValidationResult
        """
        if len(param_values) < 3:
            return ValidationResult(
                passed=False,
                message="⚠️  샘플 수 부족 (< 3)",
                details={'n_samples': len(param_values)}
            )
        
        mean_val = np.mean(param_values)
        std_val = np.std(param_values)
        cv = std_val / mean_val if mean_val != 0 else np.inf
        
        if cv < cv_threshold:
            return ValidationResult(
                passed=True,
                message=f"✅ 파라미터 안정적 (CV: {cv:.4f})",
                details={'mean': mean_val, 'std': std_val, 'cv': cv}
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"❌ 파라미터 불안정 (CV: {cv:.4f}) - 과적합 의심",
                details={'mean': mean_val, 'std': std_val, 'cv': cv}
            )


class DataProcessingGuardrails:
    """데이터 처리 가드레일 - 룩어헤드 방지"""
    
    @staticmethod
    def safe_rolling_calculation(
        data: pd.Series,
        window: int,
        func: callable,
        min_periods: int = None
    ) -> pd.Series:
        """
        안전한 롤링 계산 - 미래 데이터 사용 방지
        
        Args:
            data: 시계열 데이터
            window: 윈도우 크기
            func: 적용할 함수
            min_periods: 최소 기간
        
        Returns:
            계산 결과
        """
        if min_periods is None:
            min_periods = window
        
        # pandas rolling은 기본적으로 과거 데이터만 사용
        result = data.rolling(window=window, min_periods=min_periods).apply(func)
        
        # 검증: 각 시점의 결과가 해당 시점 이전 데이터만 사용했는지 확인
        # (샘플링으로 검증)
        sample_idx = np.random.choice(range(window, len(data)), size=min(5, len(data)-window), replace=False)
        
        for idx in sample_idx:
            manual_result = func(data.iloc[idx-window:idx])
            if abs(result.iloc[idx] - manual_result) > 1e-6:
                warnings.warn(f"롤링 계산 불일치 발견 at index {idx}")
        
        return result
    
    @staticmethod
    def safe_shift(
        data: pd.DataFrame,
        periods: int,
        fill_value: float = np.nan
    ) -> pd.DataFrame:
        """
        안전한 shift - 방향 명시 및 검증
        
        Args:
            data: 데이터
            periods: shift 기간 (양수: 과거, 음수: 미래)
            fill_value: 채울 값
        
        Returns:
            Shift된 데이터
        """
        if periods < 0:
            warnings.warn(f"⚠️  음수 shift ({periods}) 사용 - 미래 데이터 참조 가능성")
        
        return data.shift(periods, fill_value=fill_value)
    
    @staticmethod
    def safe_cross_sectional_normalize(
        data: pd.DataFrame,
        date_col: str = None,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        안전한 Cross-sectional 정규화 - 같은 날짜 내에서만 정규화
        
        Args:
            data: 데이터 (MultiIndex: date, ticker 또는 date_col 포함)
            date_col: 날짜 컬럼 (None이면 MultiIndex level 0 사용)
            method: 정규화 방법 ('zscore', 'minmax')
        
        Returns:
            정규화된 데이터
        """
        if date_col is None:
            # MultiIndex 가정
            grouped = data.groupby(level=0)
        else:
            grouped = data.groupby(date_col)
        
        if method == 'zscore':
            normalized = grouped.transform(lambda x: (x - x.mean()) / x.std())
        elif method == 'minmax':
            normalized = grouped.transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return normalized


def run_full_validation(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    returns: pd.Series,
    is_sharpe: float,
    oos_sharpe: float
) -> Dict[str, ValidationResult]:
    """
    전체 검증 실행
    
    Args:
        prices: 가격 데이터
        factors: 팩터 데이터
        returns: 전략 수익률
        is_sharpe: In-Sample Sharpe
        oos_sharpe: Out-of-Sample Sharpe
    
    Returns:
        검증 결과 딕셔너리
    """
    results = {}
    
    # 1. 시계열 검증
    ts_validator = TimeSeriesValidator()
    
    # 미래 데이터 사용 여부
    results['no_future_data'] = ts_validator.validate_no_future_data(factors)
    
    # 2. 과적합 검증
    of_validator = OverfittingValidator()
    
    # IS/OOS 일관성
    results['is_oos_consistency'] = of_validator.validate_is_oos_consistency(
        is_sharpe, oos_sharpe
    )
    
    return results


if __name__ == "__main__":
    print("구조적 룩어헤드 및 과적합 방지 메커니즘")
    print("="*100)
    print("이 모듈은 다음 기능을 제공합니다:")
    print("  1. TimeSeriesValidator: 시계열 데이터 검증 (룩어헤드 방지)")
    print("  2. OverfittingValidator: 과적합 검증")
    print("  3. DataProcessingGuardrails: 데이터 처리 가드레일")
    print("\n사용 예시:")
    print("  from utils.validation import TimeSeriesValidator, OverfittingValidator")
    print("  validator = TimeSeriesValidator()")
    print("  result = validator.validate_factor_computation(...)")
