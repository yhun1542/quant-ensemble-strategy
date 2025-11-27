#!/usr/bin/env python3
"""
룩어헤드 바이어스 테스트 프레임워크
- 라벨 셔플 테스트
- Train/Test 날짜 뒤집기 테스트
- 예측 Horizon 변경 테스트
- 날짜-티커 수동 검증
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from engines.momentum_cs_v1 import (
    MomentumCSEngineV1,
    MomentumCSEngineV1Config,
    load_price_data,
    get_monthly_rebalance_dates,
    portfolio_returns_from_weights,
    calc_monthly_metrics
)


class LookaheadBiasTests:
    """룩어헤드 바이어스 테스트 모음"""
    
    def __init__(self, prices: pd.DataFrame, universe: List[str]):
        self.prices = prices
        self.universe = universe
        self.results = {}
    
    def test_1_label_shuffle(self, n_trials: int = 10) -> Dict:
        """
        테스트 1: 라벨 셔플 테스트
        - 라벨을 랜덤으로 섞었을 때 성능이 0 근처로 떨어지는지 확인
        - 룩어헤드가 없으면 셔플 시 Sharpe ≈ 0
        """
        print("\n" + "="*100)
        print("테스트 1: 라벨 셔플 테스트")
        print("="*100)
        print(f"시행 횟수: {n_trials}")
        
        # 원본 전략 실행
        engine = MomentumCSEngineV1()
        rebalance_dates = get_monthly_rebalance_dates(self.prices.index)
        weights_by_date = engine.build_portfolio(self.prices, rebalance_dates)
        ret_original = portfolio_returns_from_weights(self.prices, weights_by_date, rebalance_dates)
        metrics_original = calc_monthly_metrics(ret_original)
        
        print(f"\n원본 전략 Sharpe: {metrics_original['sharpe']:.4f}")
        
        # 셔플 테스트
        sharpe_shuffled_list = []
        
        for trial in range(n_trials):
            # 가중치를 날짜 기준으로 셔플
            dates_list = list(weights_by_date.keys())
            weights_list = list(weights_by_date.values())
            
            np.random.seed(trial)
            shuffled_indices = np.random.permutation(len(dates_list))
            
            weights_shuffled = {
                dates_list[i]: weights_list[shuffled_indices[i]]
                for i in range(len(dates_list))
            }
            
            ret_shuffled = portfolio_returns_from_weights(self.prices, weights_shuffled, rebalance_dates)
            metrics_shuffled = calc_monthly_metrics(ret_shuffled)
            sharpe_shuffled_list.append(metrics_shuffled['sharpe'])
        
        sharpe_shuffled_mean = np.mean(sharpe_shuffled_list)
        sharpe_shuffled_std = np.std(sharpe_shuffled_list)
        
        print(f"\n셔플 테스트 결과:")
        print(f"  - 평균 Sharpe: {sharpe_shuffled_mean:.4f}")
        print(f"  - 표준편차: {sharpe_shuffled_std:.4f}")
        print(f"  - 최소값: {min(sharpe_shuffled_list):.4f}")
        print(f"  - 최대값: {max(sharpe_shuffled_list):.4f}")
        
        # Z-score 계산
        if sharpe_shuffled_std > 0:
            z_score = (metrics_original['sharpe'] - sharpe_shuffled_mean) / sharpe_shuffled_std
            print(f"\nZ-score: {z_score:.2f}")
            
            if z_score >= 3.0:
                verdict = "✅ PASS - 원본 전략이 랜덤 대비 유의미하게 우수 (z >= 3.0)"
            elif z_score >= 2.0:
                verdict = "⚠️  CAUTION - 원본 전략이 랜덤보다 우수하지만 신뢰도 중간 (2.0 <= z < 3.0)"
            else:
                verdict = "❌ FAIL - 원본 전략이 랜덤과 큰 차이 없음 (z < 2.0) - 룩어헤드 의심"
        else:
            verdict = "⚠️  INCONCLUSIVE - 셔플 결과 표준편차가 0"
        
        print(f"\n판정: {verdict}")
        
        self.results['label_shuffle'] = {
            'original_sharpe': metrics_original['sharpe'],
            'shuffled_mean': sharpe_shuffled_mean,
            'shuffled_std': sharpe_shuffled_std,
            'z_score': z_score if sharpe_shuffled_std > 0 else None,
            'verdict': verdict
        }
        
        return self.results['label_shuffle']
    
    def test_2_reverse_time(self) -> Dict:
        """
        테스트 2: Train/Test 날짜 뒤집기 테스트
        - 미래로 학습하고 과거로 테스트했을 때 성능이 떨어지는지 확인
        - 룩어헤드가 없으면 뒤집었을 때 성능 급락
        """
        print("\n" + "="*100)
        print("테스트 2: Train/Test 날짜 뒤집기 테스트")
        print("="*100)
        
        # 전체 기간을 반으로 나누기
        all_dates = self.prices.index
        mid_idx = len(all_dates) // 2
        mid_date = all_dates[mid_idx]
        
        print(f"\n전체 기간: {all_dates[0].date()} ~ {all_dates[-1].date()}")
        print(f"중간 날짜: {mid_date.date()}")
        
        # 정방향: 전반부 학습 → 후반부 테스트
        print(f"\n정방향 테스트:")
        print(f"  - 학습 기간: {all_dates[0].date()} ~ {mid_date.date()}")
        print(f"  - 테스트 기간: {mid_date.date()} ~ {all_dates[-1].date()}")
        
        prices_train = self.prices.loc[:mid_date]
        prices_test = self.prices.loc[mid_date:]
        
        engine = MomentumCSEngineV1()
        rebal_test = get_monthly_rebalance_dates(prices_test.index)
        weights_test = engine.build_portfolio(prices_test, rebal_test)
        ret_forward = portfolio_returns_from_weights(prices_test, weights_test, rebal_test)
        metrics_forward = calc_monthly_metrics(ret_forward)
        
        print(f"  → Sharpe: {metrics_forward['sharpe']:.4f}")
        
        # 역방향: 후반부 학습 → 전반부 테스트
        print(f"\n역방향 테스트:")
        print(f"  - 학습 기간: {mid_date.date()} ~ {all_dates[-1].date()}")
        print(f"  - 테스트 기간: {all_dates[0].date()} ~ {mid_date.date()}")
        
        rebal_train = get_monthly_rebalance_dates(prices_train.index)
        weights_train = engine.build_portfolio(prices_train, rebal_train)
        ret_reverse = portfolio_returns_from_weights(prices_train, weights_train, rebal_train)
        metrics_reverse = calc_monthly_metrics(ret_reverse)
        
        print(f"  → Sharpe: {metrics_reverse['sharpe']:.4f}")
        
        # 판정
        sharpe_diff = abs(metrics_forward['sharpe'] - metrics_reverse['sharpe'])
        sharpe_ratio = metrics_reverse['sharpe'] / metrics_forward['sharpe'] if metrics_forward['sharpe'] != 0 else 0
        
        print(f"\n비교:")
        print(f"  - Sharpe 차이: {sharpe_diff:.4f}")
        print(f"  - 역방향/정방향 비율: {sharpe_ratio:.2%}")
        
        if sharpe_ratio < 0.5:
            verdict = "✅ PASS - 역방향 성능이 크게 떨어짐 (정상)"
        elif sharpe_ratio < 0.8:
            verdict = "⚠️  CAUTION - 역방향 성능이 다소 떨어짐"
        else:
            verdict = "❌ FAIL - 역방향 성능이 유사함 - 룩어헤드 의심"
        
        print(f"\n판정: {verdict}")
        
        self.results['reverse_time'] = {
            'forward_sharpe': metrics_forward['sharpe'],
            'reverse_sharpe': metrics_reverse['sharpe'],
            'sharpe_diff': sharpe_diff,
            'sharpe_ratio': sharpe_ratio,
            'verdict': verdict
        }
        
        return self.results['reverse_time']
    
    def test_3_horizon_extension(self, original_horizon: int = 252, 
                                  extended_horizon: int = 504) -> Dict:
        """
        테스트 3: 예측 Horizon 변경 테스트
        - Horizon을 늘렸을 때 성능이 떨어지는지 확인
        - 룩어헤드가 없으면 horizon 늘릴수록 성능 저하
        """
        print("\n" + "="*100)
        print("테스트 3: 예측 Horizon 변경 테스트")
        print("="*100)
        print(f"원본 Horizon: {original_horizon}일 (약 12개월)")
        print(f"확장 Horizon: {extended_horizon}일 (약 24개월)")
        
        # 원본 horizon (기본 252일 lookback)
        engine_original = MomentumCSEngineV1(
            MomentumCSEngineV1Config(lookback_long=original_horizon)
        )
        rebalance_dates = get_monthly_rebalance_dates(self.prices.index)
        weights_original = engine_original.build_portfolio(self.prices, rebalance_dates)
        ret_original = portfolio_returns_from_weights(self.prices, weights_original, rebalance_dates)
        metrics_original = calc_monthly_metrics(ret_original)
        
        print(f"\n원본 Horizon Sharpe: {metrics_original['sharpe']:.4f}")
        
        # 확장 horizon
        engine_extended = MomentumCSEngineV1(
            MomentumCSEngineV1Config(lookback_long=extended_horizon)
        )
        weights_extended = engine_extended.build_portfolio(self.prices, rebalance_dates)
        ret_extended = portfolio_returns_from_weights(self.prices, weights_extended, rebalance_dates)
        metrics_extended = calc_monthly_metrics(ret_extended)
        
        print(f"확장 Horizon Sharpe: {metrics_extended['sharpe']:.4f}")
        
        # 판정
        sharpe_change = metrics_extended['sharpe'] - metrics_original['sharpe']
        sharpe_change_pct = (sharpe_change / metrics_original['sharpe'] * 100) if metrics_original['sharpe'] != 0 else 0
        
        print(f"\n변화:")
        print(f"  - Sharpe 변화: {sharpe_change:+.4f} ({sharpe_change_pct:+.1f}%)")
        
        if sharpe_change_pct < -20:
            verdict = "✅ PASS - Horizon 확장 시 성능 크게 저하 (정상)"
        elif sharpe_change_pct < -5:
            verdict = "⚠️  CAUTION - Horizon 확장 시 성능 다소 저하"
        else:
            verdict = "❌ FAIL - Horizon 확장해도 성능 유지 - 룩어헤드 의심"
        
        print(f"\n판정: {verdict}")
        
        self.results['horizon_extension'] = {
            'original_sharpe': metrics_original['sharpe'],
            'extended_sharpe': metrics_extended['sharpe'],
            'sharpe_change': sharpe_change,
            'sharpe_change_pct': sharpe_change_pct,
            'verdict': verdict
        }
        
        return self.results['horizon_extension']
    
    def test_4_manual_verification(self, n_samples: int = 5) -> Dict:
        """
        테스트 4: 날짜-티커 수동 검증
        - 랜덤 샘플에 대해 팩터를 수동으로 재계산하여 비교
        """
        print("\n" + "="*100)
        print("테스트 4: 날짜-티커 수동 검증")
        print("="*100)
        print(f"샘플 수: {n_samples}")
        
        engine = MomentumCSEngineV1()
        factors = engine.compute_momentum_factors(self.prices)
        
        # 랜덤 샘플 선택
        np.random.seed(42)
        available_dates = factors.index.get_level_values(0).unique()
        sample_dates = np.random.choice(available_dates[252:], size=min(n_samples, len(available_dates)-252), replace=False)
        
        print(f"\n검증 샘플:")
        verification_results = []
        
        for date in sample_dates:
            ticker = np.random.choice(self.universe)
            
            try:
                # 엔진에서 계산된 값
                factor_value = factors.loc[(date, ticker), 'mom_252_ex_21']
                
                # 수동 재계산
                price_hist = self.prices.loc[:date, ticker]
                if len(price_hist) >= 252:
                    p_t_minus_21 = price_hist.iloc[-21]
                    p_t_minus_252 = price_hist.iloc[-252]
                    manual_value = p_t_minus_21 / p_t_minus_252 - 1.0
                    
                    diff = abs(factor_value - manual_value)
                    match = diff < 1e-6
                    
                    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                    print(f"  - {date_str} | {ticker:5s} | 엔진: {factor_value:8.4f} | 수동: {manual_value:8.4f} | 차이: {diff:.2e} | {'✓' if match else '✗'}")
                    
                    verification_results.append({
                        'date': date,
                        'ticker': ticker,
                        'engine_value': factor_value,
                        'manual_value': manual_value,
                        'diff': diff,
                        'match': match
                    })
            except Exception as e:
                date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
                print(f"  - {date_str} | {ticker:5s} | 오류: {str(e)}")
        
        # 판정
        if verification_results:
            match_rate = sum(r['match'] for r in verification_results) / len(verification_results)
            avg_diff = np.mean([r['diff'] for r in verification_results])
            
            print(f"\n검증 결과:")
            print(f"  - 일치율: {match_rate*100:.1f}%")
            print(f"  - 평균 차이: {avg_diff:.2e}")
            
            if match_rate >= 0.95:
                verdict = "✅ PASS - 수동 계산과 일치"
            elif match_rate >= 0.8:
                verdict = "⚠️  CAUTION - 일부 불일치 발견"
            else:
                verdict = "❌ FAIL - 수동 계산과 불일치 - 룩어헤드 의심"
        else:
            verdict = "⚠️  INCONCLUSIVE - 검증 샘플 부족"
            match_rate = 0
            avg_diff = 0
        
        print(f"\n판정: {verdict}")
        
        self.results['manual_verification'] = {
            'n_samples': len(verification_results),
            'match_rate': match_rate,
            'avg_diff': avg_diff,
            'verdict': verdict
        }
        
        return self.results['manual_verification']
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("\n" + "="*100)
        print("룩어헤드 바이어스 테스트 시작")
        print("="*100)
        
        self.test_1_label_shuffle(n_trials=10)
        self.test_2_reverse_time()
        self.test_3_horizon_extension()
        self.test_4_manual_verification(n_samples=5)
        
        return self.results
    
    def generate_report(self) -> str:
        """테스트 결과 요약 보고서 생성"""
        report = []
        report.append("\n" + "="*100)
        report.append("룩어헤드 바이어스 테스트 종합 보고서")
        report.append("="*100)
        
        # 각 테스트 결과 요약
        tests = [
            ('label_shuffle', '라벨 셔플 테스트'),
            ('reverse_time', 'Train/Test 뒤집기 테스트'),
            ('horizon_extension', 'Horizon 변경 테스트'),
            ('manual_verification', '수동 검증 테스트')
        ]
        
        pass_count = 0
        total_count = len(tests)
        
        for test_key, test_name in tests:
            if test_key in self.results:
                result = self.results[test_key]
                verdict = result['verdict']
                
                if '✅ PASS' in verdict:
                    pass_count += 1
                
                report.append(f"\n{test_name}: {verdict}")
        
        # 최종 판정
        report.append("\n" + "="*100)
        report.append(f"최종 결과: {pass_count}/{total_count} 테스트 통과")
        
        if pass_count == total_count:
            final_verdict = "✅ 전체 PASS - 룩어헤드 바이어스 없음"
        elif pass_count >= total_count * 0.75:
            final_verdict = "⚠️  대부분 PASS - 일부 주의 필요"
        else:
            final_verdict = "❌ 다수 FAIL - 룩어헤드 바이어스 의심, 코드 재검토 필요"
        
        report.append(final_verdict)
        report.append("="*100)
        
        return "\n".join(report)


def main():
    """메인 실행 함수"""
    print("룩어헤드 바이어스 테스트 프레임워크")
    print("="*100)
    
    # 30종목 유니버스
    universe_30 = [
        "AAPL", "ABBV", "ACN", "ADBE", "AMZN", "AVGO", "COST", "CVX", "DIS", "GOOGL",
        "HD", "JNJ", "JPM", "KO", "LLY", "MA", "META", "MRK", "MSFT", "NFLX",
        "NKE", "NVDA", "PEP", "PG", "TMO", "TSLA", "UNH", "V", "WMT", "XOM",
    ]
    
    # 데이터 로딩
    prices = load_price_data("data/price_data_sp500.csv", universe_30)
    print(f"\n데이터 로딩 완료:")
    print(f"  - 기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    print(f"  - 거래일 수: {len(prices)}")
    print(f"  - 종목 수: {len(prices.columns)}")
    
    # 테스트 실행
    tester = LookaheadBiasTests(prices, universe_30)
    results = tester.run_all_tests()
    
    # 보고서 출력
    report = tester.generate_report()
    print(report)
    
    # 결과 저장
    output_dir = Path("tests/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "lookahead_bias_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ 테스트 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
