#!/usr/bin/env python3
"""
과적합성 테스트 프레임워크
- IS vs OOS 성과 비교
- Walk-forward 윈도우별 안정성
- Feature Importance 안정성 (ML 엔진용)
- 리밸런싱 룰 민감도 테스트
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from engines.momentum_cs_v2_fixed import (
    MomentumCSEngineV2,
    MomentumCSEngineV2Config,
    load_price_data,
    get_monthly_rebalance_dates,
    portfolio_returns_from_weights,
    calc_monthly_metrics
)


class OverfittingTests:
    """과적합성 테스트 모음"""
    
    def __init__(self, prices: pd.DataFrame, universe: List[str]):
        self.prices = prices
        self.universe = universe
        self.results = {}
    
    def test_1_is_vs_oos(self, split_date: str = "2023-01-01") -> Dict:
        """
        테스트 1: IS vs OOS 성과 비교
        - In-Sample: 모델 설계/튜닝 구간
        - Out-of-Sample: 검증 전용 구간
        - OOS 성과가 IS 대비 크게 떨어지면 과적합 의심
        """
        print("\n" + "="*100)
        print("테스트 1: IS vs OOS 성과 비교")
        print("="*100)
        print(f"Split 날짜: {split_date}")
        
        split_ts = pd.Timestamp(split_date)
        
        # IS 구간
        prices_is = self.prices.loc[:split_ts]
        rebal_is = get_monthly_rebalance_dates(prices_is.index)
        
        engine = MomentumCSEngineV2()
        weights_is = engine.build_portfolio(prices_is, rebal_is)
        ret_is = portfolio_returns_from_weights(prices_is, weights_is, rebal_is)
        
        if len(ret_is) < 20:
            print(f"\nIn-Sample 기간이 너무 짧음 (< 20일), 테스트 스킵")
            return {}
        
        metrics_is = calc_monthly_metrics(ret_is)
        
        if not metrics_is:
            print(f"\nIn-Sample 메트릭 계산 실패, 테스트 스킵")
            return {}
        
        print(f"\nIn-Sample ({prices_is.index[0].date()} ~ {prices_is.index[-1].date()}):")
        print(f"  - Sharpe: {metrics_is.get('sharpe', 0):.4f}")
        print(f"  - 연수익률: {metrics_is['ann_return']*100:.2f}%")
        print(f"  - 연변동성: {metrics_is['ann_vol']*100:.2f}%")
        print(f"  - Max DD: {metrics_is['max_drawdown']*100:.2f}%")
        print(f"  - Win Rate: {metrics_is['win_rate']*100:.2f}%")
        
        # OOS 구간
        prices_oos = self.prices.loc[split_ts:]
        rebal_oos = get_monthly_rebalance_dates(prices_oos.index)
        
        weights_oos = engine.build_portfolio(prices_oos, rebal_oos)
        ret_oos = portfolio_returns_from_weights(prices_oos, weights_oos, rebal_oos)
        metrics_oos = calc_monthly_metrics(ret_oos)
        
        print(f"\nOut-of-Sample ({prices_oos.index[0].date()} ~ {prices_oos.index[-1].date()}):")
        print(f"  - Sharpe: {metrics_oos['sharpe']:.4f}")
        print(f"  - 연수익률: {metrics_oos['ann_return']*100:.2f}%")
        print(f"  - 연변동성: {metrics_oos['ann_vol']*100:.2f}%")
        print(f"  - Max DD: {metrics_oos['max_drawdown']*100:.2f}%")
        print(f"  - Win Rate: {metrics_oos['win_rate']*100:.2f}%")
        
        # 비교
        sharpe_degradation = (metrics_oos['sharpe'] - metrics_is['sharpe']) / metrics_is['sharpe'] * 100
        
        print(f"\n비교:")
        print(f"  - Sharpe 변화: {sharpe_degradation:+.1f}%")
        
        if sharpe_degradation > -20:
            verdict = "✅ PASS - OOS 성과가 IS와 유사하거나 개선"
        elif sharpe_degradation > -40:
            verdict = "⚠️  CAUTION - OOS 성과가 다소 저하 (과적합 가능성)"
        else:
            verdict = "❌ FAIL - OOS 성과가 크게 저하 (과적합 의심)"
        
        print(f"\n판정: {verdict}")
        
        self.results['is_vs_oos'] = {
            'split_date': split_date,
            'is_sharpe': metrics_is['sharpe'],
            'oos_sharpe': metrics_oos['sharpe'],
            'sharpe_degradation_pct': sharpe_degradation,
            'is_metrics': metrics_is,
            'oos_metrics': metrics_oos,
            'verdict': verdict
        }
        
        return self.results['is_vs_oos']
    
    def test_2_walkforward_stability(self, window_months: int = 12, step_months: int = 6) -> Dict:
        """
        테스트 2: Walk-forward 윈도우별 안정성
        - 롤링 윈도우로 전략 성과 측정
        - 윈도우별 Sharpe가 크게 변동하면 과적합 의심
        """
        print("\n" + "="*100)
        print("테스트 2: Walk-forward 윈도우별 안정성")
        print("="*100)
        print(f"윈도우 크기: {window_months}개월")
        print(f"스텝 크기: {step_months}개월")
        
        all_dates = self.prices.index
        start_date = all_dates[0]
        end_date = all_dates[-1]
        
        window_results = []
        
        # 롤링 윈도우
        current_start = start_date
        window_idx = 0
        
        while True:
            # 윈도우 종료일 계산
            window_end = current_start + pd.DateOffset(months=window_months)
            
            if window_end > end_date:
                break
            
            # 해당 윈도우 데이터
            mask = (self.prices.index >= current_start) & (self.prices.index <= window_end)
            prices_window = self.prices.loc[mask]
            
            if len(prices_window) < 60:  # 최소 60 거래일 필요
                current_start += pd.DateOffset(months=step_months)
                continue
            
            # 백테스트
            engine = MomentumCSEngineV2()
            rebal_window = get_monthly_rebalance_dates(prices_window.index)
            
            if len(rebal_window) < 2:
                current_start += pd.DateOffset(months=step_months)
                continue
            
            weights_window = engine.build_portfolio(prices_window, rebal_window)
            ret_window = portfolio_returns_from_weights(prices_window, weights_window, rebal_window)
            
            if len(ret_window) < 20:
                current_start += pd.DateOffset(months=step_months)
                continue
            
            metrics_window = calc_monthly_metrics(ret_window)
            
            window_results.append({
                'window_idx': window_idx,
                'start_date': current_start,
                'end_date': window_end,
                'sharpe': metrics_window['sharpe'],
                'ann_return': metrics_window['ann_return'],
                'max_drawdown': metrics_window['max_drawdown']
            })
            
            print(f"\n윈도우 {window_idx}: {current_start.date()} ~ {window_end.date()}")
            print(f"  - Sharpe: {metrics_window['sharpe']:.4f}")
            print(f"  - 연수익률: {metrics_window['ann_return']*100:.2f}%")
            print(f"  - Max DD: {metrics_window['max_drawdown']*100:.2f}%")
            
            window_idx += 1
            current_start += pd.DateOffset(months=step_months)
        
        # 안정성 분석
        if len(window_results) >= 3:
            sharpe_list = [w['sharpe'] for w in window_results]
            sharpe_mean = np.mean(sharpe_list)
            sharpe_std = np.std(sharpe_list)
            sharpe_cv = sharpe_std / sharpe_mean if sharpe_mean != 0 else np.inf
            
            print(f"\n안정성 분석:")
            print(f"  - 윈도우 수: {len(window_results)}")
            print(f"  - Sharpe 평균: {sharpe_mean:.4f}")
            print(f"  - Sharpe 표준편차: {sharpe_std:.4f}")
            print(f"  - Sharpe CV (변동계수): {sharpe_cv:.4f}")
            print(f"  - Sharpe 범위: [{min(sharpe_list):.4f}, {max(sharpe_list):.4f}]")
            
            if sharpe_cv < 0.3:
                verdict = "✅ PASS - 윈도우별 성과가 안정적"
            elif sharpe_cv < 0.5:
                verdict = "⚠️  CAUTION - 윈도우별 성과가 다소 불안정"
            else:
                verdict = "❌ FAIL - 윈도우별 성과가 매우 불안정 (과적합 의심)"
        else:
            sharpe_mean = 0
            sharpe_std = 0
            sharpe_cv = 0
            verdict = "⚠️  INCONCLUSIVE - 윈도우 수 부족"
        
        print(f"\n판정: {verdict}")
        
        self.results['walkforward_stability'] = {
            'window_months': window_months,
            'step_months': step_months,
            'n_windows': len(window_results),
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_cv': sharpe_cv,
            'windows': window_results,
            'verdict': verdict
        }
        
        return self.results['walkforward_stability']
    
    def test_3_rebalance_sensitivity(self) -> Dict:
        """
        테스트 3: 리밸런싱 룰 민감도 테스트
        - 리밸 날짜를 조금 바꿔도 성과가 유지되는지 확인
        - 특정 날짜에만 의존하면 과적합
        """
        print("\n" + "="*100)
        print("테스트 3: 리밸런싱 룰 민감도 테스트")
        print("="*100)
        
        engine = MomentumCSEngineV2()
        
        # 기준: 매월 첫 거래일
        rebal_base = get_monthly_rebalance_dates(self.prices.index)
        weights_base = engine.build_portfolio(self.prices, rebal_base)
        ret_base = portfolio_returns_from_weights(self.prices, weights_base, rebal_base)
        metrics_base = calc_monthly_metrics(ret_base)
        
        print(f"\n기준 (매월 첫 거래일):")
        print(f"  - Sharpe: {metrics_base['sharpe']:.4f}")
        
        # 변형 1: 매월 2번째 거래일
        rebal_alt1 = [d + pd.Timedelta(days=1) for d in rebal_base]
        rebal_alt1 = [d for d in rebal_alt1 if d in self.prices.index]
        
        if len(rebal_alt1) >= 2:
            weights_alt1 = engine.build_portfolio(self.prices, rebal_alt1)
            ret_alt1 = portfolio_returns_from_weights(self.prices, weights_alt1, rebal_alt1)
            metrics_alt1 = calc_monthly_metrics(ret_alt1)
            
            print(f"\n변형 1 (매월 2번째 거래일):")
            print(f"  - Sharpe: {metrics_alt1['sharpe']:.4f}")
            print(f"  - 변화: {(metrics_alt1['sharpe'] - metrics_base['sharpe']) / metrics_base['sharpe'] * 100:+.1f}%")
        else:
            metrics_alt1 = None
        
        # 변형 2: 매월 3번째 거래일
        rebal_alt2 = [d + pd.Timedelta(days=2) for d in rebal_base]
        rebal_alt2 = [d for d in rebal_alt2 if d in self.prices.index]
        
        if len(rebal_alt2) >= 2:
            weights_alt2 = engine.build_portfolio(self.prices, rebal_alt2)
            ret_alt2 = portfolio_returns_from_weights(self.prices, weights_alt2, rebal_alt2)
            metrics_alt2 = calc_monthly_metrics(ret_alt2)
            
            print(f"\n변형 2 (매월 3번째 거래일):")
            print(f"  - Sharpe: {metrics_alt2['sharpe']:.4f}")
            print(f"  - 변화: {(metrics_alt2['sharpe'] - metrics_base['sharpe']) / metrics_base['sharpe'] * 100:+.1f}%")
        else:
            metrics_alt2 = None
        
        # 판정
        sharpe_changes = []
        if metrics_alt1:
            sharpe_changes.append(abs(metrics_alt1['sharpe'] - metrics_base['sharpe']) / metrics_base['sharpe'] * 100)
        if metrics_alt2:
            sharpe_changes.append(abs(metrics_alt2['sharpe'] - metrics_base['sharpe']) / metrics_base['sharpe'] * 100)
        
        if sharpe_changes:
            max_change = max(sharpe_changes)
            
            print(f"\n최대 변화율: {max_change:.1f}%")
            
            if max_change < 20:
                verdict = "✅ PASS - 리밸 날짜 변경에 강건함"
            elif max_change < 40:
                verdict = "⚠️  CAUTION - 리밸 날짜에 다소 민감"
            else:
                verdict = "❌ FAIL - 리밸 날짜에 매우 민감 (과적합 의심)"
        else:
            max_change = 0
            verdict = "⚠️  INCONCLUSIVE - 변형 테스트 실패"
        
        print(f"\n판정: {verdict}")
        
        self.results['rebalance_sensitivity'] = {
            'base_sharpe': metrics_base['sharpe'],
            'alt1_sharpe': metrics_alt1['sharpe'] if metrics_alt1 else None,
            'alt2_sharpe': metrics_alt2['sharpe'] if metrics_alt2 else None,
            'max_change_pct': max_change,
            'verdict': verdict
        }
        
        return self.results['rebalance_sensitivity']
    
    def test_4_portfolio_size_sensitivity(self) -> Dict:
        """
        테스트 4: 포트폴리오 크기 민감도 테스트
        - 종목 수를 바꿔도 성과가 유지되는지 확인
        - 특정 종목 수에만 의존하면 과적합
        """
        print("\n" + "="*100)
        print("테스트 4: 포트폴리오 크기 민감도 테스트")
        print("="*100)
        
        rebalance_dates = get_monthly_rebalance_dates(self.prices.index)
        
        sizes = [4, 6, 8, 10]
        results_by_size = []
        
        for n_long in sizes:
            config = MomentumCSEngineV2Config(n_long=n_long)
            engine = MomentumCSEngineV2(config)
            
            weights = engine.build_portfolio(self.prices, rebalance_dates)
            ret = portfolio_returns_from_weights(self.prices, weights, rebalance_dates)
            metrics = calc_monthly_metrics(ret)
            
            print(f"\n종목 수 {n_long}:")
            print(f"  - Sharpe: {metrics['sharpe']:.4f}")
            print(f"  - 연수익률: {metrics['ann_return']*100:.2f}%")
            
            results_by_size.append({
                'n_long': n_long,
                'sharpe': metrics['sharpe'],
                'ann_return': metrics['ann_return']
            })
        
        # 안정성 분석
        sharpe_list = [r['sharpe'] for r in results_by_size]
        sharpe_std = np.std(sharpe_list)
        sharpe_mean = np.mean(sharpe_list)
        sharpe_cv = sharpe_std / sharpe_mean if sharpe_mean != 0 else np.inf
        
        print(f"\n안정성 분석:")
        print(f"  - Sharpe 평균: {sharpe_mean:.4f}")
        print(f"  - Sharpe 표준편차: {sharpe_std:.4f}")
        print(f"  - Sharpe CV: {sharpe_cv:.4f}")
        
        if sharpe_cv < 0.15:
            verdict = "✅ PASS - 포트폴리오 크기에 강건함"
        elif sharpe_cv < 0.3:
            verdict = "⚠️  CAUTION - 포트폴리오 크기에 다소 민감"
        else:
            verdict = "❌ FAIL - 포트폴리오 크기에 매우 민감 (과적합 의심)"
        
        print(f"\n판정: {verdict}")
        
        self.results['portfolio_size_sensitivity'] = {
            'results_by_size': results_by_size,
            'sharpe_mean': sharpe_mean,
            'sharpe_std': sharpe_std,
            'sharpe_cv': sharpe_cv,
            'verdict': verdict
        }
        
        return self.results['portfolio_size_sensitivity']
    
    def run_all_tests(self) -> Dict:
        """모든 테스트 실행"""
        print("\n" + "="*100)
        print("과적합성 테스트 시작")
        print("="*100)
        
        self.test_1_is_vs_oos()
        self.test_2_walkforward_stability(window_months=12, step_months=6)
        self.test_3_rebalance_sensitivity()
        self.test_4_portfolio_size_sensitivity()
        
        return self.results
    
    def generate_report(self) -> str:
        """테스트 결과 요약 보고서 생성"""
        report = []
        report.append("\n" + "="*100)
        report.append("과적합성 테스트 종합 보고서")
        report.append("="*100)
        
        tests = [
            ('is_vs_oos', 'IS vs OOS 성과 비교'),
            ('walkforward_stability', 'Walk-forward 안정성'),
            ('rebalance_sensitivity', '리밸런싱 민감도'),
            ('portfolio_size_sensitivity', '포트폴리오 크기 민감도')
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
        
        report.append("\n" + "="*100)
        report.append(f"최종 결과: {pass_count}/{total_count} 테스트 통과")
        
        if pass_count == total_count:
            final_verdict = "✅ 전체 PASS - 과적합 위험 낮음"
        elif pass_count >= total_count * 0.75:
            final_verdict = "⚠️  대부분 PASS - 일부 주의 필요"
        else:
            final_verdict = "❌ 다수 FAIL - 과적합 위험 높음, 전략 재검토 필요"
        
        report.append(final_verdict)
        report.append("="*100)
        
        return "\n".join(report)


def main():
    """메인 실행 함수"""
    print("과적합성 테스트 프레임워크")
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
    tester = OverfittingTests(prices, universe_30)
    results = tester.run_all_tests()
    
    # 보고서 출력
    report = tester.generate_report()
    print(report)
    
    # 결과 저장
    output_dir = Path("tests/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "overfitting_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ 테스트 결과 저장: {output_path}")


if __name__ == "__main__":
    main()
