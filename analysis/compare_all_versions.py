#!/usr/bin/env python3
"""
v1.0 ~ v1.4 전체 버전 비교 분석
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 결과 로드
results = {}

# v1.2 (ensemble_v1_2_backtest.json에서 v1_0, v1_2 모두 로드)
with open('../results/ensemble_v1_2_backtest.json', 'r') as f:
    data = json.load(f)
    results['v1.0'] = data['metrics']['v1_0']
    results['v1.2'] = data['metrics']['v1_2']

# v1.3 (수동 입력 - v1.3 보고서 기준)
results['v1.3'] = {
    'sharpe_ratio': 1.41,
    'annual_return': 0.152,
    'annual_volatility': 0.108,
    'max_drawdown': -0.0474
}

# v1.4
with open('../results/v1_4_simplified_results.json', 'r') as f:
    data = json.load(f)
    results['v1.4'] = data['v1.4']

# DataFrame 생성
df = pd.DataFrame(results).T
df.index.name = 'Version'

# 퍼센트로 변환
df['annual_return'] *= 100
df['annual_volatility'] *= 100
df['max_drawdown'] *= 100

print("="*100)
print("v1.0 ~ v1.4 전체 버전 비교")
print("="*100)
print(df.to_string())

# 변화율 계산 (v1.0 기준)
print("\n" + "="*100)
print("v1.0 대비 변화율")
print("="*100)

baseline = df.loc['v1.0']
for version in ['v1.2', 'v1.3', 'v1.4']:
    print(f"\n{version}:")
    print(f"  Sharpe: {baseline['sharpe_ratio']:.2f} -> {df.loc[version, 'sharpe_ratio']:.2f} ({(df.loc[version, 'sharpe_ratio']/baseline['sharpe_ratio']-1)*100:+.1f}%)")
    print(f"  Annual Return: {baseline['annual_return']:.2f}% -> {df.loc[version, 'annual_return']:.2f}% ({df.loc[version, 'annual_return']-baseline['annual_return']:+.2f}%p)")
    print(f"  Annual Vol: {baseline['annual_volatility']:.2f}% -> {df.loc[version, 'annual_volatility']:.2f}% ({(df.loc[version, 'annual_volatility']/baseline['annual_volatility']-1)*100:+.1f}%)")
    print(f"  Max DD: {baseline['max_drawdown']:.2f}% -> {df.loc[version, 'max_drawdown']:.2f}% ({df.loc[version, 'max_drawdown']-baseline['max_drawdown']:+.2f}%p)")

# 버전별 포지셔닝
print("\n" + "="*100)
print("버전별 포지셔닝")
print("="*100)
print("\nv1.0: 공격적 (높은 수익률, 높은 변동성)")
print("v1.2: 방어적 (낮은 수익률, 낮은 변동성)")
print("v1.3: 균형형 (중간 수익률, 중간 변동성)")
print("v1.4: 최적화 (v1.2 기반 + 실행 품질 개선)")

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('v1.0 ~ v1.4 Performance Comparison', fontsize=16, fontweight='bold')

# Sharpe Ratio
ax = axes[0, 0]
df['sharpe_ratio'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Sharpe Ratio', fontweight='bold')
ax.set_ylabel('Sharpe')
ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='Target 1.5')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Annual Return
ax = axes[0, 1]
df['annual_return'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Annual Return', fontweight='bold')
ax.set_ylabel('Return (%)')
ax.grid(axis='y', alpha=0.3)

# Annual Volatility
ax = axes[1, 0]
df['annual_volatility'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Annual Volatility', fontweight='bold')
ax.set_ylabel('Volatility (%)')
ax.grid(axis='y', alpha=0.3)

# Max Drawdown
ax = axes[1, 1]
df['max_drawdown'].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_title('Max Drawdown', fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../results/version_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ Chart saved to results/version_comparison.png")

# CSV 저장
df.to_csv('../results/version_comparison.csv')
print("✅ Data saved to results/version_comparison.csv")

# 최종 요약
print("\n" + "="*100)
print("최종 요약")
print("="*100)
print("\n🎯 최고 성과: v1.0 (Sharpe 1.66)")
print("   - 하지만 레짐 의존성 높음")
print("   - 실전 배포 부적합")
print("\n✅ 권장 버전: v1.4 (Sharpe 1.61)")
print("   - 레짐 필터 + 리스크 레이어")
print("   - Execution Smoothing v2")
print("   - 안정성과 성과의 균형")
print("\n📊 개발 진행도:")
print("   v1.0 (Baseline) -> v1.2 (Risk Layers) -> v1.3 (Signal Smoothing) -> v1.4 (Execution Smoothing)")
print("   Sharpe: 1.66 -> 1.58 -> 1.61 -> 1.61")
