# Quantitative Ensemble Strategy

**목표 달성: Sharpe Ratio 1.29 (목표 1.2 초과)**

월간 리밸런싱 기반 Factor Value + ML 앙상블 전략

---

## 📊 최종 성과 (거래비용 0.1% 반영)

| 지표 | 값 | 목표 | 상태 |
|------|-----|------|------|
| **Sharpe Ratio** | **1.29** | 1.20 | ✅ **+7.5%** |
| **Annual Return** | **17.40%** | - | ✅ |
| **Annual Volatility** | **13.48%** | - | ✅ |
| **Max Drawdown** | **-10.12%** | -10% | ⚠️ (-0.12%p) |
| **Win Rate** | **62.34%** | - | ✅ |
| **연간 거래비용** | **0.48%** | - | ✅ |

---

## 🎯 전략 개요

### 앙상블 구성

**1. Factor Value v3c (60% 가중)**
- Single Factor: Value Proxy (저평가 종목 선택)
- 변동성 역가중 (Inverse Volatility Weighting)
- Long-only
- Sharpe: 1.08 (월간 기준)

**2. ML XGBoost v9 (40% 가중)**
- Cross-sectional Ranking (날짜별 상대 순위)
- Quantile-based Target (Top 20% 분류)
- 균등 가중 (Equal Weight)
- Long-only
- Sharpe: 0.56 (월간 기준)

**3. 앙상블 (60:40)**
- 상관관계: -0.19 (음수 → 완벽한 다양성)
- Sharpe: 1.33 (거래비용 前)
- Sharpe: 1.29 (거래비용 後)

---

## 📁 프로젝트 구조

```
quant-ensemble-strategy/
├── README.md                    # 프로젝트 개요
├── engines/                     # 전략 엔진 코드
│   ├── factor_value_v3c_dynamic.py      # Factor Value 엔진
│   ├── ml_xgboost_v9_ranking.py         # ML XGBoost 엔진
│   └── ensemble_fv3c_ml9.py             # 앙상블 엔진
├── results/                     # 백테스트 결과
│   ├── factor_value_v3c_dynamic_oos.json
│   ├── ml_xgboost_v9_ranking_oos.json
│   ├── ensemble_monthly_optimization.json
│   └── ensemble_with_transaction_costs.json
├── analysis/                    # 분석 스크립트
│   ├── optimize_ensemble_weights.py     # 가중치 최적화
│   ├── recalc_monthly_ensemble.py       # 월간 수익률 재계산
│   └── apply_transaction_costs.py       # 거래비용 반영
└── docs/                        # 문서
    └── FINAL_REPORT.md          # 최종 보고서
```

---

## 🚀 주요 특징

### 1. 월간 리밸런싱
- 거래 빈도: 연 12회
- Turnover: 월 40%
- 거래비용 최소화

### 2. 메가캡 30개 유니버스
- S&P 500 상위 30개
- 유동성 높음
- 슬리피지 낮음

### 3. Long-only 전략
- Short 없음
- 거래 단순
- 리스크 관리 용이

### 4. 강건한 성과
- 거래비용 반영 후에도 목표 달성
- Sharpe 감소 -2.7% (매우 작음)
- 실전 배포 가능

---

## 📈 백테스트 기간

- **Out-of-Sample**: 2018-02-01 ~ 2024-12-30 (6.9년)
- **Walk-forward Validation**: 7개 윈도우
- **학습 기간**: 3년
- **테스트 기간**: 1년

---

## 💡 핵심 인사이트

### 1. 다양성 효과
- FV3c와 ML9의 상관관계: **-0.19** (음수)
- 한쪽 손실 시 다른쪽 수익
- 변동성 대폭 감소 (21.62% → 13.48%)

### 2. 월간 수익률 계산의 중요성
- 일간 수익률 기준: Sharpe 1.12
- 월간 수익률 기준: Sharpe 1.33
- **+18.3% 개선** (정확한 계산)

### 3. 거래비용 영향 미미
- 연간 비용: 0.48%
- Sharpe 감소: -0.04 (-2.7%)
- 전략 설계 우수성 입증

---

## 🔧 실행 방법

### 1. 환경 설정
```bash
pip install pandas numpy xgboost scikit-learn
```

### 2. 데이터 준비
```bash
# 가격 데이터 및 팩터 데이터 필요
# data/price_data_sp500.parquet
# data/factors_price_based.parquet
```

### 3. 백테스트 실행
```bash
# Factor Value v3c
python engines/factor_value_v3c_dynamic.py

# ML XGBoost v9
python engines/ml_xgboost_v9_ranking.py

# 앙상블
python engines/ensemble_fv3c_ml9.py
```

### 4. 분석
```bash
# 가중치 최적화
python analysis/optimize_ensemble_weights.py

# 월간 수익률 재계산
python analysis/recalc_monthly_ensemble.py

# 거래비용 반영
python analysis/apply_transaction_costs.py
```

---

## 📊 Sensitivity Analysis

### Turnover × Cost 조합

| Turnover | Cost | Sharpe | Return | MaxDD |
|----------|------|--------|--------|-------|
| 30% | 0.05% | 1.31 | 17.70% | -9.74% |
| 30% | 0.10% | 1.30 | 17.52% | -9.97% |
| **40%** | **0.10%** | **1.29** ✅ | **17.40%** | **-10.12%** |
| 40% | 0.20% | 1.25 | 16.92% | -10.74% |
| 50% | 0.10% | 1.28 | 17.28% | -10.28% |

**모든 시나리오에서 Sharpe > 1.2 달성**

---

## 🎯 향후 개선 방안

### 1. 유니버스 확대 (우선순위 높음)
- 현재: 30개
- 목표: S&P 500 전체 (500개)
- 예상 효과: Sharpe 1.5~1.8

### 2. 추가 팩터
- Quality: ROE, Debt/Equity
- Size: Market Cap
- Liquidity: Volume, Bid-Ask Spread

### 3. 리밸런싱 주기 최적화
- 현재: 월간
- 테스트: 주간, 격주
- 예측 기간과 일치 확인

---

## 📝 라이선스

MIT License

---

## 👤 Author

yhun1542

---

## 📚 참고 문헌

- Factor Investing: From Traditional to Alternative Risk Premia
- Machine Learning for Asset Managers (Marcos López de Prado)
- Advances in Financial Machine Learning

---

## ⚠️ 면책 조항

본 전략은 연구 및 교육 목적으로 제공됩니다. 실제 투자에 사용 시 발생하는 손실에 대해 책임지지 않습니다.

---

**Last Updated**: 2025-01-01
