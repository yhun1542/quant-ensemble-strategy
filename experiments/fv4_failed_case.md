# FV4 실패 사례 분석

**작성일**: 2025-01-27  
**상태**: 실전 부적합 (Deprecated)

---

## 요약

**Factor Value v4 (FV4)**는 Signal Smoothing을 지원하는 가치 투자 엔진으로 설계되었으나, 백테스트 결과 **실전 사용 부적합**으로 판정되었습니다.

### 성과 요약

| 모드 | Sharpe | 연수익률 | Max DD | 판정 |
|------|--------|----------|--------|------|
| **롱온리** | 0.48 | 8.13% | -24.48% | ⚠️ 약한 알파 |
| **롱숏 (gross=1.0)** | NaN | -233.72% | -271.33% | ❌ 완전 실패 |

---

## 실패 원인 분석

### 1. Value Proxy의 구조적 한계

**현재 구현**:
```python
value_proxy = 1.0 / price  # 가격 역수
```

**문제점**:
- 실제 "저평가" 종목이 아니라 **"가격 단위가 작은 주식"**을 선택
- 펀더멘털 데이터(PER, PBR, EV/EBITDA, ROE 등) 부재
- 경제적 의미 없는 팩터

### 2. 롱숏 전략의 높은 신호 품질 요구

롱숏 전략은 **IC (Information Coefficient)**가 매우 높아야 작동합니다:
- 필요 IC: 0.05~0.10 이상
- FV4 추정 IC: < 0.01 (노이즈 수준)

**결과**:
- 롱온리: 간신히 양의 수익률 (Sharpe 0.48)
- 롱숏: 노이즈 + 역시그널 → 파산 수준 손실 (-234%)

### 3. 포지션 사이징은 정상

**검증 결과**:
- gross: 1.0000 (목표: 1.0) ✅
- net: 0.0000 (목표: 0.0) ✅
- 변동성 역가중 정상 작동 ✅

→ **문제는 사이징이 아니라 신호 자체**

---

## 백테스트 상세 결과

### 롱온리 모드

**설정**:
- top_quantile: 0.2 (상위 20% 선택)
- long_gross: 1.0
- short_gross: 0.0

**성과**:
- Sharpe: 0.48
- Annual Return: 8.13%
- Annual Vol: 17.04%
- Max DD: -24.48%
- Win Rate: 52.89%
- Total Return: 26.00% (3.5년)

**평가**: 약한 알파, 높은 변동성, 큰 DD → 실전 부적합

### 롱숏 모드 (gross=1.0)

**설정**:
- top_quantile: 0.2
- long_gross: 0.5 (롱 50%)
- short_gross: 0.5 (숏 50%)

**성과**:
- Sharpe: NaN (음수 수익률)
- Total Return: -233.72%
- Max DD: -271.33%
- Annual Vol: 155.64%
- Win Rate: 50.07%

**평가**: 완전 실패, 파산 수준

---

## 교훈 및 향후 방향

### 교훈

1. **가격 역수는 Value 팩터가 아니다**
   - 진짜 Value는 펀더멘털 데이터 필요
   - 단순 가격 스케일링은 노이즈

2. **롱숏은 신호 품질이 매우 높아야 한다**
   - IC < 0.05 수준은 롱숏 부적합
   - 롱온리로도 Sharpe < 1.0이면 롱숏 시도 금지

3. **포지션 사이징만으로는 나쁜 신호를 구제할 수 없다**
   - gross/net이 완벽해도 신호가 나쁘면 실패

### 향후 방향

**FV5 설계 시 필수 요소**:
1. 펀더멘털 데이터 통합 (PER, PBR, ROE, CF 등)
2. 크로스섹션 IC 검증 (최소 0.05 이상)
3. 롱온리 먼저 검증 → Sharpe 1.0+ 달성 후 롱숏 시도
4. 섹터 중립화 (sector neutralization)

---

## 파일 위치

**엔진 코드**:
- `engines/factor_value_v4_signal_smoothing.py` (deprecated)

**백테스트 스크립트**:
- `backtest_fv4_long_only.py`
- `backtest_fv4_long_short.py`

**결과 파일**:
- `results/fv4_long_only_results.json`
- `results/fv4_long_short_results.json`

---

## 최종 판정

**❌ FV4는 v1.4 앙상블에 포함하지 않음**

**✅ 기존 FV3c (Factor Value v3c) 유지**
- FV3c는 롱온리 전략으로 검증됨
- v1.0~v1.3에서 안정적 성과 확인

**📝 FV4는 "실패 케이스 스터디"로 보관**
- 나중에 FV5 설계 시 참고 자료로 활용
- "이렇게 하면 안 된다"는 명확한 증거

---

**작성자**: Manus AI  
**검토자**: Gemini 2.5 Pro, GPT-4o, Claude Opus 4, Grok 4
