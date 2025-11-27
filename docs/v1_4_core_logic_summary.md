# v1.4 핵심 로직 요약

## 전체 파이프라인

```
가격 데이터
    ↓
Signal Prices 생성 (월초 3일 평균)
    ↓
Factors 계산 (signal prices 기준)
    ↓
FV4 + ML10 엔진 실행
    ↓
Execution Smoothing (2-step 포트 전환)
    ↓
앙상블 (60:40)
    ↓
리스크 레이어
    ↓
최종 수익률
```

## 1. Signal Smoothing (`utils/signal_prices.py`)

### 핵심 로직
```python
def build_signal_price_df(
    prices: pd.DataFrame,
    cfg: SignalSmoothingConfig,
) -> pd.DataFrame:
    # 월초 3일 윈도우 계산
    for month_start in month_starts:
        window_dates = [month_start + pd.Timedelta(days=i) for i in range(cfg.window)]
        window_dates = [d for d in window_dates if d in prices.index]
        
        if len(window_dates) < cfg.window:
            continue
        
        # 3일 평균 가격
        window_prices = prices.loc[window_dates]
        avg_price = window_prices.mean(axis=0)
        
        # 윈도우 마지막 날을 리밸 기준일로 설정
        signal_date = window_dates[-1]
        signal_df.loc[signal_date] = avg_price
```

**룩어헤드 체크**:
- ✅ 윈도우 마지막 날 (Day 3)을 리밸 기준일로 설정
- ✅ Day 3 시점에서 Day 1~3 가격만 사용
- ✅ 미래 데이터 사용 없음

## 2. Factors 계산 (`utils/factors.py`)

### 핵심 로직
```python
def compute_all_factors(
    prices: pd.DataFrame,
    signal_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    px = signal_prices if signal_prices is not None else prices
    
    # 모멘텀 (signal prices 기준)
    mom_60 = px / px.shift(60) - 1.0
    
    # 변동성 (signal prices 기준)
    vol_30 = px.pct_change().rolling(30).std()
    
    # Value proxy (signal prices 기준)
    value = 1.0 / px
```

**룩어헤드 체크**:
- ✅ signal_prices 사용 시 스무딩된 가격으로 팩터 계산
- ✅ shift(), rolling() 등 과거 데이터만 사용
- ✅ 미래 데이터 사용 없음

## 3. FV4 엔진 (`engines/factor_value_v4_signal_smoothing.py`)

### 핵심 로직
```python
def build_portfolio(
    prices: pd.DataFrame,
    factors: pd.DataFrame,  # signal prices 기반
    rebalance_dates: list,
) -> Dict:
    for d in rebalance_dates:
        factors_at_date = factors.loc[d]
        
        # Value proxy 기준 정렬
        factors_sorted = factors_at_date.sort_values("value_proxy")
        
        # Top 20% Long, Bottom 20% Short
        long_tickers = factors_sorted.head(n_long).index
        short_tickers = factors_sorted.tail(n_short).index
        
        # 변동성 역가중
        for ticker in long_tickers:
            vol = factors_at_date.loc[ticker, "volatility_30d"]
            weight = (1.0 / vol) / total_inv_vol
```

**룩어헤드 체크**:
- ✅ factors는 signal prices 기반 (이미 스무딩됨)
- ✅ 리밸 날짜 d 시점의 factors만 사용
- ✅ 미래 데이터 사용 없음

## 4. ML10 엔진 (`engines/ml_xgboost_v10_signal_smoothing.py`)

### 핵심 로직
```python
def _prepare_ml_dataset(
    prices: pd.DataFrame,  # 실제 가격 (labels용)
    factors: pd.DataFrame,  # signal prices 기반 (features용)
    start_date, end_date,
):
    for date in dates_in_range:
        # Features: signal prices 기반 factors
        features = factors_at_date.loc[ticker, feature_cols]
        
        # Labels: 실제 prices 기반 forward return
        fwd_ret = prices.loc[future_date, ticker] / prices.loc[date, ticker] - 1.0
        
        X_list.append(features)
        y_list.append(target)
```

**룩어헤드 체크**:
- ✅ Features는 signal prices 기반 (스무딩됨)
- ✅ Labels는 실제 prices 기반 (정확한 수익률)
- ✅ future_date는 prediction_horizon만큼 미래
- ✅ Train/Test 분리 정확

## 5. Execution Smoothing (`utils/execution_smoothing.py`)

### 핵심 로직
```python
def portfolio_returns_with_execution_smoothing(
    prices, weights_by_date, rebalance_dates, cfg
):
    # 리밸 시 타겟 포트 결정
    w_target = weights_by_date[date]
    
    # 2-step 전환
    for step in range(1, cfg.n_steps + 1):
        pct = step / cfg.n_steps
        w_step = w_prev + pct * (w_target - w_prev)
        execution_schedule.append(w_step)
    
    # 첫 번째 step 적용 (50%)
    current_weights = execution_schedule[0]
```

**룩어헤드 체크**:
- ✅ 리밸 날짜 d에 타겟 포트 결정
- ✅ d+1부터 점진적 전환
- ✅ 미래 가격 사용 없음

## 6. 백테스트 파이프라인 (`backtest_v1_4.py`)

### 데이터 흐름
```python
# 1) 가격 데이터 로드
prices = load_price_data(data_dir)

# 2) Signal prices 생성
signal_df = build_signal_price_df(prices, signal_cfg)
signal_df = expand_signal_prices(signal_df, prices.index)

# 3) Factors 계산 (signal prices 기준)
factors = compute_all_factors(prices, signal_df)

# 4) FV4 엔진
fv4_engine = FactorValueV4(fv4_cfg)
weights_fv4 = fv4_engine.build_portfolio(prices, factors, rebalance_dates)

# 5) ML10 엔진
ml10_engine = MLXGBoostV10(ml10_cfg)
ml10_engine.train(prices, factors, train_start, train_end)
weights_ml10 = ml10_engine.build_portfolio(prices, factors, rebalance_dates)

# 6) Execution Smoothing
ret_fv4 = portfolio_returns_with_execution_smoothing(
    prices, weights_fv4, rebalance_dates, exec_cfg
)
ret_ml10 = portfolio_returns_with_execution_smoothing(
    prices, weights_ml10, rebalance_dates, exec_cfg
)

# 7) 앙상블
ret_ensemble = 0.6 * ret_fv4 + 0.4 * ret_ml10

# 8) 리스크 레이어
ret_final = apply_risk_overlays(ret_ensemble, spx_close, ...)
```

**룩어헤드 체크**:
- ✅ 데이터 흐름이 일방향 (과거 → 현재)
- ✅ 각 단계의 출력이 다음 단계의 입력으로 정확히 전달
- ✅ Train/Test 분리 명확 (2023-06-01 기준)

## 주요 설계 의도

1. **Signal Smoothing**: 리밸 날짜 민감도 감소
2. **Execution Smoothing**: 거래 비용 및 슬리피지 현실화
3. **엔진 레벨 통합**: 후처리가 아닌 팩터 계산부터 적용
4. **Features vs Labels 분리**: ML 학습 시 signal prices(features) vs 실제 prices(labels)

## 잠재적 문제점

1. **Execution Smoothing 구현**: 날짜 매핑이 단순함 (다음 날부터 순차 적용)
2. **Value Proxy**: 실제 fundamentals 없이 가격 역수 사용
3. **에러 처리**: 일부 함수에서 에러 처리 부족
4. **테스트**: 단위 테스트 없음

## 평가 요청

위 로직을 검토하고 다음을 평가해주세요:
1. 룩어헤드 바이어스가 있는가?
2. Signal/Execution Smoothing 구현이 올바른가?
3. 개선이 필요한 부분은?
