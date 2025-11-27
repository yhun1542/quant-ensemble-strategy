#!/usr/bin/env python3
"""
3엔진 앙상블 가중치 최적화
- FV3c + ML9 + Momentum CS v1
- 월간 수익률 기준 Sharpe 최대화
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_engine_returns(path: str, key: str = "daily_returns") -> pd.Series:
    """
    JSON 파일에서 일간 수익률 시리즈로 변환
    """
    with open(path, "r") as f:
        data = json.load(f)

    dr = data[key]
    
    # 구조 확인 (두 가지 형식 지원)
    if isinstance(dr, dict) and "index" in dr and "values" in dr:
        # momentum_cs_v1 형식
        idx = pd.to_datetime(dr["index"])
        values = dr["values"]
    elif isinstance(dr, list):
        # factor_value_v3c, ml_xgboost_v9 형식
        idx = pd.to_datetime([item["date"] for item in dr])
        values = [item["ret"] for item in dr]
    else:
        raise ValueError(f"Unknown format in {path}")
    
    s = pd.Series(values, index=idx)
    s = s.sort_index()
    return s


def to_monthly(ret_daily: pd.Series) -> pd.Series:
    """일간 수익률을 월간 수익률로 변환"""
    return ret_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)


def calc_metrics_monthly(ret_m: pd.Series) -> dict:
    """월간 수익률 기준 성과 지표 계산"""
    ret_m = ret_m.dropna()
    
    if len(ret_m) == 0:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
        }
    
    years = (ret_m.index[-1] - ret_m.index[0]).days / 365.25
    if years <= 0:
        years = len(ret_m) / 12.0  # 월 수 기준
    
    cum_ret = (1 + ret_m).prod()
    ann_ret = cum_ret ** (1 / years) - 1
    vol_m = ret_m.std(ddof=0)
    ann_vol = vol_m * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + ret_m).cumprod()
    running_max = wealth.cummax()
    dd = wealth / running_max - 1
    max_dd = dd.min()

    win_rate = (ret_m > 0).mean()

    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
    }


def main():
    print("=" * 100)
    print("3엔진 앙상블 가중치 최적화")
    print("=" * 100)
    
    base = Path("results")

    # 세 엔진 결과 로딩
    fv_path = base / "factor_value_v3c_dynamic_oos.json"
    ml_path = base / "ml_xgboost_v9_ranking_oos.json"
    mom_path = base / "momentum_cs_v1_oos.json"

    print("\n1. 데이터 로딩...")
    ret_fv = load_engine_returns(str(fv_path))
    ret_ml = load_engine_returns(str(ml_path))
    ret_mom = load_engine_returns(str(mom_path))
    
    print(f"   - FV3c:      {len(ret_fv)} days ({ret_fv.index[0].date()} ~ {ret_fv.index[-1].date()})")
    print(f"   - ML9:       {len(ret_ml)} days ({ret_ml.index[0].date()} ~ {ret_ml.index[-1].date()})")
    print(f"   - Momentum:  {len(ret_mom)} days ({ret_mom.index[0].date()} ~ {ret_mom.index[-1].date()})")

    # 날짜 align
    df_daily = pd.concat(
        [
            ret_fv.rename("fv3c"),
            ret_ml.rename("ml9"),
            ret_mom.rename("mom_cs"),
        ],
        axis=1,
    ).dropna()
    
    print(f"\n2. 공통 기간 정렬 완료:")
    print(f"   - 공통 거래일 수: {len(df_daily)}")
    print(f"   - 기간: {df_daily.index[0].date()} ~ {df_daily.index[-1].date()}")

    # 월간 수익률로 변환
    df_m = df_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    print(f"   - 월간 데이터 포인트: {len(df_m)}")

    # 개별 엔진 성과
    print("\n3. 개별 엔진 성과 (월간 기준):")
    print("-" * 100)
    for col in df_m.columns:
        metrics = calc_metrics_monthly(df_m[col])
        print(f"   {col.upper():10s} | Sharpe: {metrics['sharpe']:6.3f} | "
              f"Return: {metrics['ann_return']*100:6.2f}% | "
              f"Vol: {metrics['ann_vol']*100:6.2f}% | "
              f"MaxDD: {metrics['max_drawdown']*100:6.2f}% | "
              f"WinRate: {metrics['win_rate']*100:5.1f}%")

    # 엔진 간 상관계수
    corr = df_m.corr()
    print("\n4. 엔진 간 상관 행렬 (월간 수익률):")
    print("-" * 100)
    print(corr.round(3))

    # 3엔진 가중치 그리드 서치
    print("\n5. 3엔진 가중치 그리드 서치 (0.0 ~ 1.0, step 0.1)...")
    rows = []
    ws = np.linspace(0.0, 1.0, 11)  # 0.0 ~ 1.0, step 0.1

    total_combinations = 0
    for w_fv in ws:
        for w_ml in ws:
            for w_mom in ws:
                if abs(w_fv + w_ml + w_mom - 1.0) > 1e-9:
                    continue
                
                total_combinations += 1

                combo = (
                    w_fv * df_m["fv3c"]
                    + w_ml * df_m["ml9"]
                    + w_mom * df_m["mom_cs"]
                )

                metrics = calc_metrics_monthly(combo)
                rows.append(
                    {
                        "w_fv3c": w_fv,
                        "w_ml9": w_ml,
                        "w_mom": w_mom,
                        **metrics,
                    }
                )
    
    print(f"   - 총 조합 수: {total_combinations}")

    result = pd.DataFrame(rows)
    result = result.sort_values("sharpe", ascending=False)

    # 상위 몇 개 출력
    print("\n6. Sharpe 기준 상위 20개 조합:")
    print("=" * 100)
    top_20 = result[["w_fv3c", "w_ml9", "w_mom", "sharpe", "ann_return", "ann_vol", "max_drawdown", "win_rate"]].head(20)
    
    # 포맷팅
    pd.options.display.float_format = '{:.4f}'.format
    print(top_20.to_string(index=False))
    
    # 최적 조합 상세 출력
    best = result.iloc[0]
    print("\n7. 최적 조합 (Sharpe 최대):")
    print("=" * 100)
    print(f"   가중치:")
    print(f"     - FV3c:     {best['w_fv3c']*100:5.1f}%")
    print(f"     - ML9:      {best['w_ml9']*100:5.1f}%")
    print(f"     - Momentum: {best['w_mom']*100:5.1f}%")
    print(f"\n   성과:")
    print(f"     - Sharpe Ratio:      {best['sharpe']:.4f}")
    print(f"     - Annual Return:     {best['ann_return']*100:.2f}%")
    print(f"     - Annual Volatility: {best['ann_vol']*100:.2f}%")
    print(f"     - Max Drawdown:      {best['max_drawdown']*100:.2f}%")
    print(f"     - Win Rate:          {best['win_rate']*100:.2f}%")
    
    # 기존 2엔진 앙상블과 비교
    print("\n8. 기존 2엔진 앙상블(FV3c 60% + ML9 40%)과 비교:")
    print("=" * 100)
    
    combo_2engine = 0.6 * df_m["fv3c"] + 0.4 * df_m["ml9"]
    metrics_2engine = calc_metrics_monthly(combo_2engine)
    
    print(f"   2엔진 앙상블:")
    print(f"     - Sharpe Ratio:      {metrics_2engine['sharpe']:.4f}")
    print(f"     - Annual Return:     {metrics_2engine['ann_return']*100:.2f}%")
    print(f"     - Max Drawdown:      {metrics_2engine['max_drawdown']*100:.2f}%")
    
    print(f"\n   3엔진 최적 조합:")
    print(f"     - Sharpe Ratio:      {best['sharpe']:.4f} (개선: {(best['sharpe']/metrics_2engine['sharpe']-1)*100:+.2f}%)")
    print(f"     - Annual Return:     {best['ann_return']*100:.2f}% (개선: {(best['ann_return']-metrics_2engine['ann_return'])*100:+.2f}%p)")
    print(f"     - Max Drawdown:      {best['max_drawdown']*100:.2f}% (개선: {(best['max_drawdown']-metrics_2engine['max_drawdown'])*100:+.2f}%p)")

    # 필요하면 파일로 저장
    out_path = base / "ensemble_3engines_optimization.json"
    result.to_json(out_path, orient="records", indent=2)
    print(f"\n✅ 3엔진 가중치 최적화 결과 저장: {out_path}")
    
    # 최적 조합 별도 저장
    best_config = {
        "weights": {
            "fv3c": float(best['w_fv3c']),
            "ml9": float(best['w_ml9']),
            "momentum": float(best['w_mom'])
        },
        "metrics": {
            "sharpe": float(best['sharpe']),
            "ann_return": float(best['ann_return']),
            "ann_vol": float(best['ann_vol']),
            "max_drawdown": float(best['max_drawdown']),
            "win_rate": float(best['win_rate'])
        },
        "comparison_vs_2engine": {
            "sharpe_improvement_pct": float((best['sharpe']/metrics_2engine['sharpe']-1)*100),
            "return_improvement_pp": float((best['ann_return']-metrics_2engine['ann_return'])*100),
            "maxdd_improvement_pp": float((best['max_drawdown']-metrics_2engine['max_drawdown'])*100)
        }
    }
    
    best_config_path = base / "ensemble_3engines_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    
    print(f"✅ 최적 조합 설정 저장: {best_config_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
