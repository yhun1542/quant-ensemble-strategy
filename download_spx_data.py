#!/usr/bin/env python3
"""
S&P 500 종가 데이터 다운로드 (Polygon API)
"""
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta


def download_spx_close(
    api_key: str,
    start_date: str = "2021-01-01",
    end_date: str = None,
    output_path: str = "data/spx_close.csv"
) -> pd.DataFrame:
    """
    Polygon API로 S&P 500 종가 데이터 다운로드
    
    Parameters
    ----------
    api_key : str
        Polygon API 키
    start_date : str
        시작 날짜 (YYYY-MM-DD)
    end_date : str
        종료 날짜 (YYYY-MM-DD), None이면 오늘
    output_path : str
        저장 경로
    
    Returns
    -------
    pd.DataFrame
        S&P 500 종가 데이터
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Polygon API 엔드포인트
    # SPX는 Polygon에서 I:SPX 또는 SPY ETF를 사용
    # SPY를 사용하는 것이 더 안정적
    ticker = "SPY"
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
    
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key
    }
    
    print(f"S&P 500 데이터 다운로드 중... ({start_date} ~ {end_date})")
    print(f"Ticker: {ticker} (S&P 500 ETF)")
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"API 요청 실패: {response.status_code} - {response.text}")
    
    data = response.json()
    
    if "results" not in data:
        raise Exception(f"데이터 없음: {data}")
    
    # DataFrame 변환
    results = data["results"]
    df = pd.DataFrame(results)
    
    # 날짜 변환 (밀리초 timestamp)
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    
    # 종가만 추출
    spx_close = df.set_index("date")["c"].rename("SPX")
    spx_close.index.name = "date"
    
    # 저장
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    spx_close.to_csv(output_path)
    
    print(f"\n✅ S&P 500 데이터 저장 완료: {output_path}")
    print(f"   기간: {spx_close.index[0].date()} ~ {spx_close.index[-1].date()}")
    print(f"   거래일 수: {len(spx_close)}")
    print(f"   종가 범위: ${spx_close.min():.2f} ~ ${spx_close.max():.2f}")
    
    return spx_close


def main():
    """메인 실행 함수"""
    # Polygon API 키 (환경변수에서 읽거나 직접 입력)
    import os
    api_key = os.environ.get("POLYGON_API_KEY", "w7KprL4_lK7uutSH0dYGARkucXHOFXCN")
    
    # 다운로드
    spx_close = download_spx_close(
        api_key=api_key,
        start_date="2021-01-01",
        end_date=None,
        output_path="data/spx_close.csv"
    )
    
    # 간단한 통계
    print(f"\n통계:")
    print(spx_close.describe())


if __name__ == "__main__":
    main()
