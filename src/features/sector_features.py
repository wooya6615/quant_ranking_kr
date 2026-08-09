"""
섹터 지수(KRX 업종분류) 가격 히스토리로 섹터 수익률과 종목별 상대강도 feature 계산.

핵심 설계:
    - 섹터 지수 가격은 우리 풀 50종목의 평균이 아니라, pykrx가 제공하는 실제
      업종분류 지수(그 업종 전체 시가총액 가중)를 사용 -- 50종목 표본만으로 섹터
      평균을 내면 종목 수가 적은 섹터(예: 2~3종목)에서 노이즈가 커짐.
    - relative_strength = 종목 자체의 수익률 - 그 종목이 속한 섹터 지수의 수익률.
      이게 핵심 피처: "섹터 전체가 올라서 오른 것"과 "섹터 안에서 상대적으로
      더 잘한 것"을 분리해줌.

설치:
    pip install pykrx python-dotenv
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock

import pandas as pd
import numpy as np


def load_sector_price_history(sector_codes: list[str], start: str, end: str) -> dict[str, pd.Series]:
    """섹터 지수별 종가 시계열을 한 번씩만 조회해서 캐싱 (여러 종목이 같은 섹터를 공유하므로)."""
    sector_close = {}
    for code in sector_codes:
        try:
            df = stock.get_index_ohlcv_by_date(
                start.replace("-", ""), end.replace("-", ""), code
            )
            sector_close[code] = df["종가"]
            print(f"  섹터지수 {code}: {len(df)}행 조회 완료")
        except Exception as e:
            print(f"  섹터지수 {code} 조회 실패: {e}")
    return sector_close


def add_sector_features(
    df: pd.DataFrame,
    ticker_to_sector: dict[str, dict],
    sector_close: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    df: pooled BASE feature 데이터셋 (DatetimeIndex, 'ticker', 'return_5d'/'return_10d'/'return_20d' 포함)
    ticker_to_sector: build_sector_map()의 반환값
    sector_close: load_sector_price_history()의 반환값

    각 행(날짜, 종목)에 대해 그 종목이 속한 섹터의 5/10/20일 수익률과,
    종목 자체 수익률 대비 상대강도를 붙여서 반환.
    """
    df = df.copy()
    df["sector_code"] = df["ticker"].map(lambda t: ticker_to_sector.get(t, {}).get("sector_code"))

    # 섹터별 수익률 시계열 미리 계산 (섹터 개수만큼만, 종목 수만큼 반복 계산 안 함)
    sector_return_frames = {}
    for code, close in sector_close.items():
        r5 = close.pct_change(5)
        r10 = close.pct_change(10)
        r20 = close.pct_change(20)
        sector_return_frames[code] = pd.DataFrame({
            "sector_return_5d": r5, "sector_return_10d": r10, "sector_return_20d": r20,
        })

    # 날짜 인덱스 기준으로 섹터별 수익률을 join
    parts = []
    for code, ret_df in sector_return_frames.items():
        sub = df[df["sector_code"] == code].copy()
        if sub.empty:
            continue
        sub = sub.join(ret_df, how="left")
        parts.append(sub)

    result = pd.concat(parts).sort_index() if parts else df.copy()

    for cols_pair in [("return_5d", "sector_return_5d"), ("return_10d", "sector_return_10d"),
                       ("return_20d", "sector_return_20d")]:
        stock_col, sector_col = cols_pair
        rel_col = f"relative_strength_{stock_col.split('_')[1]}"
        result[rel_col] = result[stock_col] - result[sector_col]

    return result


FEATURE_COLS_SECTOR = [
    "sector_return_5d", "sector_return_10d", "sector_return_20d",
    "relative_strength_5d", "relative_strength_10d", "relative_strength_20d",
]