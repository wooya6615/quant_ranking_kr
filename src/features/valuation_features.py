"""
밸류에이션(PER/PBR/배당수익률) 피처 계산.
(quant_xgboost의 feature_engineering_valuation.py 로직을 다종목 풀링 프레임에 맞게 이식)

핵심 아이디어:
    PER/PBR은 그날 종가 기준으로 계산되는 값(EPS/BPS는 이미 공시된 분기 실적)이라
    수급/공매도와 달리 1일 shift가 필요 없음 -- 가격 feature와 동일하게 "당일 종가
    시점에 이미 확정된 정보"로 취급.
    절대 PER/PBR은 종목/업종마다 기준이 달라서, 그 종목 자체의 최근 1년(252거래일)
    분포 대비 z-score로 정규화한 값도 같이 사용.

설치:
    pip install pykrx python-dotenv
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock

import pandas as pd
import numpy as np

FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d"]


def load_valuation(ticker_krx: str, start: str, end: str) -> pd.DataFrame:
    """종목 하나의 일별 PER/PBR/배당수익률(DIV) 히스토리."""
    df = stock.get_market_fundamental_by_date(
        start.replace("-", ""), end.replace("-", ""), ticker_krx
    )
    df = df.rename(columns={"PER": "per", "PBR": "pbr", "DIV": "div"})
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df[["per", "pbr", "div"]]


def add_valuation_features(df: pd.DataFrame, valuation_df: pd.DataFrame) -> pd.DataFrame:
    """
    df: 종목 하나의 BASE feature 데이터셋 (DatetimeIndex)
    valuation_df: load_valuation()의 반환값
    """
    merged = df.join(valuation_df, how="left")

    merged["per_zscore_252d"] = (
        (merged["per"] - merged["per"].rolling(252).mean()) / merged["per"].rolling(252).std()
    )
    merged["pbr_zscore_252d"] = (
        (merged["pbr"] - merged["pbr"].rolling(252).mean()) / merged["pbr"].rolling(252).std()
    )

    return merged