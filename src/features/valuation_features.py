"""
밸류에이션(PER/PBR/배당수익률) 피처 계산.
(quant_xgboost의 feature_engineering_valuation.py 로직을 다종목 풀링 프레임에 맞게 이식)

핵심 아이디어:
    PER/PBR은 그날 종가 기준으로 계산되는 값(EPS/BPS는 이미 공시된 분기 실적)이라
    수급/공매도와 달리 1일 shift가 필요 없음 -- 가격 feature와 동일하게 "당일 종가
    시점에 이미 확정된 정보"로 취급.
    절대 PER/PBR은 종목/업종마다 기준이 달라서, 그 종목 자체의 최근 1년(252거래일)
    분포 대비 z-score로 정규화한 값도 같이 사용.

[수정 2026-09] PER<=0(적자연도)을 NaN으로 명시 처리.
배경: PER은 연 1회(매년 5월, 사업보고서 검토 후) 갱신되고 다음 해 4월까지 고정값으로
유지됨. 적자연도(EPS<=0)면 PER이 0으로 기록되고 이 0이 1년 내내 유지됨. 이 파일은
quant_xgboost/src/feature_engineering_valuation.py에서 이미 발견/수정된 것과 동일한
버그를 그대로 갖고 있었음(PER=0을 정제 없이 rolling z-score에 그대로 흘려보냄) --
적자연도로 가득 찬 rolling(252) 창에서 std가 0에 가까워지면 z-score가 NaN/inf로
튀거나, 0을 "극단적 저평가"로 오독할 위험이 있었음. 게다가 이 레포는 풀링 구조라
(build_pool_with_valuation.py) dropna 시 적자 비중이 큰 종목일수록 행이 더 많이
빠지는 식으로 풀링 구성 자체가 왜곡됐을 가능성이 있음.

[해결 방식] quant_xgboost와 동일: 0으로 채우거나 dropna로 통째로 버리는 대신, NaN을
그대로 두고 XGBoost의 네이티브 결측치 처리에 맡김. 이 함수 안에서는 PER 관련
컬럼을 dropna 대상으로 삼지 않음 -- 호출부(build_pool_with_valuation.py 등)에서
밸류에이션 컬럼을 필수 dropna 대상에서 제외해야 함 (별도 패치 참고).

설치:
    pip install pykrx python-dotenv
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock

import pandas as pd
import numpy as np

FEATURE_COLS_VALUATION = ["per", "pbr", "div", "per_zscore_252d", "pbr_zscore_252d", "is_loss"]


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

    # [수정] PER<=0(적자/계산불가)을 명시적으로 NaN 처리 + is_loss 플래그로 정보 보존
    merged["is_loss"] = (merged["per"] <= 0).astype(int)
    merged["per"] = merged["per"].where(merged["per"] > 0, np.nan)

    merged["per_zscore_252d"] = (
        (merged["per"] - merged["per"].rolling(252, min_periods=60).mean())
        / merged["per"].rolling(252, min_periods=60).std()
    )
    merged["pbr_zscore_252d"] = (
        (merged["pbr"] - merged["pbr"].rolling(252, min_periods=60).mean())
        / merged["pbr"].rolling(252, min_periods=60).std()
    )

    return merged