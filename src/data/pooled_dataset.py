"""
풀링된 다종목 데이터셋 로드 + 날짜 기준 Walk-Forward 분할.
quant_xgboost/src/train_xgboost_pooled.py의 walk_forward_splits_by_date()를 그대로 이식
(같은 날짜의 여러 종목이 train/test로 쪼개지면 날짜 기준 누수가 생기므로,
 행 개수가 아니라 '고유 날짜' 기준으로 슬라이딩해야 함).
"""

from pathlib import Path

import pandas as pd
import numpy as np

from src.features.base_features import FEATURE_COLS_BASE

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
def load_pooled_dataset(filename: str = "pooled_features_kospi200.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    required = FEATURE_COLS_BASE + ["ticker", "future_return", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}에 다음 컬럼이 없습니다: {missing}\n"
            "src/experiments/build_pool.py를 먼저 실행했는지 확인하세요."
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_BASE + ["future_return", "label"])
    df["ticker"] = df["ticker"].astype("category")
    df = df.sort_index()
    return df


# ------------------------------------------------------------------
# 2. 날짜 기준 Walk-Forward 분할
# ------------------------------------------------------------------
def walk_forward_splits_by_date(df: pd.DataFrame, train_days: int, test_days: int, step_days: int, embargo_days: int):
    unique_dates = df.index.unique().sort_values()
    n = len(unique_dates)

    splits = []
    start = 0
    while start + train_days + embargo_days + test_days <= n:
        train_dates = unique_dates[start: start + train_days]
        test_start = start + train_days + embargo_days
        test_dates = unique_dates[test_start: test_start + test_days]
        splits.append((train_dates, test_dates))
        start += step_days
    return splits