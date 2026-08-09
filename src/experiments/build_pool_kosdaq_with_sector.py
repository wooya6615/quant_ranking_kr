"""
pooled_features_kosdaq.csv(BASE)에 섹터 강도 피처를 붙여서 저장.
(src/experiments/build_pool_with_sector.py와 동일한 구조, KOSDAQ 버전 -- market="KOSDAQ")

사용법 (레포 루트에서):
    python -m src.experiments.build_pool_kosdaq_with_sector
"""

from pathlib import Path

import numpy as np

from src.data.pooled_dataset import load_pooled_dataset
from src.data.sector_map import build_sector_map
from src.features.sector_features import load_sector_price_history, add_sector_features, FEATURE_COLS_SECTOR

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


if __name__ == "__main__":
    df = load_pooled_dataset(filename="pooled_features_kosdaq.csv")
    df["ticker"] = df["ticker"].astype(str)
    tickers = df["ticker"].unique().tolist()
    print(f"pooled_features_kosdaq.csv 로드 완료: {df.shape[0]}행, 종목 {len(tickers)}개\n")

    print("섹터 매핑 조회 중 (KOSDAQ)...")
    ticker_to_sector = build_sector_map(tickers, market="KOSDAQ")
    print(f"\n매핑 성공: {len(ticker_to_sector)}/{len(tickers)}종목\n")

    sector_codes = sorted(set(v["sector_code"] for v in ticker_to_sector.values()))
    print(f"필요한 섹터지수 {len(sector_codes)}개: {sector_codes}\n")

    start, end = str(df.index.min().date()), str(df.index.max().date())
    print("섹터지수 가격 히스토리 조회 중...")
    sector_close = load_sector_price_history(sector_codes, start, end)

    print("\n섹터 피처 계산 및 병합 중...")
    result = add_sector_features(df, ticker_to_sector, sector_close)

    before_n = len(result)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_SECTOR)
    print(f"섹터 피처 NaN 제거: {before_n}행 -> {len(result)}행")

    out_path = DATA_DIR / "pooled_features_kosdaq_with_sector.csv"
    result.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")
    print(f"최종 종목 수: {result['ticker'].nunique()}개")