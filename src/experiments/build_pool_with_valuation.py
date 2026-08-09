"""
pooled_features_kospi200.csv(BASE)에 종목별 밸류에이션(PER/PBR/DIV) 피처를 붙여서 저장.

[주의] per_zscore_252d/pbr_zscore_252d는 "그 종목 자체의" 최근 252거래일 분포 대비
rolling z-score라서, 여러 종목이 섞인 풀링 데이터에 그대로 rolling을 적용하면 안 됨
(다른 종목의 값이 window에 섞여 들어감). 그래서 종목별로 부분집합을 뽑아서 개별
계산 후 다시 합침.

사용법 (레포 루트에서):
    python -m src.experiments.build_pool_with_valuation
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np

from src.data.pooled_dataset import load_pooled_dataset
from src.features.valuation_features import load_valuation, add_valuation_features, FEATURE_COLS_VALUATION

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


if __name__ == "__main__":
    df = load_pooled_dataset()
    df["ticker"] = df["ticker"].astype(str)
    tickers = sorted(df["ticker"].unique().tolist())
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, 종목 {len(tickers)}개\n")

    start, end = str(df.index.min().date()), str(df.index.max().date())

    frames = []
    failed = []
    for i, ticker in enumerate(tickers):
        try:
            sub = df[df["ticker"] == ticker].sort_index()
            ticker_padded = ticker.zfill(6)  # pykrx 호출은 반드시 6자리 zero-padded 코드로
            valuation_df = load_valuation(ticker_padded, start, end)
            merged = add_valuation_features(sub, valuation_df)
            frames.append(merged)
            print(f"  [{i+1}/{len(tickers)}] {ticker}: {len(merged)}행")
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: 실패, 건너뜀 ({e})")
            failed.append(ticker)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("모든 종목의 밸류에이션 데이터 조회에 실패했습니다.")

    result = pd.concat(frames).sort_index()
    before_n = len(result)
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_VALUATION)
    print(f"\n밸류에이션 피처 NaN 제거: {before_n}행 -> {len(result)}행 "
          f"(zscore 계산용 rolling window 252일 부족 구간 등 제외)")

    if failed:
        print(f"실패 종목: {failed}")

    out_path = DATA_DIR / "pooled_features_kospi200_with_valuation.csv"
    result.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")
    print(f"최종 종목 수: {result['ticker'].nunique()}개")