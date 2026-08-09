"""
KOSDAQ 시가총액 상위 N종목을 풀링한 BASE feature 데이터셋 생성 실행 스크립트.
(src/experiments/build_pool.py와 동일한 구조, KOSDAQ 버전)

[주의] KOSDAQ 종목은 yfinance ticker suffix가 .KQ여야 함 -- .KS를 쓰면 에러 없이
거의 빈 데이터만 돌아오므로 반드시 .KQ로 맞출 것 (quant_xgboost에서 이미 겪었던 함정).
벤치마크도 코스피 종합(^KS11)이 아니라 코스닥 종합(^KQ11)을 사용.

사용법 (레포 루트에서):
    python -m src.experiments.build_pool_kosdaq
"""

import time
from pathlib import Path

import pandas as pd

from src.data.kosdaq_universe import get_top_kosdaq_tickers
from src.features.base_features import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "^KQ11"  # 코스닥 종합지수


def build_pooled_dataset_kosdaq(
    n_tickers: int = 50,
    benchmark: str = BENCHMARK,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 10,
    cost_threshold: float = 0.005,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    tickers = get_top_kosdaq_tickers(n=n_tickers)

    frames = []
    failed = []
    for i, code in enumerate(tickers):
        yf_ticker = f"{code}.KQ"  # .KS 아님 -- KOSDAQ은 반드시 .KQ
        try:
            df = build_feature_dataset(
                ticker=yf_ticker, benchmark=benchmark, start=start, end=end,
                horizon=horizon, cost_threshold=cost_threshold,
            )
            if len(df) < 100:
                print(f"  [{i+1}/{len(tickers)}] {code}: 행 수 부족({len(df)}행)으로 건너뜀")
                failed.append(code)
                continue
            df = df.copy()
            df["ticker"] = code
            frames.append(df)
            print(f"  [{i+1}/{len(tickers)}] {code}: {len(df)}행")
        except Exception as e:
            print(f"  [{i+1}/{len(tickers)}] {code}: 실패, 건너뜀 ({e})")
            failed.append(code)
        time.sleep(sleep_sec)

    if not frames:
        raise RuntimeError("모든 종목 다운로드에 실패했습니다.")

    pooled = pd.concat(frames)
    pooled = pooled.sort_index()

    print(f"\n성공: {len(frames)}종목 / 실패·제외: {len(failed)}종목")
    if failed:
        print(f"제외된 종목: {failed}")

    return pooled


if __name__ == "__main__":
    pooled = build_pooled_dataset_kosdaq(n_tickers=50)

    print(f"\n전체 풀링 데이터: {pooled.shape[0]}행, {pooled['ticker'].nunique()}종목")
    print(f"날짜 범위: {pooled.index.min().date()} ~ {pooled.index.max().date()}")
    print(f"\n라벨 분포:\n{pooled['label'].value_counts(normalize=True)}")

    out_path = DATA_DIR / "pooled_features_kosdaq.csv"
    pooled.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")