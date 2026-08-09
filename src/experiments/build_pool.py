"""
KOSPI200 구성종목을 풀링한 BASE feature 데이터셋 생성 실행 스크립트.

핵심 설계:
    - 종목 리스트 조회(kospi200_universe.get_kospi200_tickers)는 딱 1회 호출.
    - 가격 데이터는 yfinance만 사용 (BASE feature 13개는 Close/Volume 파생이라
      pykrx 로그인이 필요한 수급/밸류에이션 데이터는 이 단계에서 불필요).
    - n_tickers로 먼저 50개 정도만 빠르게 돌려보고, 파이프라인 검증되면
      None(전체 200종목)으로 확장 권장.
      (구성종목 리스트 순서가 시가총액순이 아닐 수 있어서, 서브셋은
       "무작위에 가까운 KOSPI200 부분표본" 정도로 취급할 것)

설치:
    pip install pykrx yfinance python-dotenv

사용법 (레포 루트에서):
    python -m src.experiments.build_pool
"""

import time
from pathlib import Path

import pandas as pd

from src.data.kospi200_universe import get_kospi200_tickers
from src.features.base_features import build_feature_dataset

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK = "^KS11"  # 기존 quant_xgboost 실험과의 비교 가능성을 위해 코스피 종합지수 유지


def build_pooled_dataset_kospi200(
    n_tickers: int = 50,   # None이면 전체 KOSPI200. 먼저 50개로 파이프라인 검증 권장.
    benchmark: str = BENCHMARK,
    start: str = "2015-01-01",
    end: str = "2026-07-18",
    horizon: int = 10,
    cost_threshold: float = 0.005,
    sleep_sec: float = 0.3,  # yfinance 연속 호출 사이 딜레이 -- 과도한 요청으로 인한 차단 방지
) -> pd.DataFrame:
    tickers = get_kospi200_tickers()
    if n_tickers is not None:
        tickers = tickers[:n_tickers]
        print(f"-> 서브셋 {len(tickers)}개로 진행 (n_tickers={n_tickers})")

    frames = []
    failed = []
    for i, code in enumerate(tickers):
        yf_ticker = f"{code}.KS"
        try:
            df = build_feature_dataset(
                ticker=yf_ticker, benchmark=benchmark, start=start, end=end,
                horizon=horizon, cost_threshold=cost_threshold,
            )
            if len(df) < 100:  # 데이터가 너무 적으면(신규상장 등) 랭킹 학습에 노이즈만 추가하므로 제외
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
        raise RuntimeError("모든 종목 다운로드에 실패했습니다. 네트워크/yfinance 상태를 확인하세요.")

    pooled = pd.concat(frames)
    pooled = pooled.sort_index()

    print(f"\n성공: {len(frames)}종목 / 실패·제외: {len(failed)}종목")
    if failed:
        print(f"제외된 종목: {failed}")

    return pooled


if __name__ == "__main__":
    # 50종목 파이프라인 검증 완료 -> 전체 KOSPI200으로 확장
    pooled = build_pooled_dataset_kospi200(n_tickers=None)

    print(f"\n전체 풀링 데이터: {pooled.shape[0]}행, {pooled['ticker'].nunique()}종목")
    print(f"날짜 범위: {pooled.index.min().date()} ~ {pooled.index.max().date()}")
    print(f"\n라벨 분포:\n{pooled['label'].value_counts(normalize=True)}")

    out_path = DATA_DIR / "pooled_features_kospi200.csv"
    pooled.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")
    print("\n[다음 단계] 이 파일이 잘 나왔으면:")
    print("  1) python -m src.experiments.run_ranking_experiment 실행")
    print("  2) 문제 없으면 build_pooled_dataset_kospi200(n_tickers=None)으로 전체 200종목 재실행")