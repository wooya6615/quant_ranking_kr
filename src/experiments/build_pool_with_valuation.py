"""
pooled_features_kospi200.csv(BASE)에 종목별 밸류에이션(PER/PBR/DIV) 피처를 붙여서 저장.

[주의] per_zscore_252d/pbr_zscore_252d는 "그 종목 자체의" 최근 252거래일 분포 대비
rolling z-score라서, 여러 종목이 섞인 풀링 데이터에 그대로 rolling을 적용하면 안 됨
(다른 종목의 값이 window에 섞여 들어감). 그래서 종목별로 부분집합을 뽑아서 개별
계산 후 다시 합침.

[수정 2026-09] valuation_features.py의 PER<=0 NaN 처리 수정에 맞춰, per/is_loss/
per_zscore_252d는 필수 dropna 대상에서 제외함 -- 적자연도가 legitimate NaN이 됐는데
여기서 그대로 dropna하면 quant_xgboost 때와 똑같이 "행이 조용히 사라지는" 문제가
재발함(특히 적자 비중이 큰 종목일수록 풀링에서 과소표집됨). PBR/DIV는 이 문제가
없어서(장부가 기준이라 적자여도 정상 계산됨) 그대로 필수 dropna 유지.

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

# [수정] PER 계열(is_loss로 정보 보존되는 legitimate NaN)은 필수 dropna에서 제외
REQUIRED_VALUATION_COLS = ["pbr", "div", "pbr_zscore_252d"]


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
            print(f"  [{i + 1}/{len(tickers)}] {ticker}: {len(merged)}행")
        except Exception as e:
            print(f"  [{i + 1}/{len(tickers)}] {ticker}: 실패, 건너뜀 ({e})")
            failed.append(ticker)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("모든 종목의 밸류에이션 데이터 조회에 실패했습니다.")

    result = pd.concat(frames).sort_index()
    before_n = len(result)

    # inf만 NaN으로 치환 (per가 legitimate NaN인 행까지 여기서 날리지 않기 위해 subset 지정)
    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.dropna(subset=REQUIRED_VALUATION_COLS)

    print(f"\n밸류에이션 피처 NaN 제거: {before_n}행 -> {len(result)}행 "
          f"(pbr/div/pbr_zscore_252d 기준만; per 계열은 적자연도 legitimate NaN이라 "
          f"XGBoost 네이티브 결측치 처리에 맡기고 행을 보존함)")
    print(f"is_loss=1(적자연도) 비율: {result['is_loss'].mean():.1%}")

    if failed:
        print(f"실패 종목: {failed}")

    out_path = DATA_DIR / "pooled_features_kospi200_with_valuation.csv"
    result.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")
    print(f"최종 종목 수: {result['ticker'].nunique()}개")
    print("\n⚠️ 이 파일로 밸류에이션 ablation을 다시 돌려서, 기존 [폐기] 결론이")
    print("이 버그 때문이었는지 재확인할 것 (train_xgboost_ablation 계열 스크립트,")
    print("FEATURE_COLS_VALUATION_ONLY의 dropna도 REQUIRED_VALUATION_COLS 기준으로")
    print("맞춰서 XGBoost 학습 시 per NaN이 native missing-value handling으로 가게 할 것).")