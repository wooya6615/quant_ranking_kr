"""
연도별 edge(랭킹픽 net - 동일가중 net)가 그 해 시장 국면(추세 강도/변동성)과
상관관계가 있는지 검증.

배경 가설:
    analyze_concentration.py에서 2020년(코로나 급반등)엔 edge가 크게 마이너스였고,
    2024~2025(강세장)엔 edge가 압도적으로 플러스였음. BASE 피처 대부분이 모멘텀
    계열이라 "추세가 강할 때(주로 상승 추세) 잘 먹히고, 급반등/국면전환기엔 실패하는"
    패턴일 수 있다는 가설.

방법:
    1) 연도별 edge (analyze_concentration.py의 analyze_by_year 재사용)
    2) 연도별 코스피(^KS11) 수익률, 변동성을 yfinance로 조회
    3) 둘 사이 상관계수 계산 -- 코스피 연간 수익률과 edge가 강한 양의 상관을 보이면
       "모멘텀이 강한 상승장에서만 이 랭킹 모델이 먹힌다"는 가설이 데이터로 뒷받침됨.
       이 경우 매크로 국면 피처(예: 20일 코스피 수익률, 시장 변동성)를 추가해서
       모델이 국면을 스스로 구분하게 만드는 다음 스텝이 근거를 갖게 됨.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.analyze_regime_correlation
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import run_walk_forward_single_seed, TOP_K
from src.experiments.analyze_concentration import analyze_by_year, SEED


def get_kospi_yearly_stats(start: str = "2015-01-01", end: str = "2026-07-18") -> pd.DataFrame:
    """연도별 코스피(^KS11) 연간 수익률과 일간 변동성(연율화)을 반환."""
    df = yf.download("^KS11", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    close = df["Close"]
    daily_ret = close.pct_change()

    rows = []
    for year, g in close.groupby(close.index.year):
        if len(g) < 2:
            continue
        year_return = g.iloc[-1] / g.iloc[0] - 1
        year_vol = daily_ret[daily_ret.index.year == year].std() * np.sqrt(252)
        rows.append({"year": year, "kospi_year_return": year_return, "kospi_year_vol": year_vol})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_pooled_dataset()
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    print(f"seed={SEED}, top_k={TOP_K}로 전체 walk-forward 실행 중...\n")
    daily_all, _ = run_walk_forward_single_seed(df, seed=SEED, top_k=TOP_K)
    year_edge_df = analyze_by_year(daily_all)[["year", "n_days", "avg_edge"]]

    print("코스피 연도별 수익률/변동성 조회 중...\n")
    kospi_df = get_kospi_yearly_stats()

    merged = year_edge_df.merge(kospi_df, on="year", how="inner")

    print("=" * 80)
    print("=== 연도별 edge vs 코스피 국면 ===")
    print("=" * 80)
    print(merged.round(5).to_string(index=False))

    corr_return = merged["avg_edge"].corr(merged["kospi_year_return"])
    corr_vol = merged["avg_edge"].corr(merged["kospi_year_vol"])

    print(f"\navg_edge vs 코스피 연간수익률 상관계수: {corr_return:.3f}")
    print(f"avg_edge vs 코스피 연간변동성 상관계수: {corr_vol:.3f}")

    print("\n[해석 가이드]")
    print("- 코스피 연간수익률과의 상관계수가 +0.5 이상이면:")
    print("  강세장(추세 상승)일수록 이 랭킹 모델이 잘 먹힌다는 뜻 -- 모멘텀 피처 의존 가설을 뒷받침.")
    print("  이 경우 시장 국면(추세/변동성) 피처를 추가해서 모델이 국면을 구분하게 하는 게 다음 스텝.")
    print("- 상관계수가 0에 가까우면: 시장 방향성과는 무관한 다른 이유(예: 특정 종목 쏠림)일 수 있음 --")
    print("  이 경우 국면 피처보다 종목별/섹터별 분해가 먼저 필요.")
    print("- 코스피 변동성과 강한 음의 상관(-0.5 이하)이면: 변동성 낮고 안정적인 국면에서만 먹힌다는 뜻.")