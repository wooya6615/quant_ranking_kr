"""
랭킹픽(top_k)으로 뽑힌 종목들의 edge 기여도가 소수 종목에 쏠려있는지 확인.
(analyze_concentration.py가 "연도" 단위로 했던 걸 "종목" 단위로 재구성)

배경 가설:
    analyze_regime_correlation.py에서 edge가 코스피 지수 방향성과 무관하다는 게 확인됨
    (상관계수 -0.06). 즉 2024~2025 성공이 "시장 전체가 좋아서"가 아니라 "그 시기에
    유독 잘 오른 특정 종목/섹터를 모델이 반복적으로 골라서"일 가능성이 큼.

방법:
    run_fold에서 이미 날짜별로 top_k를 뽑고 있는데, 지금까지는 "어떤 종목이었는지"는
    버리고 수익률 평균만 남겼음. 이 스크립트는 종목까지 기록해서, 종목별로:
      - 픽으로 뽑힌 횟수
      - 그 종목이 기여한 총 edge (pick 됐을 때의 future_return - 그날 동일가중 평균)
    을 집계함. 상위 몇 종목이 전체 edge의 몇 %를 차지하는지가 핵심 지표.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.analyze_ticker_concentration
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

from src.data.pooled_dataset import load_pooled_dataset, walk_forward_splits_by_date
from src.features.base_features import FEATURE_COLS_BASE
from src.models.lightgbm_ranker import add_relevance_label, HORIZON, TOP_K, ROUND_TRIP_COST

SEED = 42  # analyze_concentration.py / analyze_regime_correlation.py와 동일 대표 seed


def run_fold_with_tickers(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int,
                           top_k: int, feature_cols: list = None):
    """run_fold과 거의 동일하지만, 랭킹 모델만 학습하고 픽된 종목(ticker)까지 기록해서 반환."""
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS_BASE
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    X_train = train_df[feature_cols + ["ticker"]]
    X_test = test_df[feature_cols + ["ticker"]]
    train_group = train_df.groupby(train_df.index).size().values

    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg",
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=random_state, verbosity=-1,
    )
    ranker.fit(X_train, train_df["relevance"], group=train_group, categorical_feature=["ticker"])

    test_df = test_df.copy()
    test_df["rank_score"] = ranker.predict(X_test)

    pick_rows = []
    for date, g in test_df.groupby(test_df.index):
        if len(g) < 2:
            continue
        k = min(top_k, len(g))
        equal_weight_return = g["future_return"].mean()
        picks = g.nlargest(k, "rank_score")
        for ticker, row in picks[["ticker", "future_return"]].set_index("ticker").iterrows():
            pick_rows.append({
                "date": date,
                "ticker": ticker,
                "future_return": row["future_return"],
                "edge_contribution": row["future_return"] - ROUND_TRIP_COST - (equal_weight_return - ROUND_TRIP_COST),
            })
    return pd.DataFrame(pick_rows)


def run_walk_forward_ticker_level(df: pd.DataFrame, seed: int, top_k: int = TOP_K,
                                   train_days=300, test_days=60, step_days=60, embargo_days=HORIZON,
                                   feature_cols: list = None):
    df = add_relevance_label(df)
    splits = walk_forward_splits_by_date(df, train_days, test_days, step_days, embargo_days)

    all_picks = []
    for train_dates, test_dates in splits:
        train_df = df[df.index.isin(train_dates)]
        test_df = df[df.index.isin(test_dates)]
        picks_df = run_fold_with_tickers(train_df, test_df, random_state=seed, top_k=top_k, feature_cols=feature_cols)
        all_picks.append(picks_df)

    return pd.concat(all_picks, ignore_index=True)


if __name__ == "__main__":
    df = load_pooled_dataset()
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    print(f"seed={SEED}, top_k={TOP_K}로 전체 walk-forward 실행 중 (종목 단위 기록)...\n")
    picks_df = run_walk_forward_ticker_level(df, seed=SEED, top_k=TOP_K)

    total_edge = picks_df["edge_contribution"].sum()

    ticker_summary = (
        picks_df.groupby("ticker")
        .agg(n_picked=("edge_contribution", "size"), total_edge_contribution=("edge_contribution", "sum"))
        .sort_values("total_edge_contribution", ascending=False)
    )
    ticker_summary["share_of_total_edge"] = ticker_summary["total_edge_contribution"] / total_edge

    print("=" * 80)
    print("=== 종목별 edge 기여도 (상위 15개) ===")
    print("=" * 80)
    print(ticker_summary.head(15).round(5).to_string())

    print("\n" + "=" * 80)
    print("=== 종목별 edge 기여도 (하위 15개 -- 오히려 손해를 끼친 종목) ===")
    print("=" * 80)
    print(ticker_summary.tail(15).round(5).to_string())

    top5_share = ticker_summary.head(5)["share_of_total_edge"].sum()
    top10_share = ticker_summary.head(10)["share_of_total_edge"].sum()
    n_tickers_total = len(ticker_summary)

    print(f"\n전체 종목 수: {n_tickers_total}개")
    print(f"상위 5종목이 차지하는 전체 edge 비중: {top5_share:.1%}")
    print(f"상위 10종목이 차지하는 전체 edge 비중: {top10_share:.1%}")

    print("\n[해석 가이드]")
    print(f"- 상위 5종목 비중이 50%를 넘으면({top5_share:.1%}): 소수 종목 쏠림이 매우 강함 --")
    print("  '모델이 일반적으로 좋은 종목을 고르는 능력'이 아니라 '특정 몇 종목의 급등을 우연히 반복해서")
    print("  잡아낸 것'일 가능성이 높음. 그 종목들이 왜 픽됐는지(섹터/이벤트) 확인 필요.")
    print("- 넓게 분산되어 있으면(상위 10종목이 30% 이하 등): 특정 종목 의존이 아닌 일반화 가능한")
    print("  패턴일 가능성이 더 높음.")