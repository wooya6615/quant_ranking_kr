"""
LightGBM lambdarank 기반 "오늘 N종목 중 뭘 살지" 랭킹 모델.

배경 / 문제의식:
    quant_xgboost에서는 XGBoost 이진분류(오를지/내릴지)로 종목을 "독립적으로" 평가했음.
    실전 트레이딩은 "여러 종목 중 뭘 살지"를 매일 결정하는 문제에 가까워서,
    lambdarank로 같은 날짜(query group) 안에서 종목들을 상대 순위로 학습시킴.

핵심 설계:
    - relevance label: 같은 날짜에 존재하는 종목들의 future_return을 오름차순 순위로 매겨서
      0 ~ (그날 종목 수 - 1)의 graded relevance로 사용.
    - group: lambdarank는 같은 query group(날짜)의 행이 데이터 안에서 연속(contiguous)
      해야 함 -> 날짜로 sort 후 date별 row 개수를 group 배열로 넘김.
    - 평가: NDCG는 참고용이고, 실전 관점에서 의미 있는 지표는
      "매일 모델이 상위 TOP_K로 뽑은 종목들의 평균 수익률".
      풀이 50~200종목처럼 크면 top-1 하나만 보는 건 노이즈가 크므로 top-k 평균으로 봄.
      이걸 (a) 동일가중 baseline, (b) 기존 XGBoost 이진분류로 확률 상위 뽑는 방식과 비교.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import ndcg_score

from src.features.base_features import FEATURE_COLS_BASE
from src.data.pooled_dataset import walk_forward_splits_by_date

SEEDS = [42, 1, 7, 123, 2024]
HORIZON = 10  # feature_engineering_pooled_kr.py 생성 시 쓴 horizon과 반드시 동일하게 맞출 것
ROUND_TRIP_COST = 0.002  # quant_xgboost와 동일 (매수+매도 왕복 0.2%)

# 매일 몇 종목을 "픽"으로 볼지. 50종목 풀이면 5(=상위 10%), 200종목 풀이면 10(=상위 5%) 정도가 적당.
TOP_K = 10


N_RELEVANCE_GRADES = 5  # LightGBM lambdarank는 label을 0~30(31단계)까지만 허용 -- 5면 200종목까지도 안전


# ------------------------------------------------------------------
# 1. 날짜별 graded relevance label 생성 (5단계 등급)
# ------------------------------------------------------------------
def add_relevance_label(df: pd.DataFrame, n_grades: int = N_RELEVANCE_GRADES) -> pd.DataFrame:
    """
    같은 날짜 안에서 future_return의 순위 백분위를 n_grades단계로 등급화.
    원시 순위(0~N-1)를 그대로 쓰면 종목 수가 늘어날수록 label 범위가 커져서
    LightGBM lambdarank의 label 상한(기본 31단계)을 넘는 문제가 생김 --
    등급화하면 종목 수와 무관하게 항상 0~(n_grades-1) 범위로 고정됨.
    """
    df = df.copy()
    rank_pct = df.groupby(df.index)["future_return"].rank(pct=True, method="first")
    relevance = (rank_pct * n_grades).apply(np.floor).astype(int)
    df["relevance"] = relevance.clip(upper=n_grades - 1)
    return df


# ------------------------------------------------------------------
# 2. fold 하나 학습 + 평가
# ------------------------------------------------------------------
def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, random_state: int,
             top_k: int = TOP_K, feature_cols: list = None):
    feature_cols = feature_cols if feature_cols is not None else FEATURE_COLS_BASE
    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    X_train = train_df[feature_cols + ["ticker"]]
    X_test = test_df[feature_cols + ["ticker"]]

    train_group = train_df.groupby(train_df.index).size().values

    # -- (1) LightGBM lambdarank
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )
    ranker.fit(
        X_train, train_df["relevance"],
        group=train_group,
        categorical_feature=["ticker"],
    )
    test_df = test_df.copy()
    test_df["rank_score"] = ranker.predict(X_test)

    # -- (2) XGBoost 이진분류 baseline (quant_xgboost와 동일 하이퍼파라미터)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", random_state=random_state,
        enable_categorical=True,
    )
    clf.fit(X_train, train_df["label"])
    test_df["binary_proba"] = clf.predict_proba(X_test)[:, 1]

    # -- NDCG@top_k (참고 지표)
    ndcg_vals = []
    for date, g in test_df.groupby(test_df.index):
        if len(g) < 2:
            continue
        k = min(top_k, len(g))
        ndcg_vals.append(
            ndcg_score([g["relevance"].values], [g["rank_score"].values], k=k)
        )
    fold_ndcg = float(np.mean(ndcg_vals)) if ndcg_vals else np.nan

    # -- 날짜별 "top_k 뽑기" 비교: ranking / binary / 동일가중(baseline), gross/net(거래비용 반영) 둘 다 저장
    #    풀이 커지면(50~200종목) 1등 하나만 맞추는 건 노이즈가 커서, 상위 top_k 평균수익률로 평가.
    #    거래비용은 종목마다 동일(왕복 0.2%)하므로 평균(gross) - cost = net, 셋 다 동일하게 적용.
    daily_rows = []
    for date, g in test_df.groupby(test_df.index):
        if len(g) < 2:
            continue
        k = min(top_k, len(g))
        rank_pick_gross = g.nlargest(k, "rank_score")["future_return"].mean()
        binary_pick_gross = g.nlargest(k, "binary_proba")["future_return"].mean()
        equal_weight_gross = g["future_return"].mean()
        daily_rows.append({
            "date": date,
            "n_candidates": len(g),
            "rank_pick_gross": rank_pick_gross,
            "binary_pick_gross": binary_pick_gross,
            "equal_weight_gross": equal_weight_gross,
            "rank_pick_net": rank_pick_gross - ROUND_TRIP_COST,
            "binary_pick_net": binary_pick_gross - ROUND_TRIP_COST,
            "equal_weight_net": equal_weight_gross - ROUND_TRIP_COST,
        })
    daily_df = pd.DataFrame(daily_rows)

    return fold_ndcg, daily_df


# ------------------------------------------------------------------
# 3. 단일 seed로 전체 walk-forward 실행 -- 날짜별 daily_df + fold별 NDCG를 그대로 반환
#    (국면 집중도 분석 등 날짜 단위 상세 데이터가 필요한 곳에서 재사용)
# ------------------------------------------------------------------
def run_walk_forward_single_seed(df: pd.DataFrame, seed: int, train_days=300, test_days=60,
                                  step_days=60, embargo_days=HORIZON, top_k=TOP_K, feature_cols: list = None):
    df = add_relevance_label(df)
    splits = walk_forward_splits_by_date(df, train_days, test_days, step_days, embargo_days)
    if not splits:
        raise ValueError("데이터가 부족해서 walk-forward split을 만들 수 없어요. train_days/test_days를 줄이세요.")

    all_daily = []
    fold_ndcgs = []
    for train_dates, test_dates in splits:
        train_df = df[df.index.isin(train_dates)]
        test_df = df[df.index.isin(test_dates)]
        fold_ndcg, daily_df = run_fold(train_df, test_df, random_state=seed, top_k=top_k, feature_cols=feature_cols)
        fold_ndcgs.append(fold_ndcg)
        all_daily.append(daily_df)

    daily_all = pd.concat(all_daily, ignore_index=True)
    mean_ndcg = float(np.nanmean(fold_ndcgs))
    return daily_all, mean_ndcg


# ------------------------------------------------------------------
# 4. 전체 walk-forward 실행 (멀티 시드) -- 위 함수를 시드별로 반복 호출
# ------------------------------------------------------------------
def run_ranking_experiment(df: pd.DataFrame, train_days=300, test_days=60, step_days=60,
                            embargo_days=HORIZON, top_k=TOP_K, feature_cols: list = None):
    n_tickers = df["ticker"].nunique()
    print(f"종목풀 {n_tickers}개, top_k={top_k}, seed {len(SEEDS)}개로 실행합니다.\n")

    all_seed_summaries = []
    for seed in SEEDS:
        daily_all, mean_ndcg = run_walk_forward_single_seed(
            df, seed, train_days=train_days, test_days=test_days,
            step_days=step_days, embargo_days=embargo_days, top_k=top_k, feature_cols=feature_cols,
        )

        summary = {
            "seed": seed,
            "ndcg@k": mean_ndcg,
            "rank_pick_gross": daily_all["rank_pick_gross"].mean(),
            "binary_pick_gross": daily_all["binary_pick_gross"].mean(),
            "equal_weight_gross": daily_all["equal_weight_gross"].mean(),
            "rank_pick_net": daily_all["rank_pick_net"].mean(),
            "binary_pick_net": daily_all["binary_pick_net"].mean(),
            "equal_weight_net": daily_all["equal_weight_net"].mean(),
            "rank_vs_binary_net": daily_all["rank_pick_net"].mean() - daily_all["binary_pick_net"].mean(),
            "rank_vs_equal_weight_net": daily_all["rank_pick_net"].mean() - daily_all["equal_weight_net"].mean(),
            "n_days": len(daily_all),
        }
        all_seed_summaries.append(summary)
        print(f"[seed={seed}] NDCG@k={summary['ndcg@k']:.4f} | "
              f"랭킹픽(top{top_k}) net={summary['rank_pick_net']:.4%} | "
              f"이진분류픽(top{top_k}) net={summary['binary_pick_net']:.4%} | "
              f"동일가중 net={summary['equal_weight_net']:.4%}")

    return pd.DataFrame(all_seed_summaries)