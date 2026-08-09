"""
랭킹픽(top_k)의 동일가중 대비 우위(edge)가 특정 연도에 몰려있는지 연도별로 분해.
(quant_xgboost의 analyze_signal_concentration.py 패턴을 이 실험 구조에 맞게 재구성)

핵심 차이점:
    quant_xgboost는 "겹치지 않는 개별 거래"들을 순차 복리로 묶어서 연도별 기여도를
    계산했음 (단일 종목, 거래 사이 겹침 없음).
    이 실험은 매일 여러 종목 중 top_k를 새로 뽑는 구조라 거래가 서로 겹침(horizon=10일인데
    매일 리밸런싱) -- 그래서 "복리 수익률"이 아니라 "일별 edge(랭킹픽 net - 동일가중 net)의
    평균"을 연도별로 비교하는 방식으로 집중도를 봄. 특정 연도의 평균 edge가 비정상적으로
    크면(혹은 음수면) 그 국면에 결과가 좌우된다는 뜻.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.analyze_concentration
"""

import pandas as pd

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import run_walk_forward_single_seed, TOP_K

SEED = 42  # 국면 집중도 확인은 대표 seed 하나로 충분 (멀티시드 평균 결과와 방향이 같은지만 보면 됨)


def analyze_by_year(daily_all: pd.DataFrame) -> pd.DataFrame:
    daily_all = daily_all.copy()
    daily_all["date"] = pd.to_datetime(daily_all["date"])
    daily_all["year"] = daily_all["date"].dt.year
    daily_all["edge"] = daily_all["rank_pick_net"] - daily_all["equal_weight_net"]

    rows = []
    for year, g in daily_all.groupby("year"):
        rows.append({
            "year": year,
            "n_days": len(g),
            "avg_edge": g["edge"].mean(),
            "avg_rank_pick_net": g["rank_pick_net"].mean(),
            "avg_equal_weight_net": g["equal_weight_net"].mean(),
        })
    year_df = pd.DataFrame(rows).sort_values("year")

    # 연도별 기여도 = 그 해 평균 edge * 그 해 일수 (기간 가중), 전체 대비 비중으로 정규화
    year_df["weighted_contribution"] = year_df["avg_edge"] * year_df["n_days"]
    total_contribution = year_df["weighted_contribution"].sum()
    year_df["share_of_total_edge"] = year_df["weighted_contribution"] / total_contribution

    return year_df


if __name__ == "__main__":
    df = load_pooled_dataset()
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    print(f"seed={SEED}, top_k={TOP_K}로 전체 walk-forward 실행 중...\n")
    daily_all, mean_ndcg = run_walk_forward_single_seed(df, seed=SEED, top_k=TOP_K)

    year_df = analyze_by_year(daily_all)

    total_edge = (daily_all["rank_pick_net"] - daily_all["equal_weight_net"]).mean()
    print(f"\n전체 기간 평균 edge(랭킹픽 net - 동일가중 net): {total_edge:.4%}\n")
    print("=" * 80)
    print("=== 연도별 분해 ===")
    print("=" * 80)
    print(year_df.round(5).to_string(index=False))

    top_year = year_df.loc[year_df["share_of_total_edge"].idxmax()]
    print(f"\n최대 기여 연도: {int(top_year['year'])}년 "
          f"(전체 edge 기여도의 {top_year['share_of_total_edge']:.1%}, "
          f"그 해 평균 edge {top_year['avg_edge']:.4%})")

    if top_year["share_of_total_edge"] > 0.5:
        print("→ ⚠️ 특정 연도 하나가 전체 edge의 절반 이상을 차지함 -- 국면 의존적 신호일 가능성 높음")
    else:
        print("→ 여러 연도에 고르게 분산됨 -- 특정 구간 우연에 기댄 결과는 아닌 것으로 보임")

    negative_years = year_df[year_df["avg_edge"] < 0]
    if len(negative_years) > 0:
        print(f"\n[참고] edge가 음수였던 연도: {negative_years['year'].tolist()} "
              f"-- 이 해들엔 랭킹픽이 동일가중보다 오히려 나빴다는 뜻")