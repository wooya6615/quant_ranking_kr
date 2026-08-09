"""
COMBINED(BASE+SECTOR) 랭킹 결과를 국면(연도)/종목 집중도까지 한 번에 검증 -- KOSDAQ 버전.
(run_sector_full_diagnostics.py와 동일한 구조)

사용법 (레포 루트에서):
    python -m src.experiments.run_kosdaq_sector_full_diagnostics
"""

from src.models.lightgbm_ranker import run_walk_forward_single_seed, TOP_K
from src.experiments.analyze_concentration import analyze_by_year, SEED
from src.experiments.analyze_ticker_concentration import run_walk_forward_ticker_level
from src.experiments.run_kosdaq_sector_ablation import load_kosdaq_sector_dataset, FEATURE_COLS_COMBINED


if __name__ == "__main__":
    df = load_kosdaq_sector_dataset()
    print(f"pooled_features_kosdaq_with_sector.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    print(f"seed={SEED}, top_k={TOP_K}, COMBINED({len(FEATURE_COLS_COMBINED)}개 피처)로 실행 중...\n")

    daily_all, _ = run_walk_forward_single_seed(df, seed=SEED, top_k=TOP_K, feature_cols=FEATURE_COLS_COMBINED)
    year_df = analyze_by_year(daily_all)
    total_edge = (daily_all["rank_pick_net"] - daily_all["equal_weight_net"]).mean()

    print("=" * 80)
    print("=== [1] 연도별 edge 분해 (KOSDAQ, COMBINED) ===")
    print("=" * 80)
    print(f"전체 기간 평균 edge: {total_edge:.4%}\n")
    print(year_df.round(5).to_string(index=False))

    top_year = year_df.loc[year_df["share_of_total_edge"].idxmax()]
    print(f"\n최대 기여 연도: {int(top_year['year'])}년 (전체 edge 기여도의 {top_year['share_of_total_edge']:.1%})")
    year_flag = top_year["share_of_total_edge"] > 0.5
    print("→ ⚠️ 특정 연도 집중" if year_flag else "→ 특정 연도 집중 아님")

    print("\n" + "=" * 80)
    print("=== [2] 종목별 edge 분해 (KOSDAQ, COMBINED) ===")
    print("=" * 80)
    picks_df = run_walk_forward_ticker_level(df, seed=SEED, top_k=TOP_K, feature_cols=FEATURE_COLS_COMBINED)
    total_pick_edge = picks_df["edge_contribution"].sum()

    ticker_summary = (
        picks_df.groupby("ticker")
        .agg(n_picked=("edge_contribution", "size"), total_edge_contribution=("edge_contribution", "sum"))
        .sort_values("total_edge_contribution", ascending=False)
    )
    ticker_summary["share_of_total_edge"] = ticker_summary["total_edge_contribution"] / total_pick_edge

    print(ticker_summary.head(10).round(5).to_string())

    top5_share = ticker_summary.head(5)["share_of_total_edge"].sum()
    print(f"\n상위 5종목 비중: {top5_share:.1%}")
    ticker_flag = top5_share > 0.5

    print("\n" + "=" * 80)
    print("=== 최종 판정 ===")
    print("=" * 80)
    if not year_flag and not ticker_flag:
        print("→ 연도/종목 둘 다 분산됨 -- KOSDAQ COMBINED 개선은 일반화 가능한 신호일 가능성이 높음.")
    elif year_flag and ticker_flag:
        print("→ 연도/종목 둘 다 쏠려있음 -- KOSPI 실험 때와 같은 패턴 반복.")
    else:
        print("→ 연도/종목 중 한쪽만 개선됨 -- 부분적 개선.")