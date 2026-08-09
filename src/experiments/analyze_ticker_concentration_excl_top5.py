"""
상위 5종목(전력기기/방산 1군) 제외 후에도, 나머지 45종목 안에서 edge가
여전히 소수 종목(특히 같은 테마의 2군 종목)에 쏠려있는지 재확인.

배경 질문:
    run_excl_top_tickers_check.py에서 상위 5종목을 빼고도 rank_vs_equal_weight_net이
    5-seed 모두 양수로 유지됐음. 근데 이게 "진짜 넓게 분산된 선별력"인지, 아니면
    "같은 테마(방산/조선)의 2군 종목(LIG넥스원, 한화오션, HD현대마린솔루션 등)으로
    쏠림이 옮겨간 것"인지 구분이 안 됨. 이 스크립트가 45종목 풀 안에서 다시
    종목별 edge 기여도를 집계해서 그 구분을 함.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.analyze_ticker_concentration_excl_top5
"""

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import TOP_K
from src.experiments.analyze_ticker_concentration import run_walk_forward_ticker_level, SEED
from src.experiments.run_excl_top_tickers_check import EXCLUDE_TICKERS

# 방산/조선 2군 종목까지 넓게 잡아서 "테마 자체가 계속 잡히는지" 확인용 참고 리스트
THEME_TICKERS = {
    "12450": "한화에어로스페이스(방산, 이미 제외)",
    "298040": "효성중공업(전력기기, 이미 제외)",
    "10120": "LS ELECTRIC(전력기기, 이미 제외)",
    "267260": "HD현대일렉트릭(전력기기, 이미 제외)",
    "42700": "한미반도체(이미 제외)",
    "79550": "LIG넥스원(방산)",
    "42660": "한화오션(조선)",
    "329180": "HD현대마린솔루션(조선)",
    "10140": "삼성중공업(조선)",
}

if __name__ == "__main__":
    df = load_pooled_dataset()
    df["ticker"] = df["ticker"].astype(str)

    before_n = df["ticker"].nunique()
    df_filtered = df[~df["ticker"].isin(EXCLUDE_TICKERS)].copy()
    df_filtered["ticker"] = df_filtered["ticker"].astype("category")
    after_n = df_filtered["ticker"].nunique()

    print(f"원래 종목 {before_n}개 -> 상위 5종목 제외 후 {after_n}개\n")
    print(f"seed={SEED}, top_k={TOP_K}로 45종목 풀에서 다시 walk-forward 실행 중...\n")

    picks_df = run_walk_forward_ticker_level(df_filtered, seed=SEED, top_k=TOP_K)
    total_edge = picks_df["edge_contribution"].sum()

    ticker_summary = (
        picks_df.groupby("ticker")
        .agg(n_picked=("edge_contribution", "size"), total_edge_contribution=("edge_contribution", "sum"))
        .sort_values("total_edge_contribution", ascending=False)
    )
    ticker_summary["share_of_total_edge"] = ticker_summary["total_edge_contribution"] / total_edge
    ticker_summary["theme_note"] = ticker_summary.index.map(lambda t: THEME_TICKERS.get(t, ""))

    print("=" * 90)
    print("=== 45종목 풀 -- 종목별 edge 기여도 (상위 15개) ===")
    print("=" * 90)
    print(ticker_summary.head(15).round(5).to_string())

    top5_share = ticker_summary.head(5)["share_of_total_edge"].sum()
    top10_share = ticker_summary.head(10)["share_of_total_edge"].sum()

    # 상위 15위 안에 THEME_TICKERS(방산/조선 2군)가 몇 개나 다시 등장하는지 카운트
    top15_tickers = set(ticker_summary.head(15).index)
    theme_2nd_tier = {"79550", "42660", "329180", "10140"}  # 2군 방산/조선만 (이미 제외된 1군 빼고)
    reappeared = top15_tickers & theme_2nd_tier

    print(f"\n상위 5종목 비중: {top5_share:.1%}")
    print(f"상위 10종목 비중: {top10_share:.1%}")
    print(f"\n상위 15위 안에 재등장한 방산/조선 2군 종목: {[THEME_TICKERS[t] for t in reappeared] or '없음'}")

    print("\n[해석 가이드]")
    if top5_share > 0.5:
        print(f"→ 5종목을 뺐는데도 상위 5개 비중이 여전히 {top5_share:.1%} -- 소수 종목 쏠림 구조가")
        print("  그대로 반복됨. 특히 재등장한 종목이 방산/조선 2군이라면, 이건 '선별력'이 아니라")
        print("  '동일 테마를 계속 잡아내는 것'에 가까움. 테마 전체를 통째로 빼고 재검증 필요.")
    else:
        print(f"→ 상위 5개 비중이 {top5_share:.1%}로 완화됨 -- 분산이 개선된 것으로 보이나,")
        print("  재등장한 테마 종목이 있는지는 위 목록을 직접 확인할 것.")