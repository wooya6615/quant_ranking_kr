"""
edge에 가장 크게 기여한 상위 종목들(테마 쏠림 종목)을 유니버스에서 제외하고,
나머지 종목들만으로 랭킹 objective가 여전히 이진분류/동일가중보다 나은지 재검증.

배경 질문:
    analyze_ticker_concentration.py에서 상위 5종목이 전체 edge의 109.7%를 차지한다는 게
    확인됨 (전력기기/방산/조선 테마 쏠림). 이게 "그 몇 종목 빼면 나머지 45종목에서는
    선별력이 전혀 없다"는 뜻인지, 아니면 "상위권은 테마 쏠림이지만 나머지에서도 약하게나마
    선별력이 남아있다"는 뜻인지 구분이 안 됐음. 이 스크립트가 그 구분을 함.

방법:
    analyze_ticker_concentration.py 결과에서 edge 기여 상위 N종목의 티커를 하드코딩으로
    받아서, pooled 데이터에서 그 종목들을 제외한 뒤 5-seed 전체 walk-forward를 다시 실행.

전제:
    src/experiments/analyze_ticker_concentration.py를 먼저 돌려서 상위 기여 종목 목록을
    확인해야 함 (아래 EXCLUDE_TICKERS 기본값은 지금까지 확인된 상위 5종목).

사용법 (레포 루트에서):
    python -m src.experiments.run_excl_top_tickers_check
"""

import pandas as pd

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import run_walk_forward_single_seed, SEEDS, TOP_K

# analyze_ticker_concentration.py 결과 기준 -- edge 기여 상위 5종목 (전력기기/방산 위주)
EXCLUDE_TICKERS = ["12450", "298040", "10120", "267260", "42700"]


def summarize(daily_all: pd.DataFrame) -> dict:
    return {
        "rank_pick_net": daily_all["rank_pick_net"].mean(),
        "binary_pick_net": daily_all["binary_pick_net"].mean(),
        "equal_weight_net": daily_all["equal_weight_net"].mean(),
        "rank_vs_binary_net": daily_all["rank_pick_net"].mean() - daily_all["binary_pick_net"].mean(),
        "rank_vs_equal_weight_net": daily_all["rank_pick_net"].mean() - daily_all["equal_weight_net"].mean(),
        "n_days": len(daily_all),
    }


if __name__ == "__main__":
    df = load_pooled_dataset()
    # pooled CSV에는 티커가 앞자리 0 없이 저장됨 (예: "012450" -> 12450) --
    # analyze_ticker_concentration.py 출력과 동일한 포맷(문자열, 0 없음)으로 맞춰서 비교
    df["ticker"] = df["ticker"].astype(str)

    before_n = df["ticker"].nunique()
    df_filtered = df[~df["ticker"].isin(EXCLUDE_TICKERS)].copy()
    df_filtered["ticker"] = df_filtered["ticker"].astype("category")
    after_n = df_filtered["ticker"].nunique()

    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, 원래 종목 {before_n}개")
    print(f"제외 대상: {EXCLUDE_TICKERS}")
    print(f"제외 후 종목 {after_n}개, {df_filtered.shape[0]}행\n")

    rows = []
    for seed in SEEDS:
        daily_all, _ = run_walk_forward_single_seed(df_filtered, seed=seed, top_k=TOP_K)
        summary = summarize(daily_all)
        summary["seed"] = seed
        rows.append(summary)
        print(f"[seed={seed}] rank_vs_equal_weight_net={summary['rank_vs_equal_weight_net']:.4%} | "
              f"rank_vs_binary_net={summary['rank_vs_binary_net']:.4%} | "
              f"rank_pick_net={summary['rank_pick_net']:.4%}")

    result_df = pd.DataFrame(rows).set_index("seed")

    print("\n" + "=" * 80)
    print(f"=== 상위 {len(EXCLUDE_TICKERS)}종목 제외 후 5-seed 결과 ===")
    print("=" * 80)
    print(result_df.round(5).to_string())

    n_positive_vs_ew = (result_df["rank_vs_equal_weight_net"] > 0).sum()
    n_positive_vs_binary = (result_df["rank_vs_binary_net"] > 0).sum()

    print(f"\nrank_vs_equal_weight_net 양수인 seed 수: {n_positive_vs_ew}/5")
    print(f"rank_vs_binary_net 양수인 seed 수: {n_positive_vs_binary}/5")

    print("\n[해석 가이드]")
    if n_positive_vs_ew >= 4 and n_positive_vs_binary >= 4:
        print("→ 테마 쏠림 종목을 빼도 우위가 유지됨 -- 나머지 종목에도 약하게나마 진짜")
        print("  선별력이 있다는 근거. 이 경우 quant_xgboost의 수급/밸류에이션 피처를")
        print("  이 랭킹 프레임에 이식하는 다음 실험이 타당함.")
    else:
        print("→ 테마 쏠림 종목을 빼면 우위가 사라짐/불안정해짐 -- 이 프레임(BASE 피처 +")
        print("  lambdarank + 10일 horizon + top5)은 일반적인 종목 선별력이 없고,")
        print("  전 실험의 성공은 순전히 몇 개 테마주 우연 포착이었다는 결론이 더 힘을 얻음.")
        print("  이 경우 섹터 모멘텀을 명시적 feature로 인정하고 문제 설정 자체를")
        print("  바꾸는 쪽(섹터 로테이션 전략)이 더 합리적인 다음 스텝.")