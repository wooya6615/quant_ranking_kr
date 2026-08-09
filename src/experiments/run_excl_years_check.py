"""
2024~2025년(전체 edge 기여도의 97%를 차지한 두 해)을 제외하고 재검증.
(quant_xgboost의 backtest_valuation_excl_2025.py / backtest_fx_excl_regime_year.py 패턴을
 이 실험 구조에 맞게 재구성 -- 5-seed 전체로 확장해서 "그 두 해를 빼도 랜덤성과 무관하게
 결과가 뒤집히는지"까지 같이 봄)

핵심 질문:
    analyze_concentration.py에서 2024~2025 두 해가 전체 edge의 97%를 차지한다는 걸 확인함.
    이 두 해를 빼고 나머지 9개 연도만으로 다시 계산했을 때도 rank_vs_equal_weight_net이
    양수로 남는지 -- 남으면 "강세장 두 해가 크게 도왔을 뿐 나머지 기간에도 최소한의 엣지는
    있다"는 뜻이고, 음수로 뒤집히면 "이 결과는 사실상 2024~2025 두 해 발견한 것에 불과하다"는
    뜻이라 훨씬 조심스럽게 봐야 함.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.run_excl_years_check
"""

import pandas as pd

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import run_walk_forward_single_seed, SEEDS, TOP_K

EXCLUDE_YEARS = [2024, 2025]  # analyze_concentration.py 결과 기준 -- 전체 edge의 97%를 차지한 두 해


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
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"제외 대상 연도: {EXCLUDE_YEARS}\n")

    rows = []
    for seed in SEEDS:
        daily_all, _ = run_walk_forward_single_seed(df, seed=seed, top_k=TOP_K)
        daily_all = daily_all.copy()
        daily_all["year"] = pd.to_datetime(daily_all["date"]).dt.year

        full = summarize(daily_all)
        excl = summarize(daily_all[~daily_all["year"].isin(EXCLUDE_YEARS)])

        rows.append({
            "seed": seed,
            "full_rank_vs_equal_weight_net": full["rank_vs_equal_weight_net"],
            "excl_rank_vs_equal_weight_net": excl["rank_vs_equal_weight_net"],
            "full_rank_vs_binary_net": full["rank_vs_binary_net"],
            "excl_rank_vs_binary_net": excl["rank_vs_binary_net"],
            "full_rank_pick_net": full["rank_pick_net"],
            "excl_rank_pick_net": excl["rank_pick_net"],
            "excl_n_days": excl["n_days"],
        })
        print(f"[seed={seed}] 전체 rank_vs_equal_weight_net={full['rank_vs_equal_weight_net']:.4%} "
              f"-> 2024~25 제외 후={excl['rank_vs_equal_weight_net']:.4%}")

    result_df = pd.DataFrame(rows).set_index("seed")

    print("\n" + "=" * 80)
    print("=== 5-seed 전체 vs 2024~2025 제외 비교 ===")
    print("=" * 80)
    print(result_df.round(5).to_string())

    n_positive_full = (result_df["full_rank_vs_equal_weight_net"] > 0).sum()
    n_positive_excl = (result_df["excl_rank_vs_equal_weight_net"] > 0).sum()
    print(f"\nrank_vs_equal_weight_net 양수인 seed 수: "
          f"전체 기간 {n_positive_full}/5 -> 2024~25 제외 후 {n_positive_excl}/5")

    if n_positive_excl == 5:
        print("→ 2024~25를 빼도 5개 seed 모두 양수 -- 강세장이 크게 거들었을 뿐, "
              "나머지 기간에도 최소한의 엣지는 남아있는 것으로 보임.")
    elif n_positive_excl == 0:
        print("→ 2024~25를 빼면 5개 seed 모두 음수로 뒤집힘 -- 이 결과는 사실상 "
              "2024~2025 두 해에서 발견한 패턴이라고 보는 게 정확함. 일반화 가능한 엣지로 "
              "보기 어려움.")
    else:
        print(f"→ 2024~25를 빼면 seed에 따라 부호가 갈림({n_positive_excl}/5 양수) -- "
              "제외 후 결과 자체가 불안정하다는 뜻이라, 엣지가 있다고 결론 내리기 어려움.")