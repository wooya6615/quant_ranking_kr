"""
LightGBM 랭킹 vs XGBoost 이진분류 vs 동일가중 baseline 비교 실행 스크립트.

전제:
    src/experiments/build_pool.py를 먼저 실행해서 data/pooled_features_kospi200.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.run_ranking_experiment
"""

from src.data.pooled_dataset import load_pooled_dataset
from src.models.lightgbm_ranker import run_ranking_experiment


if __name__ == "__main__":
    df = load_pooled_dataset()
    print(f"pooled_features_kospi200.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    summary_df = run_ranking_experiment(df)

    print("\n" + "=" * 70)
    print("=== 5-seed 평균 요약 ===")
    print("=" * 70)
    print(summary_df.set_index("seed").round(5).to_string())

    print("\n[해석 가이드] (모두 거래비용 왕복 0.2% 반영한 net 기준)")
    print("- rank_vs_equal_weight_net > 0 이 5개 seed 모두에서 일관되게 나오면:")
    print("  거래비용 감안해도 BASE 피처로 종목 선별에 엣지가 있다는 근거.")
    print("- rank_vs_binary_net > 0 이 나오면 '오늘 뭘 살지' 문제에서는")
    print("  랭킹 objective가 이진분류보다 낫다는 근거.")
    print("- rank_pick_net 자체가 음수면, gross로는 이겼어도 비용 감안하면")
    print("  실제로는 손실이라는 뜻 -- 이 경우 전략 자체를 재검토해야 함.")