"""
BASE(13개) vs COMBINED(BASE+섹터 6개) 랭킹 모델 비교.
(quant_xgboost의 train_xgboost_ablation.py 패턴을 랭킹 프레임에 맞게 재구성)

핵심 설계:
    같은 데이터, 같은 fold 구성, 같은 하이퍼파라미터로 BASE만 vs BASE+SECTOR만
    비교함 -- 그래야 성능 차이가 순수하게 "섹터 피처 추가 효과"인지 확인 가능.

전제:
    src/experiments/build_pool_with_sector.py를 먼저 실행해서
    pooled_features_kospi200_with_sector.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.run_sector_ablation
"""

import pandas as pd
import numpy as np

from src.data.pooled_dataset import DATA_DIR
from src.features.base_features import FEATURE_COLS_BASE
from src.features.sector_features import FEATURE_COLS_SECTOR
from src.models.lightgbm_ranker import run_walk_forward_single_seed, SEEDS, TOP_K

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_SECTOR


def load_sector_dataset(filename: str = "pooled_features_kospi200_with_sector.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path, index_col=0, parse_dates=True, dtype={"ticker": str})
    required = FEATURE_COLS_COMBINED + ["ticker", "future_return", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}에 다음 컬럼이 없습니다: {missing}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS_COMBINED + ["future_return", "label"])
    df["ticker"] = df["ticker"].astype("category")
    df = df.sort_index()
    return df


def summarize(daily_all: pd.DataFrame) -> dict:
    return {
        "rank_pick_net": daily_all["rank_pick_net"].mean(),
        "equal_weight_net": daily_all["equal_weight_net"].mean(),
        "rank_vs_equal_weight_net": daily_all["rank_pick_net"].mean() - daily_all["equal_weight_net"].mean(),
        "n_days": len(daily_all),
    }


if __name__ == "__main__":
    df = load_sector_dataset()
    print(f"pooled_features_kospi200_with_sector.csv 로드 완료: {df.shape[0]}행, "
          f"종목 {df['ticker'].nunique()}개, {df.index.min().date()} ~ {df.index.max().date()}\n")

    feature_sets = {"BASE": FEATURE_COLS_BASE, "COMBINED": FEATURE_COLS_COMBINED}
    all_results = {}

    for label, cols in feature_sets.items():
        print(f"=== {label} ({len(cols)}개 피처) 실행 중 ===")
        rows = []
        for seed in SEEDS:
            daily_all, _ = run_walk_forward_single_seed(df, seed=seed, top_k=TOP_K, feature_cols=cols)
            summary = summarize(daily_all)
            summary["seed"] = seed
            rows.append(summary)
            print(f"  [seed={seed}] rank_vs_equal_weight_net={summary['rank_vs_equal_weight_net']:.4%}")
        all_results[label] = pd.DataFrame(rows).set_index("seed")
        print()

    print("=" * 80)
    print("=== BASE vs COMBINED 비교 ===")
    print("=" * 80)
    compare = pd.DataFrame({
        "BASE_rank_vs_ew": all_results["BASE"]["rank_vs_equal_weight_net"],
        "COMBINED_rank_vs_ew": all_results["COMBINED"]["rank_vs_equal_weight_net"],
    })
    compare["COMBINED_minus_BASE"] = compare["COMBINED_rank_vs_ew"] - compare["BASE_rank_vs_ew"]
    print(compare.round(5).to_string())

    n_improved = (compare["COMBINED_minus_BASE"] > 0).sum()
    print(f"\nCOMBINED가 BASE보다 나은 seed 수: {n_improved}/5")

    print("\n[해석 가이드]")
    if n_improved >= 4:
        print("→ 섹터 피처 추가가 5개 seed 중 대부분에서 개선 -- 다음은 반드시")
        print("  analyze_ticker_concentration.py를 COMBINED 결과에도 돌려서, 이번엔")
        print("  진짜 분산된 개선인지 처음부터 확인할 것 (이전처럼 나중에 발견하지 말고).")
    else:
        print("→ 섹터 피처가 뚜렷한 개선을 주지 못함 -- 섹터 강도 자체도 결국 가격")
        print("  파생이라 근본적인 모멘텀 의존성은 해결 안 됐을 가능성. 다음은 밸류에이션처럼")
        print("  가격과 독립적인 정보원으로 넘어가는 게 나을 수 있음.")