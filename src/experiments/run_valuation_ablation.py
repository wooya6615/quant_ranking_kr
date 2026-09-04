"""
BASE(13개) vs COMBINED(BASE+밸류에이션 6개) 랭킹 모델 비교.

[수정 2026-09] valuation_features.py의 PER<=0 NaN 처리 수정에 맞춰,
load_valuation_dataset()의 dropna에서 per/is_loss/per_zscore_252d를 제외함.
기존엔 FEATURE_COLS_COMBINED 전체를 dropna 대상으로 넣어서, quant_xgboost의
PER=0 버그와 정확히 같은 방식으로(적자연도가 legitimate NaN이 됐는데 그 행을
통째로 드롭) 데이터가 조용히 걸러지고 있었음 -- 이게 기존 "[폐기]"(COMBINED
5/5 세션 다 BASE보다 나쁨) 판정의 원인이었을 가능성이 있어 재검증함.
LightGBM은 XGBoost와 마찬가지로 결측치를 네이티브로 처리하므로(use_missing=True
기본값), per 계열 NaN은 dropna 대신 모델의 결측치 분기 처리에 맡김.

전제:
    src/experiments/build_pool_with_valuation.py를 먼저 실행해서(수정본으로)
    pooled_features_kospi200_with_valuation.csv가 있어야 함.

사용법 (레포 루트에서):
    python -m src.experiments.run_valuation_ablation
"""

import pandas as pd
import numpy as np

from src.data.pooled_dataset import DATA_DIR
from src.features.base_features import FEATURE_COLS_BASE
from src.features.valuation_features import FEATURE_COLS_VALUATION
from src.models.lightgbm_ranker import run_walk_forward_single_seed, SEEDS, TOP_K

FEATURE_COLS_COMBINED = FEATURE_COLS_BASE + FEATURE_COLS_VALUATION

# [수정] per/is_loss/per_zscore_252d는 적자연도 legitimate NaN이라 dropna 대상에서 제외
# (LightGBM 네이티브 결측치 처리에 맡김). pbr/div/pbr_zscore_252d는 이 문제가 없어
# (장부가 기준이라 적자여도 정상 계산) 그대로 필수 dropna 유지.
COLS_FOR_DROPNA = FEATURE_COLS_BASE + ["pbr", "div", "pbr_zscore_252d"] + ["future_return", "label"]


def load_valuation_dataset(filename: str = "pooled_features_kospi200_with_valuation.csv") -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path, index_col=0, parse_dates=True, dtype={"ticker": str})
    required = FEATURE_COLS_COMBINED + ["ticker", "future_return", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}에 다음 컬럼이 없습니다: {missing}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=COLS_FOR_DROPNA)
    df["ticker"] = df["ticker"].astype("category")
    df = df.sort_index()

    print(f"is_loss=1(적자연도) 비율: {df['is_loss'].mean():.1%} "
          f"(이 행들도 per=NaN인 채로 학습에 포함됨 -- 기존엔 여기서 다 빠졌었음)")
    return df


def summarize(daily_all: pd.DataFrame) -> dict:
    return {
        "rank_pick_net": daily_all["rank_pick_net"].mean(),
        "equal_weight_net": daily_all["equal_weight_net"].mean(),
        "rank_vs_equal_weight_net": daily_all["rank_pick_net"].mean() - daily_all["equal_weight_net"].mean(),
        "n_days": len(daily_all),
    }


if __name__ == "__main__":
    df = load_valuation_dataset()
    print(f"pooled_features_kospi200_with_valuation.csv 로드 완료: {df.shape[0]}행, "
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
    print("=== BASE vs COMBINED(밸류에이션) 비교 [재검증: PER=0 버그 수정 후] ===")
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
    print("- 이전 판정(0/5, [폐기])과 다르게 나온다면 -- PER=0 버그가 정말 결론을 뒤집었던")
    print("  것. 개선된 쪽이면 [폐기] 철회하고 다음 단계(국면/종목 집중도 진단)로 진행할 것.")
    print("- 여전히 대부분 마이너스면 -- 버그와 무관하게 원래 결론([폐기])이 맞았던 것으로")
    print("  확정.")
    print("\n결과와 상관없이 반드시 run_valuation_full_diagnostics.py로 국면/종목 집중도를")
    print("처음부터 같이 확인할 것 (섹터 실험 때 배운 교훈 -- 평균 개선만 보고 끝내지 않기).")