# quant_ranking_kr

> **결론 (2026-08)**: 랭킹 objective(LightGBM lambdarank)가 이진분류(XGBoost)보다
> 낫다는 근거는 2024~2025 특정 테마주(전력기기/방산/조선) 쏠림에서 나온 착시로 확인됨.
> 국면/종목 집중도 검증 후에는 일반화 가능한 우위로 보기 어려움. 상세 내용은
> [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) 참고.

`quant_xgboost` / `quant_lightgbm`은 전부 "개별 종목이 오를지 내릴지"를 독립적으로
맞추는 이진분류 문제였음. 이 레포는 문제 설정 자체를 바꿔서, "오늘 여러 종목 중
어떤 걸 살지"를 상대 순위(ranking)로 학습하는 실험을 함.

## 배경 문제

지금까지의 이진분류 접근은 두 가지 한계가 있었음:
1. 종목별 방향성 예측이 개별적으로는 맞아도, 실전 트레이딩은 "오늘의 최선의 선택"을
   매일 골라야 하는 문제에 더 가까움 -- 이진분류는 이걸 직접 최적화하지 않음.
2. 지금까지의 백테스트에서 BASE/COMBINED 둘 다 Buy & Hold를 못 이긴 구조적 한계가
   있었는데(quant_xgboost PROJECT_SUMMARY.md 참고), 이게 feature 문제인지
   "문제 설정(방향 예측)" 자체의 한계인지 분리해서 보고 싶었음.

## 방법론

- **relevance label**: 같은 날짜에 존재하는 종목들의 `future_return`을 오름차순 순위로
  매겨서 0 ~ (그날 종목 수 - 1)의 graded relevance로 사용. "오를지 내릴지"가 아니라
  "그날 상대적으로 뭐가 나았는지"를 학습시키는 게 목적.
- **모델**: LightGBM `lambdarank` objective (LGBMRanker). 같은 날짜(query group) 안에서
  종목들을 상대 순위로 학습.
- **baseline 비교**: 같은 feature/하이퍼파라미터로 학습한 XGBoost 이진분류(quant_xgboost와
  동일 설정)와, 그냥 동일가중으로 다 사는 baseline까지 3자 비교.
- **평가**: NDCG는 참고용이고, 핵심 지표는 "매일 모델이 상위 TOP_K로 뽑은 종목들의
  평균 실제 수익률" -- 종목 풀이 커질수록(50~200종목) top-1 하나만 보는 건 노이즈가
  커서 top-k 평균으로 봄.
- **feature는 quant_xgboost와 동일한 BASE 13개**를 그대로 사용 -- 그래야 성능 차이가
  순수하게 "이진분류 -> 랭킹 objective 전환 효과"인지 확인 가능. feature 확장은
  이 실험으로 방향이 검증된 이후 단계.

## 구조

```
src/
  data/
    kospi200_universe.py   # pykrx로 KOSPI200 구성종목 코드 조회 (1회성 호출)
    pooled_dataset.py       # 풀링 CSV 로드 + 날짜 기준 walk-forward 분할
  features/
    base_features.py        # quant_xgboost의 BASE 13개 feature + label 생성 이식
  models/
    lightgbm_ranker.py       # LGBMRanker 학습/평가 + XGBoost baseline 비교 로직
  experiments/
    build_pool.py            # KOSPI200 풀링 데이터셋 생성 실행
    run_ranking_experiment.py # 랭킹 vs 이진분류 vs 동일가중 비교 실행
data/                        # 생성되는 CSV (gitignore 처리)
docs/
  PROJECT_SUMMARY.md         # (실험 진행하며 채울 예정)
```

## 셋업

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install yfinance pandas numpy lightgbm xgboost scikit-learn pykrx python-dotenv
```

`.env` 파일 (레포 루트, git 커밋 금지):
```
KRX_ID=본인_krx_아이디
KRX_PW=본인_krx_비밀번호
```

## 실행 순서

```bash
python -m src.experiments.build_pool                # 50종목으로 pooled_features_kospi200.csv 생성
python -m src.experiments.run_ranking_experiment     # LightGBM 랭킹 vs XGBoost 이진분류 vs 동일가중, 5-seed 비교
```

**확인할 것**
- `build_pool.py` 실행 시 실패 종목이 20개 이상이면 yfinance rate limit 가능성 -- `sleep_sec`을 0.5~1.0으로 올릴 것.
- 결과 해석: `rank_vs_equal_weight`가 5개 seed 모두 양수면 BASE 피처로도 종목 선별에
  엣지가 있다는 뜻. `rank_vs_binary`가 양수면 이진분류보다 랭킹 objective가 실제로
  유리하다는 근거.
- 50종목 검증 끝나면 `build_pooled_dataset_kospi200(n_tickers=None)`으로 전체 200종목
  재실행해서 스케일 확인.

## 다음 단계 후보 (검증되면)

- feature 확장: 미시구조(호가/체결강도), DART 공시 내용 분류(호재/악재), 애널리스트
  추정치 변경 -- "저유동성 종목일수록 신호가 강하다"는 quant_xgboost 가설을
  랭킹 문제로도 재검증
- 예측 확률/점수 기반 포지션 사이징 (지금은 top-k 균등 픽만 봄)