"""
KOSPI200 구성종목 코드 조회.
개별 종목 데이터가 아니라 "구성종목 리스트"만 받는 호출이라 1회성이고 가벼움 --
수급/밸류에이션 실험 때 겪은 pykrx IP 블로킹/rate limit 이슈와는 성격이 다름.

[주의] pykrx 공식 예제 기준으로 get_index_portfolio_deposit_file()은
ticker가 첫 번째, date가 선택적 두 번째 인자다 (date, ticker 순서 아님):
    stock.get_index_portfolio_deposit_file("1005")  # date 생략 시 최근 영업일 자동 사용

설치:
    pip install pykrx python-dotenv

사전 준비:
    레포 루트에 .env 파일 (KRX_ID/KRX_PW) -- quant_xgboost와 동일.
    pykrx가 2025년 12월 KRX 회원제 전환 이후 로그인 방식으로 바뀌었으므로,
    load_dotenv()는 반드시 pykrx import보다 먼저 실행해야 함.
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock

KOSPI200_INDEX_CODE = "1028"  # pykrx 지수 코드: 코스피 200


def get_kospi200_tickers(date: str = None) -> list[str]:
    """
    date: YYYYMMDD 형식. None이면 pykrx가 내부적으로 최근 영업일을 자동 사용.
    반환: 6자리 종목코드 리스트 (예: ["005930", "000660", ...])
    """
    tickers = stock.get_index_portfolio_deposit_file(KOSPI200_INDEX_CODE, date)
    print(f"KOSPI200 구성종목 {len(tickers)}개 조회 완료")
    return tickers