"""
KOSDAQ 시가총액 상위 N종목 조회.
(src/data/kospi200_universe.py와 동일한 패턴, market만 KOSDAQ으로)

설치:
    pip install pykrx python-dotenv
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock


def get_top_kosdaq_tickers(n: int = 50, date: str = None) -> list[str]:
    """
    시가총액 기준 KOSDAQ 상위 n종목의 6자리 코드 리스트를 반환.
    date: YYYYMMDD 형식. None이면 최근 영업일 자동 사용.
    """
    if date is None:
        date = stock.get_nearest_business_day_in_a_week()

    df = stock.get_market_cap_by_ticker(date, market="KOSDAQ")
    if df.empty:
        raise RuntimeError(f"{date} 기준 KOSDAQ 시가총액 데이터가 비어있습니다.")

    df = df.sort_values("시가총액", ascending=False)
    tickers = df.index.tolist()[:n]
    print(f"KOSDAQ 시가총액 상위 {len(tickers)}종목 조회 완료 (기준일: {date})")
    return tickers