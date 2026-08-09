"""
종목 -> 업종분류 코드 매핑 (시장 무관, KOSPI/KOSDAQ 둘 다 지원).

[변경 이력] 원래는 KOSPI 업종분류 지수코드(1005~1027)를 하드코딩했었는데,
코스닥은 지수코드 체계가 달라서 재사용이 안 됨. 그래서 pykrx의
get_index_ticker_list(market=...)로 그 시장에 속한 지수 코드를 동적으로 받아오고,
각 지수의 구성종목 수가 적은(=업종이 좁고 구체적인) 지수부터 먼저 매칭시키는
방식으로 바꿈 -- 코스피/코스닥 전체지수 같은 "포괄 지수"가 먼저 매칭돼서 모든
종목을 다 가져가버리는 걸 방지하기 위함 (전체지수는 구성종목 수가 압도적으로 많음).

설치:
    pip install pykrx python-dotenv
"""

from dotenv import load_dotenv

load_dotenv()
from pykrx import stock


def build_sector_map(tickers: list[str], market: str = "KOSPI") -> dict[str, dict]:
    """
    tickers: pooled 데이터셋의 ticker 값들 (CSV 왕복으로 앞자리 0이 잘린 상태, 예: "12450")
    market: "KOSPI" 또는 "KOSDAQ"

    반환: {ticker: {"sector_code": ..., "sector_name": ...}}

    [주의] pykrx가 반환하는 구성종목 코드는 6자리 zero-padded 문자열("012450")이라서,
    비교 시점에만 tickers를 6자리로 맞춰서 매칭하고, 반환 dict의 key는 입력받은
    tickers 원본 포맷(0 없음) 그대로 유지함.
    """
    padded_to_original = {t.zfill(6): t for t in tickers}

    print(f"  {market} 지수 목록 조회 중...")
    all_index_codes = stock.get_index_ticker_list(market=market)

    # [주의] get_index_ohlcv_by_date()가 최근 추가된 GICS 스타일 "코스피 200 XX" 서브지수
    # (1150번대 이상, 1894 등)에서 반복적으로 실패하는 게 확인됨 ('종가' KeyError,
    # JSON 파싱 에러 -- KRX 쪽 데이터 포맷/제공 방식이 다른 것으로 보임).
    # 전통 업종분류 코드(1005~1027)는 안정적으로 동작하는 게 검증됐으므로,
    # KOSPI는 이 범위로 제한. KOSDAQ 등 다른 시장은 필터링 없이 전체 사용.
    if market == "KOSPI":
        index_codes = [c for c in all_index_codes if c.isdigit() and 1005 <= int(c) <= 1027]
        print(f"  {market} 지수 {len(all_index_codes)}개 중 안정적인 업종분류 {len(index_codes)}개로 제한, 구성종목 조회 중...")
    else:
        index_codes = all_index_codes
        print(f"  {market} 지수 {len(index_codes)}개 발견, 구성종목 조회 중...")

    code_constituents = {}
    for code in index_codes:
        try:
            constituents = stock.get_index_portfolio_deposit_file(code)
        except Exception:
            continue
        if constituents:
            code_constituents[code] = constituents

    # 구성종목 수가 적은(=업종이 구체적인) 지수부터 매칭 -- 전체지수/규모별지수(대형주 등)처럼
    # 포괄적인 지수가 먼저 모든 종목을 가져가버리는 걸 방지
    sorted_codes = sorted(code_constituents.keys(), key=lambda c: len(code_constituents[c]))

    ticker_to_sector = {}
    for code in sorted_codes:
        matched_tickers = []
        for t in code_constituents[code]:
            original = padded_to_original.get(t)
            if original is None or original in ticker_to_sector:
                continue
            matched_tickers.append(original)

        if not matched_tickers:
            continue

        try:
            name = stock.get_index_ticker_name(code)
        except Exception:
            name = code

        for original in matched_tickers:
            ticker_to_sector[original] = {"sector_code": code, "sector_name": name}

        print(f"  [{code} {name}] 구성종목 {len(code_constituents[code])}개 중 {len(matched_tickers)}개 신규 매칭")

    missing = set(tickers) - set(ticker_to_sector.keys())
    if missing:
        print(f"\n섹터 매핑 실패한 종목: {sorted(missing)}")

    return ticker_to_sector