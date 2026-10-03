import pandas as pd
import pytest

from core.tpex_prices import parse_daily, parse_month


def payload():
    return {"stat": "ok", "code": "6488", "tables": [{"fields": ["日期", "成交仟股", "成交仟元", "開盤", "最高", "最低", "收盤", "漲跌", "筆數"],
            "data": [["115/10/02", "16,256", "18,622,066", "1,090", "1,190", "1,075", "1,190", "105", "30,486"]]}]}


def test_month_unit_and_date():
    result = parse_month(payload(), "6488", "2026-10-01")
    assert result.loc[pd.Timestamp("2026-10-02"), "volume"] == 16256000
    assert result.iloc[0].close == 1190


def test_current_month_schema_uses_lots_of_1000_shares():
    source = payload()
    source["tables"][0]["fields"][1] = "成交張數"
    assert parse_month(source, "6488", "2026-10-01").iloc[0].volume == 16256000


@pytest.mark.parametrize("change", ["symbol", "unit", "date", "empty"])
def test_month_rejects_mismatch(change):
    source = payload()
    if change == "symbol": source["code"] = "2330"
    if change == "unit": source["tables"][0]["fields"][1] = "成交股數"
    if change == "date": source["tables"][0]["data"][0][0] = "115/09/30"
    if change == "empty": source["tables"][0]["data"] = []
    with pytest.raises(RuntimeError):
        parse_month(source, "6488", "2026-10-01")


def test_daily_uses_shares_and_omits_suspended():
    row = {"SecuritiesCompanyCode": "6488", "Date": "1151002", "Open": "1090", "High": "1190", "Low": "1075", "Close": "1190", "TradingShares": "16256593"}
    frame = parse_daily([row, dict(row, SecuritiesCompanyCode="1234", Close="--")])
    assert frame.symbol.tolist() == ["6488"]
    assert frame.iloc[0].volume == 16256593


def test_daily_rejects_mixed_dates():
    row = {"SecuritiesCompanyCode": "6488", "Date": "1151002", "Open": "1090", "High": "1190", "Low": "1075", "Close": "1190", "TradingShares": "1"}
    with pytest.raises(RuntimeError):
        parse_daily([row, dict(row, SecuritiesCompanyCode="1234", Date="1151001")])


def test_nonfinite_prices_cannot_enter_daily_cache():
    row = {"SecuritiesCompanyCode": "6488", "Date": "1151002", "Open": "1090", "High": "inf", "Low": "1075", "Close": "1190", "TradingShares": "1"}
    assert parse_daily([row]).empty
