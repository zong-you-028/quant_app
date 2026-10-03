"""Meaningful parser and fail-closed coverage for the isolated universe builder."""
from copy import deepcopy

import pytest

from core.market_cap_universe import (
    UniverseSnapshotError, build_snapshot, fetch_snapshot, normalize_date, parse_common_shares,
)


def _html(market, records):
    return "最近更新日期:2026/10/03<table>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in
            (f"{code}　{name}", "TW0000000000", listed, market, "半導體", cfi, "")) + "</tr>"
        for code, name, listed, cfi in records) + "</table>"


def _inputs():
    return dict(asof="2026-10-02", top_n=2,
        twse_closes=[{"Date": "1151002", "Code": "2330", "ClosingPrice": "100.00"}],
        twse_shares=[{"出表日期": "1151002", "公司代號": "2330", "已發行普通股數或TDR原股發行股數": "20,000,000"}],
        tpex_values=[{"Date": "1151002", "SecuritiesCompanyCode": "6488",
                      "Capitals": "30,000,000", "ClosePrice": "100.00", "MarketValue": "3000"}],
        twse_isin_html=_html("上市", [("2330", "台積電", "1994/09/05", "ESVUFR")]),
        tpex_isin_html=_html("上櫃", [("6488", "環球晶", "2011/10/18", "ESVUFR")]))


def test_cross_market_units_and_order():
    result = build_snapshot(**_inputs())
    assert result["symbols"] == ["6488", "2330"]
    assert result["top150"][0]["market_cap_ntd"] == 3_000_000_000
    assert result["selected_market_counts"] == {"twse": 1, "tpex": 1}
    assert result["classification_dates"] == {"twse": "2026-10-03", "tpex": "2026-10-03"}


def test_official_cfi_excludes_etf_preferred_and_tdr_not_just_code_length():
    html = _html("上市", [("2330", "台積電", "1994/09/05", "ESVUFR"),
        ("0050", "ETF", "2003/06/30", "CEOGEU"),
        ("1101B", "特別股", "2019/01/01", "EPVUFR"),
        ("9103", "四碼TDR", "2009/01/01", "EDSXFR"),
        ("9999", "四碼特別股", "2019/01/01", "EPVUFR"),
        ("7999", "尚未掛牌", "2026/10/05", "ESVUFR")])
    classified, _ = parse_common_shares(html, "twse", "2026-10-02")
    assert set(classified) == {"2330"}


@pytest.mark.parametrize("field,value", [("twse_closes", []), ("tpex_values", []),
    ("twse_shares", {"stat": "fail"}), ("tpex_isin_html", "<html>error</html>")])
def test_market_or_classifier_failure_never_degrades_to_single_market(field, value):
    data = _inputs()
    data[field] = value
    with pytest.raises(UniverseSnapshotError):
        build_snapshot(**data)


def test_mixed_source_dates_rejected():
    data = _inputs()
    data["tpex_values"][0]["Date"] = "1151001"
    with pytest.raises(UniverseSnapshotError, match="TPEx dates"):
        build_snapshot(**data)


def test_missing_common_share_is_not_silently_dropped():
    data = _inputs()
    data["twse_isin_html"] = _html("上市", [("2330", "台積電", "1994/09/05", "ESVUFR"),
                                               ("2317", "鴻海", "1991/06/18", "ESVUFR")])
    with pytest.raises(UniverseSnapshotError, match="Missing common-share valuations"):
        build_snapshot(**data)


def test_explicit_priced_scope_discloses_missing_quote_without_guessing_status():
    data = _inputs()
    data["twse_isin_html"] = _html("上市", [("2330", "台積電", "1994/09/05", "ESVUFR"),
                                               ("2317", "鴻海", "1991/06/18", "ESVUFR")])
    result = build_snapshot(**data, scope="priced_ordinary")
    assert result["symbols"] == ["6488", "2330"]
    assert result["source_incomplete"] is True
    assert result["quote_scoped_pool_complete"] is True
    assert result["excluded_unpriced"][0]["symbol"] == "2317"
    assert result["excluded_unpriced"][0]["trading_status_confirmed"] is False


def test_priced_scope_cannot_hide_missing_share_count_for_quoted_stock():
    data = _inputs()
    data["twse_isin_html"] = _html("上市", [("2330", "台積電", "1994/09/05", "ESVUFR"),
                                               ("2317", "鴻海", "1991/06/18", "ESVUFR")])
    data["twse_closes"].append({"Date": "1151002", "Code": "2317", "ClosingPrice": "100"})
    with pytest.raises(UniverseSnapshotError, match="Missing TWSE issued shares"):
        build_snapshot(**data, scope="priced_ordinary")


def test_priced_scope_cannot_drop_an_entire_market_even_with_nonempty_source():
    data = _inputs()
    data["twse_closes"][0]["Code"] = "0050"
    data["top_n"] = 1
    with pytest.raises(UniverseSnapshotError, match="cannot degrade to one market"):
        build_snapshot(**data, scope="priced_ordinary")


def test_selection_boundary_is_recorded_and_estimated_label_is_explicit():
    data = _inputs()
    data["top_n"] = 1
    result = build_snapshot(**data, scope="priced_ordinary")
    assert result["boundary"]["last_selected"]["symbol"] == "6488"
    assert result["boundary"]["first_excluded"]["symbol"] == "2330"
    assert result["boundary"]["gap_ntd"] == 1_000_000_000
    assert result["boundary"]["gap_pct_of_last_selected"] == pytest.approx(100 / 3)
    assert result["estimated"] is True
    assert "估算市值" in result["display_label"]


def test_tpex_unit_mismatch_rejected():
    data = _inputs()
    data["tpex_values"][0]["MarketValue"] = "30"
    with pytest.raises(UniverseSnapshotError, match="unit/arithmetic"):
        build_snapshot(**data)


def test_official_twse_published_cap_crosscheck():
    data = _inputs()
    data["twse_check"] = {"ranking": {"market": [{"NO": "2330", "DATE": "20261002", "AMT": "20.00"}]}}
    assert build_snapshot(**data)["twse_published_cap_checks"][0]["symbol"] == "2330"
    data["twse_check"]["ranking"]["market"][0]["AMT"] = "20000"
    with pytest.raises(UniverseSnapshotError, match="cap disagrees"):
        build_snapshot(**data)


def test_duplicate_and_nonfinite_valuation_rejected():
    data = _inputs()
    data["tpex_values"].append(deepcopy(data["tpex_values"][0]))
    with pytest.raises(UniverseSnapshotError, match="Duplicate"):
        build_snapshot(**data)
    data = _inputs()
    data["twse_closes"][0]["ClosingPrice"] = "NaN"
    with pytest.raises(UniverseSnapshotError, match="Invalid"):
        build_snapshot(**data)


def test_date_parsing_is_strict_and_handles_roc():
    assert normalize_date("115/10/02") == "2026-10-02"
    assert normalize_date("20261002") == "2026-10-02"
    with pytest.raises(UniverseSnapshotError):
        normalize_date("20260230")


def test_live_fetch_cannot_overwrite_archived_research(tmp_path):
    preserved = tmp_path / "sources.json"
    preserved.write_text("preserve", encoding="utf-8")
    with pytest.raises(UniverseSnapshotError, match="already exists"):
        fetch_snapshot("2026-10-02", output_dir=tmp_path)
    assert preserved.read_text(encoding="utf-8") == "preserve"
