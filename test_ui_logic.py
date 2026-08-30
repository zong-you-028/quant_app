# -*- coding: utf-8 -*-
"""
test_ui_logic.py - 無 GUI 驗證「個股分析(策略適配檢查)」的 UI 更新邏輯
建立與 main() 同名的控件,跑 rotation.analyze_stock,呼叫 apply_fit,
斷言:適配大卡底色已變、KPI(動能排名/絕對動能/波動)已填、6 個月走勢圖已建立。
(個股分析已從「沒 edge 的 ML 燈號」改為「這檔符不符合輪動策略」的檢查工具)
"""
import flet as ft

from main import apply_fit
from core import rotation


def build_ui():
    """重建 main() 中需被 apply_fit 更新的控件集合。"""
    def T():
        return ft.Text("--")
    ui = {k: T() for k in ("signal_name", "signal_sub", "price_val", "price_chg",
                           "price_date", "note_hint", "rank_val", "abs_val",
                           "vol_val", "mom_line")}
    ui["signal_card"] = ft.Container(bgcolor="#9E9E9E")
    ui["price_holder"] = ft.Container(content=ft.Text("近 5 日"))
    ui["chart_holder"] = ft.Container(content=ft.Text("近 6 個月走勢"))
    return ui


def main():
    ui = build_ui()
    before_color = ui["signal_card"].bgcolor

    res = rotation.analyze_stock("2330")        # 模擬按下「分析這檔」
    apply_fit(ui, res)

    print("=== 個股分析(策略適配)UI 驗證 ===")
    print(f"適配大卡底色 : {before_color} -> {ui['signal_card'].bgcolor}")
    print(f"判定         : {ui['signal_name'].value}")
    print(f"副標         : {ui['signal_sub'].value}")
    print(f"說明         : {ui['note_hint'].value}")
    print(f"動能排名     : {ui['rank_val'].value}")
    print(f"絕對動能     : {ui['abs_val'].value} (color={ui['abs_val'].color})")
    print(f"近60日波動   : {ui['vol_val'].value}")
    print(f"當前股價     : {ui['price_val'].value} ({ui['price_chg'].value})")
    print(f"動能脈絡     : {ui['mom_line'].value}")
    print(f"5日曲線型別  : {type(ui['price_holder'].content).__name__}")
    print(f"6個月圖型別  : {type(ui['chart_holder'].content).__name__}")

    # --- 斷言:適配大卡顏色已設為判定色(非預設灰,除非剛好弱勢=灰)---
    assert ui["signal_card"].bgcolor in ("#D32F2F", "#2E7D32", "#9E9E9E"), "適配大卡底色未設為判定色"
    assert ui["signal_name"].value in ("符合策略", "會被擋成現金", "強度不足", "不符合"), "判定文字異常"
    # --- 斷言:KPI 已填入 ---
    assert "/" in ui["rank_val"].value, "動能排名未更新(應為 X/N)"
    assert ui["abs_val"].value.endswith("%"), "絕對動能未更新"
    assert ui["vol_val"].value.endswith("%"), "波動未更新"
    # --- 斷言:當前股價已填、5 日曲線與 6 個月走勢圖已建立 ---
    assert float(ui["price_val"].value) > 0, "當前股價應為正數"
    assert "%)" in ui["price_chg"].value, "當日漲跌未更新"
    assert isinstance(ui["price_holder"].content, ft.Image), "5 日股價曲線未建立"
    assert ui["price_holder"].content.src, "5 日曲線 base64 為空"
    assert isinstance(ui["chart_holder"].content, ft.Image), "6 個月走勢圖未建立"
    assert ui["chart_holder"].content.src, "6 個月走勢圖 base64 為空"

    print("\n個股分析(策略適配)UI 更新邏輯 PASSED。")


if __name__ == "__main__":
    main()
