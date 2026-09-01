# -*- coding: utf-8 -*-
"""
main.py - Flet 手機直式 UI + Async 編排(專案進入點)
版面:
  頂部:股票代號 TextField + 「開始運算」按鈕
  燈號:大字彩色卡(依最新訊號顯示中文名 + 台股慣例顏色)
  中間:matplotlib 權益曲線圖
  底部:總報酬率 / 最大回撤 / 勝率 三張 KPI 卡
非同步:on_click 為 async,重運算用 asyncio.to_thread 丟背景執行緒,
        期間顯示 ProgressBar,完成後一次 page.update()。
"""
import asyncio
import base64
import hashlib
import hmac
import io
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                       # 非互動後端;搭配下方 FigureCanvasAgg
from matplotlib.figure import Figure        # 用 Figure 物件而非全域 pyplot(thread-safe)
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 讓圖表標題/標註可顯示繁體中文(股票名稱);優先用 Windows 內建黑體,
# 找不到時自動退回預設字型(僅中文會變方框,英文標題不受影響)。
for _cjk in ("Microsoft JhengHei", "Microsoft YaHei", "DFKai-SB", "SimSun"):
    try:
        matplotlib.rcParams["font.sans-serif"] = [_cjk] + matplotlib.rcParams.get("font.sans-serif", [])
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False   # 負號正常顯示(避免用 Unicode minus)

import flet as ft

from core.conservative_ai import format_conservative_report, run_conservative_wfa
import config
from core.data_pipeline import ensure_data, ensure_db, get_stock_name, load_ohlcv, update_symbols
from core.exit_radar import RadarSettings, analyze_exit_radar
from core import rotation
from core import journal

ensure_db()

# --- 新舊版相容:Colors / Icons / Border(新版大寫、舊版小寫)---
C = getattr(ft, "Colors", None) or getattr(ft, "colors", None)
I = getattr(ft, "Icons", None) or getattr(ft, "icons", None)
B = getattr(ft, "Border", None) or getattr(ft, "border", None)

AUTH_STORAGE_KEY = "quant_app.remembered_login.v1"


def _auth_token(password: str) -> str:
    """產生瀏覽器記住登入用的憑證；不保存明文密碼。"""
    return hmac.new(
        password.encode("utf-8"),
        b"quant-app-remembered-login-v1",
        hashlib.sha256,
    ).hexdigest()


# 註:flet-charts 0.85 的 MatplotlibChart 是「互動式」後端(WebAgg 串流),
# 需要 figure.canvas.manager,裸 Figure 會在 resize 時 add_web_socket 報錯。
# 權益曲線是靜態圖,故改用 Agg 轉 PNG base64 -> ft.Image,桌面/web 皆穩定且 thread-safe。


# ---------------------------------------------------------------------------
# 繪製近 6 個月走勢 / 近 5 日收盤(回傳 matplotlib Figure 物件)
# ---------------------------------------------------------------------------
def make_price_figure(price5, target=None) -> Figure:
    """
    畫近 5 日收盤折線(上漲紅、下跌綠,符合台股慣例)。
    target 有給時加一條虛線水平線標註「預期目標價」。
    """
    vals = price5.values
    up = vals[-1] >= vals[0]
    color = "#D32F2F" if up else "#2E7D32"
    fig = Figure(figsize=(5, 1.7), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(range(len(vals)), vals, color=color, linewidth=1.8,
            marker="o", markersize=4)
    # 預期目標價:水平虛線 + 右側文字標註
    if target is not None and np.isfinite(target):
        ax.axhline(target, color="#1565C0", linewidth=1.1, linestyle="--",
                   label=f"Target {target:.2f}")
        ax.annotate(f"目標 {target:.2f}", (len(vals) - 1, target), fontsize=7,
                    color="#1565C0", xytext=(2, 2), textcoords="offset points",
                    ha="right", va="bottom")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([d.strftime("%m/%d") for d in price5.index], fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_title("Last 5 Closes", fontsize=9)
    fig.tight_layout()
    return fig


def _fig_to_image(fig: Figure) -> ft.Image:
    """Figure -> Agg 渲染 PNG -> base64 -> ft.Image(thread-safe,無互動 websocket)。"""
    canvas = FigureCanvasAgg(fig)            # 明確綁定 Agg canvas
    buf = io.BytesIO()
    canvas.print_png(buf)                    # 渲染成 PNG bytes
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    # ft.Image 的 src 可吃 base64 字串
    return ft.Image(src=b64, fit=ft.BoxFit.CONTAIN, expand=True)


def price_image(price5, target=None) -> ft.Image:
    """近 5 日收盤曲線(可疊預期目標價)-> PNG base64 -> ft.Image。"""
    return _fig_to_image(make_price_figure(price5, target))


def make_stock_price_figure(price6, title=None) -> Figure:
    """近 6 個月收盤走勢(上漲紅、下跌綠);取代沒 edge 的權益曲線。"""
    vals = price6.values
    up = len(vals) > 1 and vals[-1] >= vals[0]
    color = "#D32F2F" if up else "#2E7D32"
    fig = Figure(figsize=(5, 2.3), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(price6.index, vals, color=color, linewidth=1.6)
    ax.set_title((f"{title} · " if title else "") + "近 6 個月走勢", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def stock_price_image(price6, title=None) -> ft.Image:
    """近 6 個月走勢 -> PNG base64 -> ft.Image。"""
    return _fig_to_image(make_stock_price_figure(price6, title))


def make_exit_radar_figure(result: dict, title: str = "") -> Figure:
    """近一年價格、20/60 日線、ATR/Chandelier 與結構防守價。"""
    df = result["frame"].tail(250)
    fig = Figure(figsize=(5, 2.8), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(df.index, df["close"], color="#263238", linewidth=1.4, label="收盤")
    ax.plot(df.index, df["sma20"], color="#1976D2", linewidth=1.0, label="20日線")
    ax.plot(df.index, df["sma60"], color="#7B1FA2", linewidth=1.0, label="60日線")
    atr_line = df["high"].rolling(60, min_periods=1).max() - result["atr_multiplier"] * df["atr14"]
    ax.plot(df.index, atr_line, color="#EF6C00", linewidth=.9, linestyle="--", label="ATR停利")
    ax.plot(df.index, df["chandelier"], color="#00897B", linewidth=.8, linestyle=":", label="Chandelier")
    if result.get("pivot_low"):
        ax.axhline(result["pivot_low"], color="#B71C1C", linewidth=.9, linestyle="--", label="波段低點")
    ax.set_title((title + " · " if title else "") + "出場雷達", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=.18)
    ax.legend(fontsize=6.5, ncol=3, loc="best")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    return fig


def _mini_radar_kpi(title: str, value: str) -> ft.Container:
    return ft.Container(content=ft.Column([
        ft.Text(title, size=10, color=getattr(C, "GREY_700", "#616161")),
        ft.Text(value, size=14, weight=ft.FontWeight.BOLD),
    ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=getattr(C, "GREY_100", "#F5F5F5"), padding=9,
        border_radius=10, expand=True, alignment=ft.Alignment.CENTER)


def make_exit_radar_controls(result: dict, title: str = "") -> list:
    """把純分析結果轉為手機友善、可解釋的 Flet 卡片。"""
    if not result.get("available"):
        return [ft.Container(content=ft.Column([
            ft.Text("出場雷達：資料不足", size=18, weight=ft.FontWeight.BOLD),
            ft.Text(result.get("message", "無法分析"), size=12),
        ], spacing=4), padding=14, border_radius=14, bgcolor="#FFF8E1")]

    def money(v):
        return "--" if v is None or not np.isfinite(v) else f"{v:,.2f}"

    status = ft.Container(content=ft.Column([
        ft.Text(result["status"], size=23, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
        ft.Text(result["advice"], size=12, color="#FFFFFF"),
        ft.Text(f"{title} · 資料 {result['asof']} · {result['price_basis']}", size=10, color="#FFFFFF"),
    ], spacing=4), bgcolor=result["color"], padding=16, border_radius=14)
    levels = ft.Row([
        _mini_radar_kpi("現價", money(result["last_close"])),
        _mini_radar_kpi("第一防守", money(result["first_defense"])),
        _mini_radar_kpi("關鍵防守", money(result["key_defense"])),
    ], spacing=7)
    risk = ft.Row([
        _mini_radar_kpi(f"ATR {result['atr_multiplier']:.0f}×", money(result["atr_stop"])),
        _mini_radar_kpi("Chandelier", money(result["chandelier"])),
        _mini_radar_kpi("高點回撤", f"{result['drawdown']*100:.1f}%"),
    ], spacing=7)
    signal_rows = []
    for signal in result["signals"]:
        icon = {"leading": "⚠", "confirm": "◆", "reinforce": "●"}[signal["kind"]]
        signal_rows.append(ft.Text(f"{icon} {signal['title']}｜{signal['detail']}", size=11))
    if not signal_rows:
        signal_rows = [ft.Text("✓ 目前沒有觸發技術面轉弱條件", size=11, color="#2E7D32")]
    pending = [ft.Text("— " + item, size=11, color=getattr(C, "GREY_700", "#616161"))
               for item in result.get("pending", [])]
    basis = ("未提供此持倉買入日，ATR 停利暫以近 60 個交易日高點估算。"
             if result.get("holding_fallback") else "ATR 停利依此持倉買入日至今最高價計算。")
    details = ft.Container(content=ft.Column([
        ft.Text("已觸發警訊", size=13, weight=ft.FontWeight.BOLD), *signal_rows,
        ft.Divider(), ft.Text("尚未確認", size=13, weight=ft.FontWeight.BOLD),
        *(pending or [ft.Text("目前主要確認條件均已觸發。", size=11)]), ft.Divider(),
        ft.Text(f"RSI {money(result['rsi14'])}　量比 {money(result['relative_volume'])}×　ATR {money(result['atr14'])}", size=11),
        ft.Text(basis, size=10, color=getattr(C, "GREY_700", "#616161")),
        ft.Text("技術訊號可能出現假突破或延遲，請配合持倉週期與可承受風險使用。",
                size=10, color=getattr(C, "GREY_700", "#616161")),
    ], spacing=6), bgcolor="#FFFFFF", padding=12, border_radius=12,
        border=B.all(1, getattr(C, "GREY_300", "#E0E0E0")) if B else None)
    chart = ft.Container(content=_fig_to_image(make_exit_radar_figure(result, title)), height=285,
                         bgcolor="#FFFFFF", border_radius=12)
    return [status, levels, risk, chart, details]


# ---------------------------------------------------------------------------
# 套用回測結果到 UI 控件(抽出為 module-level,方便無 GUI 單元測試)
def apply_fit(ui: dict, res: dict) -> None:
    """
    把 rotation.analyze_stock 的結果套到「個股分析(策略適配檢查)」控件:
      - 大卡:符合/不符合/轉現金 + 動能排名(取代沒 edge 的 ML 燈號)
      - KPI:動能排名 / 絕對動能 / 近 60 日波動
      - 圖:近 6 個月走勢(取代沒 edge 的權益曲線)
    純控件賦值,不呼叫 page.update();便於測試。
    """
    name = res.get("name") or ""
    label = (f"{name} {res['symbol']}".strip()
             if name and name != res["symbol"] else res["symbol"])
    ui["signal_name"].value = res["verdict_short"]
    ui["signal_sub"].value = (
        f"{label} · 動能 {res['mom']*100:+.0f}% · 排名 {res['rank']}/{res['n']}"
        f" · 資料 {res['asof']}")
    ui["signal_card"].bgcolor = res["verdict_color"]
    if "note_hint" in ui:
        ui["note_hint"].value = res["note"]
        ui["note_hint"].color = res["verdict_color"]

    # 當前股價卡(收盤 + 當日漲跌)
    dc = res["day_change"]
    up = dc >= 0
    pc = "#D32F2F" if up else "#2E7D32"
    ui["price_val"].value = f"{res['last_close']:.2f}"
    ui["price_val"].color = pc
    ui["price_chg"].value = f"{'▲' if up else '▼'} {dc:+.2f} ({res['day_change_pct']*100:+.2f}%)"
    ui["price_chg"].color = pc
    ui["price_date"].value = f"資料日 {res['asof']}"
    ui["price_holder"].content = price_image(res["price5"])
    ui["price_holder"].bgcolor = "#FFFFFF"

    # 近 6 個月走勢(取代權益曲線)
    ui["chart_holder"].content = stock_price_image(res["price6"], label)
    ui["chart_holder"].bgcolor = "#FFFFFF"

    # KPI:動能排名 / 絕對動能 / 波動
    ui["rank_val"].value = f"{res['rank']}/{res['n']}"
    ui["rank_val"].color = "#D32F2F" if res["in_top_k"] else getattr(C, "GREY_700", "#616161")
    m = res["mom"]
    ui["abs_val"].value = f"{m*100:+.0f}%"
    ui["abs_val"].color = "#D32F2F" if m >= 0 else "#2E7D32"
    ui["vol_val"].value = f"{res['vol_annual']*100:.0f}%"
    if "mom_line" in ui:
        ui["mom_line"].value = (
            f"近5日 {res['mom5']*100:+.1f}% · 近20日 {res['mom20']*100:+.1f}% · "
            f"近60日 {m*100:+.1f}%　|　近1年最大回撤 {res['mdd']*100:.0f}%")


# ---------------------------------------------------------------------------
# 今日買賣建議:跑遍 WATCHLIST,依「今日訊號」給每檔買/賣/觀望建議,取 5 支
def monthly_holdings(defensive: bool = False) -> dict:
    """跑相對強弱輪動,回傳本期應持有清單與回測統計(供 UI 渲染)。
    defensive=True 啟用融資濾網防禦模式(空頭少賠、平時報酬低)。"""
    return rotation.run_rotation(defensive=defensive)


def make_holdings_rows(res: dict, on_add=None, held_trades=None,
                       on_renew=None) -> list:
    """
    把輪動結果渲染成 Flet 卡片:頂部統計列 + 每檔持有卡 + 賣出提示。
    on_add 有給時,每檔持有卡附「金額/買入價/停損/停利」輸入 + 一鍵「加入庫存」按鈕;
    回呼簽章:on_add(symbol, name, amount_field, price_field, stop_field, take_field)。
    held_trades:{代號: 持有中紀錄}(來源=輪動)。換股日又選到同一檔(續抱)時,
    該卡改顯示持有資訊 + 「續抱·更新輪替日」按鈕(on_renew(symbol)),
    而不是再開一次買入欄位 —— 對應「20 天後還選到同一支 -> 更新輪替日期」。
    """
    held_trades = held_trades or {}
    cards = []
    # 0) 費半市場燈(RISK ON/OFF;策略切換門檻)
    sox = res.get("sox") or {}
    market_off = bool(sox.get("ok") and not sox.get("risk_on", True))
    if sox.get("ok"):
        on = sox.get("risk_on", True)
        cards.append(ft.Container(
            content=ft.Column([
                ft.Text("🟢 市場 RISK ON" if on else "🔴 市場 RISK OFF",
                        size=18, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
                ft.Text(f"費半 {sox['close']:.0f} {'>' if on else '<'} "
                        f"{sox['ma_len']}日均線 {sox['ma']:.0f}({sox['pct']:+.1f}%)"
                        f" · 資料 {sox['asof']}",
                        size=11, color="#FFFFFF"),
                ft.Text("正常持有(費半在均線上)" if on
                        else "⚠ 建議整批轉現金,等費半站回均線再進場",
                        size=13, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
            ], spacing=2),
            bgcolor="#2E7D32" if on else "#B71C1C",
            padding=14, border_radius=14))
    # 1) 策略統計列(年化 vs 大盤、回撤、規則)
    beat = res["cagr"] >= res["market_cagr"]
    stat = ft.Column([
        ft.Text(
            f"策略年化 {res['cagr']*100:.1f}%　vs　大盤 {res['market_cagr']*100:.1f}%"
            f"　{'✓ 勝出' if beat else '✗ 落後'}",
            size=13, weight=ft.FontWeight.BOLD,
            color="#D32F2F" if beat else "#2E7D32"),
        ft.Text(
            f"最大回撤 {res['mdd']*100:.1f}%(大盤 {res['market_mdd']*100:.1f}%)"
            f"　·　約 {res['years']:.0f} 年回測",
            size=11, color=getattr(C, "GREY_700", "#616161")),
        ft.Text(
            f"模式:{'🛡 防禦(融資濾網·空頭少賠)' if res.get('defensive') else '⚡ 標準(衝報酬)'}"
            f" · {res['mom_days']} 日動能選前 {res['top_k']} 強 · 每 {res['rebal_days']} 交易日換股 · 資料到 {res['last_date']}",
            size=10, color=getattr(C, "GREY", "#9E9E9E")),
    ], spacing=2)
    # 絕對動能閘門狀態(本期有幾檔轉現金)
    cash_syms = res.get("cash_symbols", [])
    if res.get("abs_mom"):
        if cash_syms:
            stat.controls.append(ft.Text(
                f"⚠ 絕對動能閘門:前 {res['top_k']} 強有 {len(cash_syms)} 檔動能翻負 → 該檔轉現金、不進場",
                size=11, weight=ft.FontWeight.BOLD, color="#2E7D32"))
        else:
            stat.controls.append(ft.Text(
                "絕對動能閘門:本期前 K 強動能皆為正,滿倉進場",
                size=10, color=getattr(C, "GREY", "#9E9E9E")))
    cards.append(ft.Container(content=stat, bgcolor="#FFF8E1",
                              padding=12, border_radius=12))

    # 2) 每檔持有卡(等權;新進=買進紅、續抱=藍)+ 一鍵加入庫存
    slot = getattr(config, "ROTATION_SLOT_AMOUNT", 30000)
    name_map = res.get("names", {})
    rank_map = {s: (nm, mv) for s, nm, mv in res["ranking"]}
    cash_set = set(res.get("cash_symbols", []))
    for i, sym in enumerate(res["holdings"], 1):
        nm, mv = rank_map.get(sym, (name_map.get(sym, ""), 0.0))
        title = f"{nm} {sym}".strip()
        is_cash = sym in cash_set                      # 絕對動能翻負 -> 轉現金
        is_new = sym in res.get("buys", [])
        if is_cash:
            tag, tag_color = "轉現金", "#2E7D32"
        elif is_new:
            tag, tag_color = "買進", "#D32F2F"
        else:
            tag, tag_color = "續抱", "#1565C0"
        badge = ft.Container(
            content=ft.Text(tag, size=12, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
            bgcolor=tag_color, padding=ft.Padding(10, 3, 10, 3), border_radius=8)
        head = ft.Row(
            [ft.Text(f"{i}. {title}", size=16, weight=ft.FontWeight.BOLD),
             badge,
             ft.Text(f"動能 {mv*100:+.0f}%", size=13, weight=ft.FontWeight.BOLD,
                     color="#D32F2F" if mv >= 0 else "#2E7D32")],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER)
        col = [head]
        held_t = held_trades.get(sym)
        if is_cash:
            # 不提供加入庫存,顯示現金原因(RISK OFF / 外資急賣 / 絕對動能翻負)
            if market_off:
                reason = "市場 RISK OFF(費半跌破均線)→ 本期轉現金,站回再進"
            elif sym in (res.get("fastsell_symbols") or []):
                reason = "⚠ 外資急賣(5日持股驟降)→ 大戶在跑,該檔轉現金"
            else:
                reason = "絕對動能翻負(相對最強但仍在跌)→ 建議現金 / 不進場"
            col.append(ft.Text(reason, size=12, weight=ft.FontWeight.BOLD,
                               color="#2E7D32"))
        elif held_t is not None:
            # 已持有(上次輪替買入、本期又選到 = 續抱):更新輪替日,不重複開買入欄
            lr = held_t.get("last_rotation") or held_t.get("buy_time", "")[:10]
            col.append(ft.Text(
                f"✓ 已持有 {held_t['shares']:.2f} 股 @ {held_t['buy_price']:.2f}"
                f"　·　最近輪替 {lr or '—'}",
                size=12, color=getattr(C, "GREY_700", "#616161")))
            if on_renew is not None:
                try:
                    renew_btn = ft.Button(content="續抱·更新輪替日",
                                          icon=getattr(I, "EVENT_REPEAT", None))
                except Exception:
                    renew_btn = ft.ElevatedButton(text="續抱·更新輪替日",
                                                  icon=getattr(I, "EVENT_REPEAT", None))
                renew_btn.on_click = (lambda e, s=sym: on_renew(s))
                col.append(ft.Row([renew_btn]))
        elif on_add is None:
            col.append(ft.Text(f"等權持有　建議投入 約 NT$ {slot:,}", size=12,
                               color=getattr(C, "GREY_700", "#616161")))
        else:
            # 預設買入價=最新收盤;停損/停利用 ATR 波動式(波動大→自動放寬),皆可改
            last = journal.get_last_close(sym)
            p0 = float(last) if last else 0.0
            pv = f"{p0:.2f}" if p0 else ""
            sl_lvl, tk_lvl = rotation.stop_take_levels(sym, p0) if p0 else (0.0, None)
            amt_f = ft.TextField(label="金額", value=str(slot), width=86,
                                 dense=True, text_size=13)
            price_f = ft.TextField(label="買入價", value=pv, width=86,
                                   dense=True, text_size=13)
            stop_f = ft.TextField(label="停損", width=82, dense=True, text_size=13,
                                  value=(f"{sl_lvl:.2f}" if p0 else ""))
            take_f = ft.TextField(label="停利", width=82, dense=True, text_size=13,
                                  value=(f"{tk_lvl:.2f}" if (p0 and tk_lvl) else ""))
            try:
                add_btn = ft.Button(content="加入庫存",
                                    icon=getattr(I, "ADD_SHOPPING_CART", None))
            except Exception:
                add_btn = ft.ElevatedButton(text="加入庫存",
                                            icon=getattr(I, "ADD_SHOPPING_CART", None))
            add_btn.on_click = (
                lambda e, s=sym, n=nm, a=amt_f, p=price_f, sl=stop_f, tp=take_f:
                on_add(s, n, a, p, sl, tp))
            col.append(ft.Row([amt_f, price_f], spacing=6))
            col.append(ft.Row([stop_f, take_f, add_btn], spacing=6,
                              vertical_alignment=ft.CrossAxisAlignment.CENTER))
        cards.append(ft.Container(
            content=ft.Column(col, spacing=6),
            bgcolor=getattr(C, "GREY_100", "#F5F5F5"), padding=12, border_radius=12))

    # 3) 賣出提示(上期持有、本期掉出前 K 的標的)
    sells = res.get("sells", [])
    if sells:
        names = "、".join(f"{name_map.get(s,'')} {s}".strip() for s in sells)
        cards.append(ft.Container(
            content=ft.Text(f"賣出(已轉弱、移出前 {res['top_k']} 強):{names}",
                            size=12, weight=ft.FontWeight.BOLD, color="#2E7D32"),
            bgcolor="#E8F5E9", padding=12, border_radius=12))
    return cards


def make_allocation_rows(alloc: dict) -> list:
    """把零股配置結果渲染成卡片:頂部彙總 + 每檔『買幾股 × 股價 = 成本(權重)』。"""
    rows = []
    rows.append(ft.Container(
        content=ft.Text(
            f"總預算 {alloc['budget']:,.0f} → 已配置 {alloc['spent']:,.0f}"
            f"(剩餘現金 {alloc['leftover']:,.0f})· 等權分散 {alloc['n']} 檔",
            size=12, weight=ft.FontWeight.BOLD),
        bgcolor="#E3F2FD", padding=10, border_radius=10))
    for x in alloc.get("rows", []):
        title = f"{x['name']} {x['symbol']}".strip()
        if x["affordable"]:
            detail = (f"{x['shares']} 股 × {x['price']:.1f} = "
                      f"{x['cost']:,.0f}　({x['weight']*100:.0f}%)")
            color = getattr(C, "GREY_700", "#616161")
        else:
            detail = "預算不足,買不到 1 股(加預算或減檔數)"
            color = "#B71C1C"
        rows.append(ft.Container(
            content=ft.Row([ft.Text(title, size=14, weight=ft.FontWeight.BOLD),
                            ft.Text(detail, size=12, color=color)],
                           alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=getattr(C, "GREY_100", "#F5F5F5"), padding=10, border_radius=10))
    return rows


# ---------------------------------------------------------------------------
# 投資紀錄(交易日誌):從 0 記錄投入金額、買入/賣出點位與時間、報酬
# ---------------------------------------------------------------------------
def make_summary_text(s: dict) -> str:
    """顯示總資產、現金、成本與已／未實現績效。"""
    current_return = s.get("current_return")
    if current_return is None and s.get("unrealized_pnl") is not None:
        current_return = (s.get("unrealized_pnl", 0) / s["invested"]
                          if s.get("invested") else 0.0)
    total_assets = (f"{s['total_assets']:,.0f}" if s.get("total_assets") is not None
                    else "待行情")
    market_value = (f"{s['market_value']:,.0f}" if s.get("market_value") is not None
                    else "待行情")
    current_gain = (f"{s['unrealized_pnl']:+,.0f}（{current_return*100:+.1f}%）"
                    if s.get("unrealized_pnl") is not None else "待行情")
    total_pnl = (f"{s['total_pnl']:+,.0f}" if s.get("total_pnl") is not None else "待行情")
    total_return = (f"{s['total_return']*100:+.1f}%" if s.get("total_return") is not None
                    else "待行情")
    return (
        f"總資產 {total_assets}　"
        f"持倉市值 {market_value}　"
        f"現金 {s.get('cash', 0):,.0f}\n"
        f"持倉成本 {s['invested']:,.0f}　"
        f"目前收益 {current_gain}　"
        f"已實現 {s['realized_pnl']:+,.0f}\n"
        f"證券總損益 {total_pnl}　"
        f"累計成本報酬率 {total_return}　·　"
        f"持倉 lot {s['n_open']} · 已平倉 {s['n_closed']}"
    )


def make_position_rows(positions: list) -> list:
    """把彙總持倉與逐筆交易 lot 分開顯示。"""
    cards = []
    for position in positions:
        title = f"{position.get('name') or ''} {position['symbol']}".strip()
        pnl = position.get("pnl")
        ret = position.get("return")
        color = "#D32F2F" if (pnl or 0) >= 0 else "#2E7D32"
        market = ("現值待行情" if position.get("market_value") is None
                  else f"現值 {position['market_value']:,.0f}")
        performance = ("損益待行情" if pnl is None
                       else f"損益 {pnl:+,.0f}（{ret*100:+.1f}%）")
        cards.append(ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(title, size=15, weight=ft.FontWeight.BOLD),
                    ft.Text(performance, size=13, weight=ft.FontWeight.BOLD,
                            color=color),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Text(
                    f"{position['shares']:g} 單位 · 平均成本 "
                    f"{position['average_cost']:,.2f} · 成本 {position['cost']:,.0f}",
                    size=12),
                ft.Text(f"{market} · {position['lots']} 個成本 lot",
                        size=11, color=getattr(C, "GREY_700", "#616161")),
            ], spacing=4),
            bgcolor=getattr(C, "GREY_100", "#F5F5F5"),
            padding=12, border_radius=12))
    return cards


def _edit_card(t: dict, on_save, on_cancel) -> "ft.Container":
    """
    單筆紀錄的「編輯模式」卡:可手動改 金額/買入價/買入時間/停損停利/備註,
    已平倉再加 賣出價/賣出時間。儲存 -> on_save(id, {欄位: TextField});取消 -> on_cancel()。
    (股數不直接改:改金額或買入價後由 journal.update_trade 自動重算 = 金額/買入價)
    """
    tid = t["id"]
    is_open = t["status"] == "open"
    title = f"{t.get('name') or ''} {t['symbol']}".strip()

    def F(label, val, w=95):
        return ft.TextField(label=label, value=("" if val is None else str(val)),
                            width=w, dense=True, text_size=13)

    f = {
        "amount": F("金額", f"{t['amount']:.0f}"),
        "buy_price": F("買入價", f"{t['buy_price']:.2f}"),
        "buy_time": F("買入時間", t.get("buy_time") or "", 130),
        "stop_loss": F("停損(空=無)",
                       "" if t.get("stop_loss") is None else f"{t['stop_loss']:.2f}", 105),
        "take_profit": F("停利(空=無)",
                         "" if t.get("take_profit") is None else f"{t['take_profit']:.2f}", 105),
        "note": F("備註", t.get("note") or "", 230),
    }
    rows = [ft.Row([f["amount"], f["buy_price"], f["buy_time"]], spacing=6),
            ft.Row([f["stop_loss"], f["take_profit"]], spacing=6)]
    if not is_open:
        f["sell_price"] = F("賣出價",
                            "" if t.get("sell_price") is None else f"{t['sell_price']:.2f}")
        f["sell_time"] = F("賣出時間", t.get("sell_time") or "", 130)
        rows.append(ft.Row([f["sell_price"], f["sell_time"]], spacing=6))
    rows.append(ft.Row([f["note"]], spacing=6))
    try:
        save_btn = ft.Button(content="儲存", icon=getattr(I, "SAVE", None))
        cancel_btn = ft.Button(content="取消")
    except Exception:
        save_btn = ft.ElevatedButton(text="儲存", icon=getattr(I, "SAVE", None))
        cancel_btn = ft.ElevatedButton(text="取消")
    save_btn.on_click = (lambda e, _id=tid, _f=f: on_save(_id, _f))
    cancel_btn.on_click = (lambda e: on_cancel())
    return ft.Container(
        content=ft.Column(
            [ft.Text(f"✏ 編輯:{title or f'#{tid}'}", size=14,
                     weight=ft.FontWeight.BOLD)] + rows
            + [ft.Row([save_btn, cancel_btn], spacing=8)],
            spacing=6),
        bgcolor="#FFF3E0", padding=12, border_radius=12)


def make_journal_rows(trades: list, on_sell, on_delete, on_edit=None,
                      editing_id=None, on_save=None, on_cancel=None,
                      on_reopen=None) -> list:
    """
    把交易紀錄渲染成 Flet 卡片(module-level,callback 由 main 注入)。
      on_sell(trade_id, price_field, qty_field):股數留空=整筆平倉,填了=部分賣出。
      on_delete(trade_id):刪除;on_edit(trade_id):進入編輯模式;
      on_reopen(trade_id):把誤平倉復原成持有中。
      editing_id:目前編輯中的紀錄 id(該卡渲染成可改欄位的編輯卡)。
    持有中:成本 + 浮動損益 + 停損停利警示 + 賣出價/股數 + 賣出/編輯/刪除。
    已平倉:賣出價/時間/持有天數 + 已實現損益 + 復原/編輯/刪除。
    """
    cards = []
    for t in trades:
        tid = t["id"]
        is_open = t["status"] == "open"
        title = f"{t.get('name') or ''} {t['symbol']}".strip()
        if editing_id == tid and on_save is not None:
            cards.append(_edit_card(t, on_save, on_cancel))
            continue
        tag, tag_color = ("持有中", "#1565C0") if is_open else ("已平倉", "#616161")
        if t.get("source") == "dca":
            tag += "·定額"
        elif t.get("source") == "rotation":
            tag += "·推薦"
        badge = ft.Container(
            content=ft.Text(tag, size=11, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
            bgcolor=tag_color, padding=ft.Padding(8, 2, 8, 2), border_radius=8)

        # 損益(台股紅正綠負;持有中=浮動、已平倉=已實現)
        pnl = t.get("pnl")
        ret = t.get("ret")
        if pnl is None:
            pnl_str = "現價未知(僅顯示成本)"
            pnl_color = getattr(C, "GREY_700", "#616161")
        else:
            pnl_color = "#D32F2F" if pnl >= 0 else "#2E7D32"
            kind = "浮動損益" if is_open else "已實現"
            pnl_str = f"{kind} {pnl:+,.0f}　({ret*100:+.1f}%)"

        head = ft.Row(
            [ft.Text(title or f"#{tid}", size=15, weight=ft.FontWeight.BOLD),
             badge,
             ft.Text(pnl_str, size=12, weight=ft.FontWeight.BOLD, color=pnl_color)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER)

        cost_line = (f"投入 {t['amount']:,.0f}　{t['shares']:.2f} 股 @ "
                     f"{t['buy_price']:.2f}　買入 {t['buy_time']}")
        lr = t.get("last_rotation") or ""
        if lr and lr != (t.get("buy_time") or "")[:10]:
            cost_line += f"　·　最近輪替 {lr}"

        # 刪除鈕(共用)
        del_btn = ft.IconButton(
            icon=getattr(I, "DELETE_OUTLINE", None), icon_size=18,
            icon_color="#B71C1C", tooltip="刪除這筆紀錄",
            on_click=(lambda e, _id=tid: on_delete(_id)))

        col = [head, ft.Text(cost_line, size=11,
                             color=getattr(C, "GREY_700", "#616161"))]
        if t.get("note"):
            col.append(ft.Text(f"📝 {t['note']}", size=11,
                               color=getattr(C, "GREY_700", "#616161")))

        # 停損/停利(由本月推薦加入時記錄);持有中且現價觸價時跳警示
        sl, tp = t.get("stop_loss"), t.get("take_profit")
        if sl or tp:
            cur = t.get("cur_price")
            seg = []
            if sl:
                seg.append(f"停損 {sl:.2f}")
            if tp:
                seg.append(f"停利 {tp:.2f}")
            line, line_color = "　".join(seg), getattr(C, "GREY_700", "#616161")
            if is_open and cur is not None:
                if sl and cur <= sl:
                    line += "　⚠ 已觸停損,建議賣出"
                    line_color = "#2E7D32"
                elif tp and cur >= tp:
                    line += "　✔ 已達停利,可獲利了結"
                    line_color = "#D32F2F"
            col.append(ft.Text(line, size=11, weight=ft.FontWeight.BOLD,
                               color=line_color))

        if is_open:
            cur = t.get("cur_price")
            cur_line = (f"現價 {cur:.2f}　現值 {t['value']:,.0f}　持有 {t['hold_days']} 天"
                        if cur is not None else f"現價未知　持有 {t['hold_days']} 天")
            col.append(ft.Text(cur_line, size=11,
                               color=getattr(C, "GREY_700", "#616161")))
            sell_field = ft.TextField(
                label="賣出價", width=88, dense=True, text_size=13,
                value=(f"{cur:.2f}" if cur is not None else ""))
            qty_field = ft.TextField(
                label="股數(空=全部)", width=110, dense=True, text_size=13)
            try:
                sell_btn = ft.Button(content="賣出")
            except Exception:
                sell_btn = ft.ElevatedButton(text="賣出")
            sell_btn.on_click = (lambda e, _id=tid, _f=sell_field, _q=qty_field:
                                 on_sell(_id, _f, _q))
            edit_btn = ft.IconButton(
                icon=getattr(I, "EDIT_OUTLINED", None), icon_size=18,
                tooltip="編輯這筆(金額/價格/時間/停損停利/備註)",
                on_click=(lambda e, _id=tid: on_edit(_id))) if on_edit else None
            row = [sell_field, qty_field, sell_btn]
            if edit_btn:
                row.append(edit_btn)
            row.append(del_btn)
            col.append(ft.Row(row, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                              spacing=6))
        else:
            sold_line = (f"賣出 {t['sell_price']:.2f}　{t.get('sell_time') or ''}"
                         f"　持有 {t['hold_days']} 天")
            btns = []
            if on_reopen:
                btns.append(ft.IconButton(
                    icon=getattr(I, "UNDO", None), icon_size=18,
                    tooltip="復原成持有中(誤平倉用)",
                    on_click=(lambda e, _id=tid: on_reopen(_id))))
            if on_edit:
                btns.append(ft.IconButton(
                    icon=getattr(I, "EDIT_OUTLINED", None), icon_size=18,
                    tooltip="編輯這筆",
                    on_click=(lambda e, _id=tid: on_edit(_id))))
            btns.append(del_btn)
            col.append(ft.Row(
                [ft.Text(sold_line, size=11,
                         color=getattr(C, "GREY_700", "#616161")),
                 ft.Row(btns, spacing=0)],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                vertical_alignment=ft.CrossAxisAlignment.CENTER))

        cards.append(ft.Container(
            content=ft.Column(col, spacing=4),
            bgcolor=getattr(C, "GREY_100", "#F5F5F5"), padding=12, border_radius=12))
    return cards


def make_dca_rows(plans: list, on_toggle, on_delete) -> list:
    """
    把定期定額計畫渲染成卡片(module-level,callback 由 main 注入)。
      on_toggle(plan_id, active):啟用/停用;on_delete(plan_id):刪除。
    顯示:代號名稱、每期金額、頻率、下次扣款日、啟用狀態 + 切換/刪除鈕。
    """
    cards = []
    for p in plans:
        pid = p["id"]
        title = f"{p.get('name') or ''} {p['symbol']}".strip()
        active = p["active"]
        tag, tag_color = ("啟用中", "#2E7D32") if active else ("已停用", "#9E9E9E")
        badge = ft.Container(
            content=ft.Text(tag, size=11, weight=ft.FontWeight.BOLD, color="#FFFFFF"),
            bgcolor=tag_color, padding=ft.Padding(8, 2, 8, 2), border_radius=8)
        head = ft.Row(
            [ft.Text(title or f"#{pid}", size=15, weight=ft.FontWeight.BOLD), badge,
             ft.Text(f"{p['freq_label']} {p['amount']:,.0f}", size=13,
                     weight=ft.FontWeight.BOLD, color="#1565C0")],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER)
        toggle_btn = ft.IconButton(
            icon=getattr(I, "PAUSE_CIRCLE_OUTLINE" if active else "PLAY_CIRCLE_OUTLINE", None),
            icon_size=20, tooltip=("停用" if active else "啟用"),
            on_click=(lambda e, _id=pid, _a=active: on_toggle(_id, not _a)))
        del_btn = ft.IconButton(
            icon=getattr(I, "DELETE_OUTLINE", None), icon_size=18, icon_color="#B71C1C",
            tooltip="刪除計畫", on_click=(lambda e, _id=pid: on_delete(_id)))
        info = ft.Row(
            [ft.Text(f"下次扣款 {p['next_date']}", size=11,
                     color=getattr(C, "GREY_700", "#616161")),
             ft.Row([toggle_btn, del_btn], spacing=0)],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER)
        cards.append(ft.Container(
            content=ft.Column([head, info], spacing=4),
            bgcolor="#FFF8E1", padding=12, border_radius=12))
    return cards


def make_asset_history_rows(history: list, on_delete=None) -> list:
    """把總資產快照渲染成精簡清單(時間 → 總資產 / 總損益 [+刪除鈕])。"""
    rows = []
    for h in history:
        pnl = h.get("total_pnl") or 0.0
        cells = [ft.Text(h["ts"], size=11, color=getattr(C, "GREY_700", "#616161")),
                 ft.Text(f"總資產 {h.get('total_assets', 0):,.0f}", size=12,
                         weight=ft.FontWeight.BOLD),
                 ft.Text(f"{pnl:+,.0f}", size=12,
                         color="#D32F2F" if pnl >= 0 else "#2E7D32")]
        if on_delete is not None and h.get("id") is not None:
            cells.append(ft.IconButton(
                icon=getattr(I, "DELETE_OUTLINE", None), icon_size=16,
                icon_color="#B71C1C", tooltip="刪除這筆快照",
                on_click=(lambda e, _id=h["id"]: on_delete(_id))))
        rows.append(ft.Row(cells, alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                           vertical_alignment=ft.CrossAxisAlignment.CENTER))
    return rows


def make_asset_figure(history: list) -> Figure:
    """總資產成長曲線(快照時間序;紅=賺 綠=賠,疊上累計投入虛線作對照)。"""
    hs = sorted(history, key=lambda h: h["ts"])          # 舊到新
    ts = [pd.to_datetime(h["ts"]) for h in hs]
    assets = [h.get("total_assets") or 0.0 for h in hs]
    invested = [h.get("invested") or 0.0 for h in hs]
    up = assets[-1] >= invested[-1] if hs else True
    color = "#D32F2F" if up else "#2E7D32"
    fig = Figure(figsize=(5, 2.0), dpi=110)
    ax = fig.add_subplot(111)
    ax.plot(ts, assets, color=color, linewidth=1.8, marker="o",
            markersize=3, label="總資產")
    ax.plot(ts, invested, color="#9E9E9E", linewidth=1.2, linestyle="--",
            label="累計投入")
    ax.set_title("總資產成長曲線", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="upper left")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def asset_history_image(history: list) -> ft.Image:
    """總資產快照 -> 成長曲線 PNG -> ft.Image。"""
    return _fig_to_image(make_asset_figure(history))


# ---------------------------------------------------------------------------
# Flet 主程式
# ---------------------------------------------------------------------------
def _build_app(page: ft.Page, on_logout=None):
    page.title = "量化交易燈號系統"
    if on_logout is not None:
        try:
            page.appbar = ft.AppBar(
                title=ft.Text("量化交易燈號系統"),
                actions=[ft.IconButton(icon=getattr(I, "LOGOUT", None),
                                       tooltip="登出", on_click=on_logout)])
        except Exception:
            page.appbar = None
    page.theme_mode = getattr(ft.ThemeMode, "LIGHT", None)
    # 分頁版面:外層不滾動(固定視窗高),改由每個分頁內容各自滾動
    page.scroll = None
    page.padding = 16
    # 手機直式視窗尺寸(新版 page.window;包 try/except 以相容環境差異)
    try:
        page.window.width = 420
        page.window.height = 880
    except Exception:
        pass

    # --- 頂部:代號輸入 + 按鈕 ---
    symbol_field = ft.TextField(
        label="股票代號", value="2330", width=160,
        text_size=16, dense=True,
    )
    # 新舊版相容:新版 ft.Button(content=...)、舊版 ft.ElevatedButton(text=...)
    try:
        run_btn = ft.Button(content="分析這檔", icon=getattr(I, "SEARCH", None))
    except Exception:
        run_btn = ft.ElevatedButton(text="分析這檔", icon=getattr(I, "SEARCH", None))
    # 本月持有清單按鈕(相對強弱輪動;新舊版相容)
    try:
        scan_btn = ft.Button(content="本月持有清單", icon=getattr(I, "LEADERBOARD", None))
    except Exception:
        scan_btn = ft.ElevatedButton(text="本月持有清單", icon=getattr(I, "LEADERBOARD", None))
    # 更新每日資料按鈕(增量:抓最新收盤)
    try:
        update_btn = ft.Button(content="更新每日資料", icon=getattr(I, "CLOUD_DOWNLOAD", None))
    except Exception:
        update_btn = ft.ElevatedButton(text="更新每日資料", icon=getattr(I, "CLOUD_DOWNLOAD", None))
    try:
        conservative_ai_btn = ft.Button(content="驗證保守 AI", icon=getattr(I, "SCIENCE", None))
    except Exception:
        conservative_ai_btn = ft.ElevatedButton(text="驗證保守 AI", icon=getattr(I, "SCIENCE", None))
    progress = ft.ProgressBar(visible=False)      # 個股分析進度條
    scan_progress = ft.ProgressBar(visible=False) # 本月持有進度條

    # --- 策略適配大卡(符合/不符合/轉現金;取代沒 edge 的 ML 燈號)---
    signal_name = ft.Text("尚未分析", size=30, weight=ft.FontWeight.BOLD,
                          color="#FFFFFF", text_align=ft.TextAlign.CENTER)
    signal_sub = ft.Text("輸入代號後按「分析這檔」(檢查它符不符合輪動策略)", size=13,
                         color="#FFFFFF", text_align=ft.TextAlign.CENTER)
    signal_card = ft.Container(
        content=ft.Column([signal_name, signal_sub],
                          horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                          spacing=4),
        bgcolor="#9E9E9E", padding=22, border_radius=16,
        alignment=ft.Alignment.CENTER, height=120,
    )

    # --- 適配說明橫幅(分析後顯示:為什麼符合/不符合 + 閘門邏輯)---
    trade_hint = ft.Text("分析後顯示:這檔符不符合輪動策略(動能排名 + 絕對動能閘門)",
                         size=14, weight=ft.FontWeight.BOLD,
                         color=getattr(C, "GREY_700", "#616161"),
                         text_align=ft.TextAlign.CENTER)
    trade_card = ft.Container(
        content=trade_hint, padding=14, border_radius=14,
        alignment=ft.Alignment.CENTER,
        bgcolor=getattr(C, "GREY_100", "#F5F5F5"),
        border=B.all(1, getattr(C, "GREY_300", "#E0E0E0")) if B else None,
    )

    # --- 當前股價卡(收盤價 + 當日漲跌)+ 近 5 日收盤曲線 ---
    price_val = ft.Text("--", size=34, weight=ft.FontWeight.BOLD, color="#212121")
    price_chg = ft.Text("查詢後顯示當前股價", size=14, color=getattr(C, "GREY_700", "#616161"))
    price_date = ft.Text("", size=11, color=getattr(C, "GREY", "#9E9E9E"))
    price_info = ft.Column(
        [ft.Text("當前股價", size=12, color=getattr(C, "GREY_700", "#616161")),
         price_val, price_chg, price_date],
        horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2,
    )
    price_holder = ft.Container(
        content=ft.Text("近 5 日收盤曲線", size=11, color=getattr(C, "GREY", "#9E9E9E")),
        height=130, alignment=ft.Alignment.CENTER,
        bgcolor=getattr(C, "GREY_100", "#F5F5F5"), border_radius=10,
    )
    price_card = ft.Container(
        content=ft.Column([price_info, price_holder], spacing=8),
        bgcolor="#FFFFFF", padding=14, border_radius=14,
        border=B.all(1, getattr(C, "GREY_300", "#E0E0E0")) if B else None,
    )

    # --- 中間:近 6 個月走勢圖容器 ---
    chart_holder = ft.Container(
        content=ft.Text("近 6 個月走勢將顯示於此", size=12, color=getattr(C, "GREY", "#9E9E9E")),
        height=240, alignment=ft.Alignment.CENTER,
        bgcolor=getattr(C, "GREY_100", "#F5F5F5"), border_radius=12,
    )

    # --- 底部:三張 KPI 卡(動能排名 / 絕對動能 / 波動)---
    def kpi_card(title: str):
        val = ft.Text("--", size=20, weight=ft.FontWeight.BOLD)
        box = ft.Container(
            content=ft.Column(
                [ft.Text(title, size=12, color=getattr(C, "GREY_700", "#616161")), val],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=2),
            bgcolor=getattr(C, "GREY_100", "#F5F5F5"), padding=14,
            border_radius=12, expand=True, alignment=ft.Alignment.CENTER,
        )
        return box, val

    rank_box, rank_val = kpi_card("動能排名")
    abs_box, abs_val = kpi_card("絕對動能")
    vol_box, vol_val = kpi_card("近60日波動")
    kpi_row = ft.Row([rank_box, abs_box, vol_box], spacing=10)

    # 動能脈絡文字(近 5/20/60 日動能 + 近一年回撤)
    bench = ft.Text("分析後顯示:近 5/20/60 日動能 與 近一年最大回撤", size=12,
                    color=getattr(C, "GREY_700", "#616161"),
                    text_align=ft.TextAlign.CENTER)

    # --- 技術面出場雷達（沿用持倉買入日；無持倉時以近 60 日估算）---
    radar_horizon = ft.Dropdown(
        label="持有週期", width=140, value="swing",
        options=[ft.DropdownOption(key="short", text="短線"),
                 ft.DropdownOption(key="swing", text="波段"),
                 ft.DropdownOption(key="long", text="中長期")])
    radar_atr_mode = ft.Dropdown(
        label="移動停利", width=140, value="balanced",
        options=[ft.DropdownOption(key="aggressive", text="積極 2×ATR"),
                 ft.DropdownOption(key="balanced", text="平衡 3×ATR"),
                 ft.DropdownOption(key="loose", text="寬鬆 4×ATR")])
    radar_panel = ft.Column([
        ft.Container(content=ft.Column([
            ft.Text("技術面出場雷達", size=18, weight=ft.FontWeight.BOLD),
            ft.Text("分析後顯示動能警訊、趨勢確認、關鍵防守與移動停利。", size=11),
        ], spacing=3), padding=14, border_radius=14, bgcolor="#ECEFF1")
    ], spacing=10)
    # --- 本月持有清單區塊(相對強弱輪動)---
    scan_title = ft.Text("本月持有清單(相對強弱輪動 + 絕對動能閘門 · 空頭轉現金)", size=14,
                         weight=ft.FontWeight.BOLD)
    scan_panel = ft.Column(
        [ft.Text("按上方「本月持有清單」開始(首次抓取真實資料較慢)",
                 size=12, color=getattr(C, "GREY", "#9E9E9E"))],
        spacing=8,
    )
    scan_msg = ft.Text("", size=12, weight=ft.FontWeight.BOLD)  # 加入庫存的結果提示
    scan_state = {"res": None}                  # 暫存最近一次本月持有結果(給零股配置器用)
    conservative_ai_panel = ft.Column([
        ft.Text("固定 6 項特徵、深度 2 的淺樹、每 20 個交易日再平衡。需同時通過 OOS Sharpe、006208、衰減與換手四道門檻。",
                size=12, color=getattr(C, "GREY", "#9E9E9E"))
    ], spacing=8)

    # 防禦模式開關(融資濾網):預設關=衝報酬;開=空頭少賠但平時報酬低
    defensive_switch = ft.Switch(
        label="防禦模式(融資濾網 · 空頭少賠、平時報酬低)",
        value=getattr(config, "ROTATION_DEFENSIVE", False))

    # --- 零股配置器(輸入預算 → 每支該買幾股,等權分散)---
    budget_field = ft.TextField(label="總預算", value="100000", width=120,
                                dense=True, text_size=14)
    try:
        alloc_btn = ft.Button(content="零股配置", icon=getattr(I, "CALCULATE", None))
    except Exception:
        alloc_btn = ft.ElevatedButton(text="零股配置", icon=getattr(I, "CALCULATE", None))
    alloc_panel = ft.Column(
        [ft.Text("輸入總預算 → 算每支該買幾股(零股、等權分散到本期進場標的)",
                 size=12, color=getattr(C, "GREY", "#9E9E9E"))],
        spacing=8)

    # --- 投資紀錄區塊(交易日誌:從 0 記錄投入/買賣點位與時間/報酬)---
    jrnl_title = ft.Text("匯入目前資產與現金", size=14,
                         weight=ft.FontWeight.BOLD)
    j_symbol = ft.TextField(label="代號", width=90, dense=True, text_size=14)
    j_amount = ft.TextField(label="持有數量", width=110, dense=True, text_size=14)
    j_price = ft.TextField(label="平均成本", width=100, dense=True, text_size=14)
    j_date = ft.TextField(label="成本日期(選填)", width=180,
                          dense=True, text_size=13)
    j_note = ft.TextField(label="備註(選填)", width=170, dense=True, text_size=13)
    try:
        j_add_btn = ft.Button(content="新增目前資產", icon=getattr(I, "ADD", None))
    except Exception:
        j_add_btn = ft.ElevatedButton(text="新增目前資產", icon=getattr(I, "ADD", None))
    j_cash = ft.TextField(label="目前現金", width=140, dense=True, text_size=14)
    try:
        j_cash_btn = ft.Button(content="儲存現金", icon=getattr(I, "ACCOUNT_BALANCE_WALLET", None))
    except Exception:
        j_cash_btn = ft.ElevatedButton(text="儲存現金")

    tx_title = ft.Text("快速記錄新交易", size=14, weight=ft.FontWeight.BOLD)
    tx_side = ft.Dropdown(
        label="交易", width=95, value="buy",
        options=[ft.DropdownOption(key="buy", text="買進"),
                 ft.DropdownOption(key="sell", text="賣出")])
    tx_symbol = ft.TextField(label="代號", width=90, dense=True, text_size=14)
    tx_qty = ft.TextField(label="成交數量", width=110, dense=True, text_size=14)
    tx_price = ft.TextField(label="成交價", width=100, dense=True, text_size=14)
    tx_date = ft.TextField(label="日期（選填）", width=160, dense=True, text_size=13)
    tx_note = ft.TextField(label="備註（選填）", width=170, dense=True, text_size=13)
    try:
        tx_add_btn = ft.Button(content="記錄交易", icon=getattr(I, "ADD_CHART", None))
    except Exception:
        tx_add_btn = ft.ElevatedButton(text="記錄交易")
    tx_msg = ft.Text("", size=11, color="#B71C1C")

    positions_title = ft.Text("目前持倉", size=14, weight=ft.FontWeight.BOLD)
    positions_panel = ft.Column([], spacing=8)

    j_msg = ft.Text("", size=11, color="#B71C1C")
    j_summary = ft.Text("尚無紀錄", size=12, weight=ft.FontWeight.BOLD,
                        color=getattr(C, "GREY_700", "#616161"))
    j_quote_status = ft.Text("", size=10, color=getattr(C, "GREY_700", "#616161"))
    j_summary_card = ft.Container(content=ft.Column([j_summary, j_quote_status], spacing=4),
                                  bgcolor="#E3F2FD", padding=12, border_radius=12)
    try:
        j_records_btn = ft.Button(content="展開投資紀錄", icon=getattr(I, "EXPAND_MORE", None))
    except Exception:
        j_records_btn = ft.ElevatedButton(text="展開投資紀錄",
                                          icon=getattr(I, "EXPAND_MORE", None))
    j_panel = ft.Column(
        [ft.Text("還沒有資產，輸入代號、持有數量與平均成本即可新增。",
                 size=12, color=getattr(C, "GREY", "#9E9E9E"))],
        spacing=8, visible=False)

    # 總資產快照(記錄總資產隨時間變化)
    try:
        j_snap_btn = ft.Button(content="記錄總資產", icon=getattr(I, "SAVE_ALT", None))
    except Exception:
        j_snap_btn = ft.ElevatedButton(text="記錄總資產", icon=getattr(I, "SAVE_ALT", None))
    j_hist_panel = ft.Column([], spacing=4)
    j_hist_chart = ft.Container(visible=False, height=170,
                                alignment=ft.Alignment.CENTER,
                                bgcolor="#FFFFFF", border_radius=10)
    try:
        j_history_btn = ft.Button(content="展開總資產紀錄",
                                  icon=getattr(I, "EXPAND_MORE", None))
    except Exception:
        j_history_btn = ft.ElevatedButton(text="展開總資產紀錄")
    j_history_content = ft.Column(
        [j_hist_chart, j_hist_panel], spacing=8, visible=False)

    # 定期定額(DCA)設定:代號 / 每期金額 / 頻率 / 起始日
    dca_title = ft.Text("定期定額(自動回補買入)", size=13, weight=ft.FontWeight.BOLD)
    dca_symbol = ft.TextField(label="代號", width=80, dense=True, text_size=14)
    dca_amount = ft.TextField(label="每期金額", width=100, dense=True, text_size=14)
    dca_freq = ft.Dropdown(
        label="頻率", width=100, value="monthly",
        options=[ft.DropdownOption(key=k, text=v) for k, v in journal.FREQ_LABELS.items()])
    dca_start = ft.TextField(label="起始日(YYYY-MM-DD)", width=160, dense=True,
                             text_size=14)
    try:
        dca_add_btn = ft.Button(content="新增計畫", icon=getattr(I, "ADD_TASK", None))
        dca_run_btn = ft.Button(content="立即更新", icon=getattr(I, "AUTORENEW", None))
    except Exception:
        dca_add_btn = ft.ElevatedButton(text="新增計畫", icon=getattr(I, "ADD_TASK", None))
        dca_run_btn = ft.ElevatedButton(text="立即更新", icon=getattr(I, "AUTORENEW", None))
    dca_msg = ft.Text("", size=11, color=getattr(C, "GREY_700", "#616161"))
    dca_panel = ft.Column(
        [ft.Text("尚無定期定額計畫。設定後按「立即更新」會依排程用歷史收盤價自動補買。",
                 size=12, color=getattr(C, "GREY", "#9E9E9E"))],
        spacing=8)

    # 將所有需更新的控件收進 dict,交給 module-level 的 apply_results 處理,
    # 使「按鈕後燈號變色 + KPI/圖表更新」的邏輯可獨立於 GUI 進行單元測試。
    ui = {
        "signal_name": signal_name, "signal_sub": signal_sub,
        "signal_card": signal_card, "chart_holder": chart_holder,
        "note_hint": trade_hint,
        "price_val": price_val, "price_chg": price_chg,
        "price_date": price_date, "price_holder": price_holder,
        "rank_val": rank_val, "abs_val": abs_val, "vol_val": vol_val,
        "mom_line": bench,
    }

    # --- 事件處理(async)---
    async def on_run(e):
        # 進入分析:鎖按鈕、顯示進度條
        run_btn.disabled = True
        progress.visible = True
        signal_sub.value = "分析中(載入動能池)..."
        page.update()

        symbol = (symbol_field.value or "2330").strip().upper()
        try:
            # 輪動適配與出場雷達共用同一批本地行情。
            res = await asyncio.to_thread(rotation.analyze_stock, symbol)
            apply_fit(ui, res)
            price_df = await asyncio.to_thread(load_ohlcv, symbol)
            open_trade = next((trade for trade in journal.list_trades()
                               if trade["status"] == "open" and trade["symbol"] == symbol), None)
            settings = RadarSettings(
                horizon=radar_horizon.value or "swing",
                atr_mode=radar_atr_mode.value or "balanced",
                buy_price=open_trade.get("buy_price") if open_trade else None,
                buy_date=open_trade.get("buy_time") if open_trade else None,
            )
            radar = await asyncio.to_thread(analyze_exit_radar, price_df, settings)
            label = f"{res.get('name') or ''} {symbol}".strip()
            radar_panel.controls = make_exit_radar_controls(radar, label)
        except Exception as ex:
            signal_name.value = "分析失敗"
            signal_sub.value = str(ex)
            signal_card.bgcolor = "#B71C1C"
            radar_panel.controls = [ft.Text(f"出場雷達載入失敗：{ex}",
                                            size=12, color="#B71C1C")]
        finally:
            # 一次性收尾更新
            run_btn.disabled = False
            progress.visible = False
            page.update()

    run_btn.on_click = on_run

    # --- 本月持有清單事件(async;相對強弱輪動)---
    async def on_scan(e):
        scan_btn.disabled = True
        scan_progress.visible = True
        scan_panel.controls = [ft.Text("計算動能排名中,請稍候...", size=12)]
        page.update()
        try:
            res = await asyncio.to_thread(monthly_holdings, defensive_switch.value)
            scan_state["res"] = res
            if res and res.get("holdings"):
                held = {t["symbol"]: t for t in journal.list_trades()
                        if t["status"] == "open" and t["source"] == "rotation"}
                scan_panel.controls = make_holdings_rows(
                    res, on_add=on_add_inventory, held_trades=held,
                    on_renew=on_renew_rotation)
            else:
                scan_panel.controls = [ft.Text(
                    "無法產生持有清單(觀察清單資料不足)", size=12)]
        except Exception as ex:
            scan_panel.controls = [ft.Text(f"計算失敗:{ex}", size=12, color="#B71C1C")]
        finally:
            scan_btn.disabled = False
            scan_progress.visible = False
            page.update()

    scan_btn.on_click = on_scan

    async def on_run_conservative_ai(e):
        conservative_ai_btn.disabled = True
        scan_progress.visible = True
        conservative_ai_panel.controls = [ft.Text("以 purged walk-forward 驗證中，約需一分鐘...", size=12)]
        page.update()
        try:
            result = await asyncio.to_thread(run_conservative_wfa, None, False)
            color = "#1B5E20" if result["gate"]["eligible"] else "#B71C1C"
            conservative_ai_panel.controls = [
                ft.Text("保守 AI 驗證", size=14, weight=ft.FontWeight.BOLD, color=color),
                ft.Text(format_conservative_report(result), size=12, selectable=True),
            ]
        except Exception as ex:
            conservative_ai_panel.controls = [ft.Text(f"驗證失敗：{ex}", size=12, color="#B71C1C")]
        finally:
            conservative_ai_btn.disabled = False
            scan_progress.visible = False
            page.update()

    conservative_ai_btn.on_click = on_run_conservative_ai

    # --- 零股配置器事件 ---
    async def on_allocate(e):
        alloc_btn.disabled = True
        alloc_panel.controls = [ft.Text("計算中...", size=12)]
        page.update()
        try:
            budget = float(budget_field.value)
            res = scan_state.get("res")
            if not res:                          # 還沒按過本月持有 -> 先算一次
                res = await asyncio.to_thread(monthly_holdings, defensive_switch.value)
                scan_state["res"] = res
            held = (res.get("held") or res.get("holdings") or []) if res else []
            if not held:
                alloc_panel.controls = [ft.Text(
                    "目前無建議進場標的(可能全數轉現金)", size=12)]
            else:
                alloc = await asyncio.to_thread(
                    rotation.allocate_odd_lots, budget, held, None, res.get("names"))
                alloc_panel.controls = make_allocation_rows(alloc)
        except Exception as ex:
            alloc_panel.controls = [ft.Text(f"計算失敗:{ex}", size=12, color="#B71C1C")]
        finally:
            alloc_btn.disabled = False
            page.update()

    alloc_btn.on_click = on_allocate

    # --- 每日資料更新 ---
    def _update_targets():
        """要更新的標的:輪動池 + 觀察清單 + 對標 + 庫存中持有(去重)。"""
        syms = (list(getattr(config, "UNIVERSE", []))
                + list(config.WATCHLIST)
                + [getattr(config, "BENCHMARK_SYMBOL", "006208")]
                + [t["symbol"] for t in journal.list_trades()
                   if t["status"] == "open"])
        return list(dict.fromkeys(s for s in syms if s))

    async def on_update(e):
        update_btn.disabled = True
        scan_progress.visible = True
        scan_msg.value = "更新每日資料中(抓取最新收盤,首次較久)..."
        scan_msg.color = getattr(C, "GREY_700", "#616161")
        page.update()
        try:
            # 手動按 -> 略過時間節流,但仍只重抓「落後」的標的
            res = await asyncio.to_thread(
                update_symbols, _update_targets(), False, True)
            try:                                   # 順便刷新費半市場燈
                from core import market_regime
                await asyncio.to_thread(market_regime.refresh_sox)
            except Exception:
                pass
            scan_msg.value = (
                f"已更新到 {res['asof']}　更新 {res['updated']} 檔 · "
                f"已最新 {res['current']} · 失敗 {res['failed']} · 費半燈已更新")
            scan_msg.color = "#2E7D32"
            refresh_journal()       # 庫存現價/損益/停損停利警示一起刷新
        except Exception as ex:
            scan_msg.value = f"更新失敗:{ex}"
            scan_msg.color = "#B71C1C"
        finally:
            update_btn.disabled = False
            scan_progress.visible = False
            page.update()

    update_btn.on_click = on_update

    async def _auto_update():
        """啟動時背景增量更新一次(受 6 小時節流;不打擾操作)。"""
        try:
            quote_res = await asyncio.to_thread(journal.refresh_open_market_data)
            res = await asyncio.to_thread(update_symbols, _update_targets())
            try:                                   # 背景刷新費半市場燈
                from core import market_regime
                await asyncio.to_thread(market_regime.refresh_sox)
            except Exception:
                pass
            refresh_journal()
            if quote_res["symbols"]:
                j_quote_status.value = (
                    f"市值資料：更新 {quote_res['updated']} 檔、已最新 {quote_res['current']} 檔、"
                    f"無法取得 {quote_res['failed']} 檔（以最近收盤計算）")
            if res.get("updated"):
                scan_msg.value = f"已自動更新每日資料到 {res['asof']}(更新 {res['updated']} 檔)"
                scan_msg.color = "#2E7D32"
            page.update()
        except Exception:
            pass

    # 續抱:本期又選到已持有的標的 -> 更新輪替日 + 依現價重算停損/停利(月度移動停損)
    def on_renew_rotation(symbol):
        try:
            cur = journal.get_last_close(symbol)
            sl = tk = None
            if cur:
                sl, tk = rotation.stop_take_levels(symbol, cur)
                sl, tk = round(sl, 2), (round(tk, 2) if tk else None)
            n = journal.mark_rotation(symbol, stop_loss=sl, take_profit=tk)
            if n:
                scan_msg.value = (f"已續抱 {symbol}:輪替日更新為今天"
                                  + (f",停損/停利依現價重算({sl} / {tk})" if sl else ""))
                scan_msg.color = "#1565C0"
                refresh_journal()
                # 重畫持有卡上的「最近輪替」
                res = scan_state.get("res")
                if res and res.get("holdings"):
                    held = {t["symbol"]: t for t in journal.list_trades()
                            if t["status"] == "open" and t["source"] == "rotation"}
                    scan_panel.controls = make_holdings_rows(
                        res, on_add=on_add_inventory, held_trades=held,
                        on_renew=on_renew_rotation)
            else:
                scan_msg.value = f"{symbol} 沒有持有中的輪動紀錄可更新"
                scan_msg.color = "#B71C1C"
        except Exception as ex:
            scan_msg.value = f"續抱更新失敗:{ex}"
            scan_msg.color = "#B71C1C"
        page.update()

    # 一鍵把本月推薦的標的加入庫存(source='rotation',另記停損/停利)
    def on_add_inventory(symbol, name, amt_field, price_field, stop_field, take_field):
        try:
            amount = float(amt_field.value)
            price = float(price_field.value)
            stop = (float(stop_field.value)
                    if (stop_field.value or "").strip() else None)
            take = (float(take_field.value)
                    if (take_field.value or "").strip() else None)
            journal.add_buy(symbol, amount, price, name=name, source="rotation",
                            stop_loss=stop, take_profit=take)
            scan_msg.value = (f"已加入庫存:{name} {symbol}　投入 {amount:,.0f} @ "
                              f"{price:.2f}"
                              + (f"　停損 {stop:.2f}" if stop else "")
                              + (f"　停利 {take:.2f}" if take else "")
                              + "（見「投資紀錄」分頁）")
            scan_msg.color = "#2E7D32"
            refresh_journal()        # 同步更新投資紀錄分頁
        except Exception as ex:
            scan_msg.value = f"加入失敗:{ex}"
            scan_msg.color = "#B71C1C"
        page.update()

    # --- 投資紀錄事件 ---
    j_state = {"editing": None, "records_expanded": False, "history_expanded": False}

    def refresh_journal():
        """重建紀錄清單 + 彙總 + 總資產快照 + 定期定額清單(讀 DB),由呼叫端 page.update()。"""
        trades = journal.list_trades()
        current_positions = journal.positions()
        positions_panel.controls = (
            make_position_rows(current_positions) if current_positions else
            [ft.Text("目前沒有持倉", size=12,
                     color=getattr(C, "GREY", "#9E9E9E"))]
        )
        j_cash.value = f"{journal.cash_balance():g}"
        records_label = ("收合" if j_state["records_expanded"] else "展開")
        records_label += f"投資紀錄（{len(trades)} 筆）"
        j_records_btn.content = records_label
        j_records_btn.visible = bool(trades)
        j_panel.visible = bool(trades) and j_state["records_expanded"]
        if trades:
            open_trades = [t for t in trades if t["status"] == "open"]
            closed_trades = [t for t in trades if t["status"] == "closed"]

            def _rows(ts):
                return make_journal_rows(
                    ts, on_sell_trade, on_delete_trade,
                    on_edit=on_edit_trade, editing_id=j_state["editing"],
                    on_save=on_save_trade, on_cancel=on_cancel_edit,
                    on_reopen=on_reopen_trade)

            controls = [ft.Text(f"持有中({len(open_trades)})", size=13,
                                weight=ft.FontWeight.BOLD, color="#1565C0")]
            controls += (_rows(open_trades) if open_trades else
                         [ft.Text("目前沒有持有中的標的", size=12,
                                  color=getattr(C, "GREY", "#9E9E9E"))])
            # 已平倉獨立成一區,與持有中分開顯示
            if closed_trades:
                controls.append(ft.Divider())
                controls.append(ft.Text(
                    f"已平倉({len(closed_trades)})", size=13,
                    weight=ft.FontWeight.BOLD,
                    color=getattr(C, "GREY_700", "#616161")))
                controls += _rows(closed_trades)
            j_panel.controls = controls
        else:
            j_panel.controls = [ft.Text(
                "還沒有資產，輸入代號、持有數量與平均成本即可新增。",
                size=12, color=getattr(C, "GREY", "#9E9E9E"))]
        j_summary.value = make_summary_text(journal.summary())
        # 總資產快照清單 + 成長曲線(≥2 筆才畫圖)
        hist = journal.list_asset_history()
        history_label = ("收合" if j_state["history_expanded"] else "展開")
        j_history_btn.content = f"{history_label}總資產紀錄（{len(hist)} 筆）"
        j_history_btn.visible = bool(hist)
        j_history_content.visible = bool(hist) and j_state["history_expanded"]
        j_hist_panel.controls = (
            make_asset_history_rows(hist, on_delete=on_delete_snapshot) if hist else
            [ft.Text("尚無快照,按「記錄總資產」存一筆", size=11,
                     color=getattr(C, "GREY", "#9E9E9E"))])
        if len(hist) >= 2:
            j_hist_chart.content = asset_history_image(hist)
            j_hist_chart.visible = True
        else:
            j_hist_chart.visible = False
        # 定期定額計畫清單
        plans = journal.list_dca_plans()
        dca_panel.controls = (make_dca_rows(plans, on_toggle_dca, on_delete_dca) if plans else
                              [ft.Text("尚無定期定額計畫。設定後按「立即更新」自動補買。",
                                       size=12, color=getattr(C, "GREY", "#9E9E9E"))])

    def on_toggle_records(e):
        j_state["records_expanded"] = not j_state["records_expanded"]
        refresh_journal()
        page.update()

    def on_toggle_history(e):
        j_state["history_expanded"] = not j_state["history_expanded"]
        refresh_journal()
        page.update()

    def on_add_buy(e):
        j_msg.value = ""
        try:
            sym = (j_symbol.value or "").strip()
            quantity = float(j_amount.value)
            price = float(j_price.value)
            journal.add_current_asset(
                sym, quantity, price,
                asof=(j_date.value or "").strip() or None,
                note=(j_note.value or "").strip())
            j_msg.value = f"已匯入 {sym.upper()}：{quantity:g} 單位，平均成本 {price:g}"
            j_msg.color = "#2E7D32"
            j_state["records_expanded"] = False
            j_symbol.value = j_amount.value = j_price.value = ""
            j_date.value = j_note.value = ""
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"匯入失敗：{ex}"
            j_msg.color = "#B71C1C"
        page.update()

    def on_set_cash(e):
        j_msg.value = ""
        try:
            balance = journal.set_cash_balance(float(j_cash.value))
            j_msg.value = f"目前現金已更新為 {balance:,.0f}"
            j_msg.color = "#2E7D32"
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"現金更新失敗：{ex}"
            j_msg.color = "#B71C1C"
        page.update()

    def on_record_transaction(e):
        tx_msg.value = ""
        try:
            side = tx_side.value or "buy"
            symbol = (tx_symbol.value or "").strip()
            quantity = float(tx_qty.value)
            price = float(tx_price.value)
            when = (tx_date.value or "").strip() or None
            note = (tx_note.value or "").strip()
            if side == "sell":
                journal.record_sell(symbol, quantity, price, when, note)
                action = "賣出"
            else:
                journal.record_buy(symbol, quantity, price, when, note)
                action = "買進"
            tx_msg.value = f"已記錄 {action} {symbol.upper()} {quantity:g} 單位 @ {price:g}"
            tx_msg.color = "#2E7D32"
            tx_symbol.value = tx_qty.value = tx_price.value = ""
            tx_date.value = tx_note.value = ""
            j_state["records_expanded"] = False
            refresh_journal()
        except Exception as ex:
            tx_msg.value = f"交易失敗：{ex}"
            tx_msg.color = "#B71C1C"
        page.update()

    def on_sell_trade(trade_id, price_field, qty_field=None):
        """股數留空 = 整筆平倉;有填 = 部分賣出(拆成已平倉 + 剩餘持有)。"""
        j_msg.value = ""
        try:
            qty = (qty_field.value or "").strip() if qty_field is not None else ""
            if qty:
                journal.sell_partial(trade_id, float(qty), float(price_field.value))
            else:
                journal.close_trade(trade_id, float(price_field.value))
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"賣出失敗:{ex}"
        page.update()

    def on_edit_trade(trade_id):
        j_msg.value = ""
        j_state["editing"] = trade_id
        refresh_journal()
        page.update()

    def on_cancel_edit():
        j_state["editing"] = None
        refresh_journal()
        page.update()

    def on_save_trade(trade_id, fmap):
        """把編輯卡的欄位寫回(空白=不變;停損/停利空白=清除)。"""
        j_msg.value = ""
        try:
            journal.update_trade(trade_id,
                                 **{k: fl.value for k, fl in fmap.items()})
            j_state["editing"] = None
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"儲存失敗:{ex}"
        page.update()

    def on_reopen_trade(trade_id):
        j_msg.value = ""
        try:
            journal.reopen_trade(trade_id)
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"復原失敗:{ex}"
        page.update()

    def on_delete_snapshot(snap_id):
        j_msg.value = ""
        try:
            journal.delete_asset_snapshot(snap_id)
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"刪除快照失敗:{ex}"
        page.update()

    def on_delete_trade(trade_id):
        j_msg.value = ""
        try:
            journal.delete_trade(trade_id)
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"刪除失敗:{ex}"
        page.update()

    def on_snapshot(e):
        j_msg.value = ""
        try:
            journal.snapshot_assets()
            j_state["history_expanded"] = False
            refresh_journal()
        except Exception as ex:
            j_msg.value = f"記錄總資產失敗:{ex}"
        page.update()

    def on_add_dca(e):
        dca_msg.value = ""
        try:
            sym = (dca_symbol.value or "").strip()
            amt = float(dca_amount.value)
            journal.add_dca_plan(sym, amt, dca_freq.value or "monthly",
                                 (dca_start.value or "").strip() or None)
            dca_symbol.value = dca_amount.value = dca_start.value = ""
            refresh_journal()
            dca_msg.value = "已新增計畫。按「立即更新」開始自動補買。"
        except Exception as ex:
            dca_msg.value = f"新增計畫失敗:{ex}"
        page.update()

    async def on_run_dca(e):
        dca_run_btn.disabled = True
        dca_msg.value = "更新中(抓取歷史價並回補)..."
        page.update()
        try:
            r = await asyncio.to_thread(journal.run_dca_update)
            refresh_journal()
            dca_msg.value = f"已自動補買 {r['created']} 筆({r['plans']} 個計畫)。"
        except Exception as ex:
            dca_msg.value = f"更新失敗:{ex}"
        finally:
            dca_run_btn.disabled = False
            page.update()

    def on_toggle_dca(plan_id, active):
        try:
            journal.set_dca_active(plan_id, active)
            refresh_journal()
        except Exception as ex:
            dca_msg.value = f"切換失敗:{ex}"
        page.update()

    def on_delete_dca(plan_id):
        try:
            journal.delete_dca_plan(plan_id)
            refresh_journal()
        except Exception as ex:
            dca_msg.value = f"刪除失敗:{ex}"
        page.update()

    j_add_btn.on_click = on_add_buy
    j_cash_btn.on_click = on_set_cash
    tx_add_btn.on_click = on_record_transaction
    j_records_btn.on_click = on_toggle_records
    j_history_btn.on_click = on_toggle_history
    j_snap_btn.on_click = on_snapshot
    dca_add_btn.on_click = on_add_dca
    dca_run_btn.on_click = on_run_dca
    refresh_journal()        # 啟動時載入既有紀錄

    # --- 組裝版面(三個分頁,不再一路向下延伸)---
    _scroll = getattr(ft, "ScrollMode", None) and ft.ScrollMode.AUTO

    # 分頁 1:個股分析
    tab_stock = ft.Column(
        [
            ft.Row([symbol_field, run_btn],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            progress,
            signal_card,
            trade_card,
            price_card,
            chart_holder,
            kpi_row,
            bench,
            ft.Divider(),
            ft.Row([radar_horizon, radar_atr_mode], spacing=8),
            radar_panel,
        ],
        spacing=16, scroll=_scroll, expand=True,
    )

    # 分頁 2:本月持有(相對強弱輪動)+ 零股配置器
    tab_holdings = ft.Column(
        [ft.Row([scan_btn, update_btn, conservative_ai_btn], spacing=8),
         defensive_switch,
         ft.Text("保守 AI 驗證（研究用途；未通過門檻不產生交易建議）", size=14,
                 weight=ft.FontWeight.BOLD),
         conservative_ai_panel,
         ft.Divider(),
         scan_progress, scan_title, scan_msg, scan_panel,
         ft.Divider(),
         ft.Text("零股配置器(輸入預算 → 每支該買幾股)", size=14,
                 weight=ft.FontWeight.BOLD),
         ft.Row([budget_field, alloc_btn], spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER),
         alloc_panel],
        spacing=16, scroll=_scroll, expand=True,
    )

    # 分頁 3:投資紀錄(手動買賣 + 總資產 + 定期定額)
    tab_journal = ft.Column(
        [
            jrnl_title,
            ft.Text("第一次使用時，逐筆輸入目前持倉；這些期初資料不會扣除現金。",
                    size=11, color=getattr(C, "GREY_700", "#616161")),
            ft.Row([j_symbol, j_amount, j_price],
                   alignment=ft.MainAxisAlignment.START, spacing=8),
            ft.Row([j_date, j_note],
                   alignment=ft.MainAxisAlignment.START, spacing=8),
            ft.Row([j_add_btn, j_cash, j_cash_btn],
                   alignment=ft.MainAxisAlignment.START, spacing=8),
            j_msg,
            ft.Divider(),
            tx_title,
            ft.Row([tx_side, tx_symbol, tx_qty],
                   alignment=ft.MainAxisAlignment.START, spacing=8),
            ft.Row([tx_price, tx_date, tx_note],
                   alignment=ft.MainAxisAlignment.START, spacing=8),
            tx_add_btn,
            tx_msg,
            ft.Divider(),
            ft.Text("績效與成本", size=14, weight=ft.FontWeight.BOLD),
            j_summary_card,
            ft.Divider(),
            positions_title,
            positions_panel,
            ft.Divider(),
            ft.Text("交易紀錄", size=14, weight=ft.FontWeight.BOLD),
            j_records_btn,
            j_panel,
            ft.Divider(),
            ft.Text("總資產紀錄", size=13, weight=ft.FontWeight.BOLD),
            j_snap_btn,
            j_history_btn,
            j_history_content,
            ft.Divider(),
            dca_title,
            ft.Row([dca_symbol, dca_amount, dca_freq],
                   alignment=ft.MainAxisAlignment.START, spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([dca_start, dca_add_btn, dca_run_btn],
                   alignment=ft.MainAxisAlignment.START, spacing=8,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            dca_msg,
            dca_panel,
        ],
        spacing=16, scroll=_scroll, expand=True,
    )

    # 新版 Flet(v1)分頁:Tabs(length) 包 TabBar(分頁標籤) + TabBarView(各頁內容);
    # 舊版退回單一捲動 Column,確保任何環境都能顯示。
    try:
        tabs = ft.Tabs(
            length=3, selected_index=0, expand=True,
            content=ft.Column(
                [
                    ft.TabBar(tabs=[
                        ft.Tab(label="個股分析", icon=getattr(I, "INSIGHTS", None)),
                        ft.Tab(label="本月持有", icon=getattr(I, "LEADERBOARD", None)),
                        ft.Tab(label="投資紀錄", icon=getattr(I, "BOOK", None)),
                    ]),
                    ft.TabBarView(
                        controls=[tab_stock, tab_holdings, tab_journal],
                        expand=True),
                ],
                expand=True, spacing=8,
            ),
        )
        page.add(tabs)
    except Exception:
        # 舊版相容:無分頁元件時,退回單欄捲動版面
        page.scroll = _scroll
        page.add(ft.Column(
            [ft.Text("個股分析", size=16, weight=ft.FontWeight.BOLD), tab_stock,
             ft.Divider(),
             ft.Text("本月持有", size=16, weight=ft.FontWeight.BOLD), tab_holdings,
             ft.Divider(),
             ft.Text("投資紀錄", size=16, weight=ft.FontWeight.BOLD), tab_journal],
            spacing=16, scroll=_scroll))

    # 啟動後背景自動更新每日資料一次(受節流;不卡 UI)
    if hasattr(page, "run_task"):
        try:
            page.run_task(_auto_update)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 進入點:
#   桌面視窗:  python main.py
#   瀏覽器:    flet run -w main.py   (Flet 0.85 web 模式請用 CLI;
#              programmatic 的 view=WEB_BROWSER 在 0.85 會 add_web_socket 報錯)
# 新版 ft.run(main)、舊版 ft.app(target=main),try/except 相容。
async def main(page: ft.Page):
    """本機直接開啟；Web 部署先驗證 APP_PASSWORD 才載入投資資料。"""
    page.title = "量化交易燈號系統｜登入"
    page.theme_mode = getattr(ft.ThemeMode, "LIGHT", None)
    page.padding = 20
    configured_password = os.environ.get("APP_PASSWORD", "")
    web_deployment = os.environ.get("FLET_FORCE_WEB_SERVER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    preferences = ft.SharedPreferences()

    def clear_page():
        page.appbar = None
        try:
            page.clean()
        except Exception:
            page.controls.clear()

    async def logout(e=None):
        try:
            await preferences.remove(AUTH_STORAGE_KEY)
        except Exception:
            pass
        clear_page()
        show_login()
        page.update()

    def unlock():
        clear_page()
        _build_app(page, on_logout=logout)

    async def submit_login(e=None):
        candidate = password_field.value or ""
        if configured_password and hmac.compare_digest(candidate, configured_password):
            if remember_box.value:
                await preferences.set(AUTH_STORAGE_KEY, _auth_token(configured_password))
            else:
                try:
                    await preferences.remove(AUTH_STORAGE_KEY)
                except Exception:
                    pass
            unlock()
            return
        login_message.value = "密碼錯誤"
        login_message.color = "#B71C1C"
        password_field.value = ""
        page.update()

    def show_login():
        nonlocal password_field, remember_box, login_message
        password_field = ft.TextField(
            label="密碼", password=True, can_reveal_password=True,
            width=300, autofocus=True, on_submit=submit_login)
        remember_box = ft.Checkbox(label="在此瀏覽器記住登入", value=True)
        login_message = ft.Text("", size=12)
        try:
            login_button = ft.Button(
                content="登入", icon=getattr(I, "LOGIN", None),
                on_click=submit_login)
        except Exception:
            login_button = ft.ElevatedButton(text="登入", on_click=submit_login)
        if not configured_password:
            password_field.disabled = True
            remember_box.disabled = True
            login_button.disabled = True
            login_message.value = "網站尚未設定 APP_PASSWORD，已安全鎖定。"
            login_message.color = "#B71C1C"
        card = ft.Container(
            content=ft.Column([
                ft.Text("量化交易燈號系統", size=24, weight=ft.FontWeight.BOLD),
                ft.Text("請先登入，才能查看或修改投資資料。", size=12),
                password_field,
                remember_box,
                login_button,
                login_message,
            ], spacing=12, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            width=360, padding=24, border_radius=16, bgcolor="#FFFFFF")
        page.add(ft.Column(
            [card], expand=True,
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER))

    if not web_deployment:
        clear_page()
        _build_app(page, on_logout=None)
        return

    password_field = remember_box = login_message = None
    remembered = None
    if configured_password:
        try:
            remembered = await preferences.get(AUTH_STORAGE_KEY)
        except Exception:
            remembered = None
    expected = _auth_token(configured_password) if configured_password else ""
    if (isinstance(remembered, str) and expected
            and hmac.compare_digest(remembered, expected)):
        unlock()
    else:
        show_login()


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        ft.run(main)                 # Flet 0.80+ 桌面
    except (AttributeError, TypeError):
        ft.app(target=main)          # 舊版 fallback
