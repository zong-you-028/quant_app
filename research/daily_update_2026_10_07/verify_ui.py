"""Real 150-stock startup and rotation UI, isolated desktop/mobile checks."""
from contextlib import closing
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time

import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
checks = []
report = {"checks": checks, "accounting_actions_clicked": False,
          "actual_startup_and_rotation": True, "positions_are_disposable_and_empty": True,
          "browser_frontend_assets_allowed": True,
          "external_network_counter_scope": "Python app/data pipeline; browser Flet frontend assets excluded"}
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1440,1500")
service = Service(r"C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe")
service.creation_flags = subprocess.CREATE_NO_WINDOW
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 60)


def body():
    return driver.find_element(By.TAG_NAME, "body").text


def check(name, result):
    assert result, name
    checks.append(name)


def button(label):
    return wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='button' and normalize-space(.)='{label}']"))


def tab(label):
    item = wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='tab' and @aria-label='{label}']"))
    item.click()
    wait.until(lambda d: item.get_attribute("aria-selected") == "true")
    time.sleep(.8)
    return item


def layout(prefix):
    geometry = driver.execute_script("""
      const all = [...document.querySelectorAll('[role="button"], [role="tab"]')];
      const controls = all.filter(e => e.getClientRects().length).map(e => {
        const r = e.getBoundingClientRect();
        return {label: e.getAttribute('aria-label') || e.innerText,
                left: r.left, right: r.right, top: r.top, bottom: r.bottom};
      });
      const beyond = [...document.querySelectorAll('*')].filter(e => {
        const r = e.getBoundingClientRect();
        return r.right > innerWidth + 1 && r.width > 0;
      }).slice(0, 30).map(e => ({tag:e.tagName, role:e.getAttribute('role'),
        text:(e.innerText || e.getAttribute('aria-label') || '').slice(0,120),
        rect: JSON.stringify(e.getBoundingClientRect().toJSON()),
        style:e.getAttribute('style')}));
      const visibleBeyond = [...document.querySelectorAll('*')].filter(e => {
        const r = e.getBoundingClientRect(), s = getComputedStyle(e);
        return r.right > innerWidth + 1 && r.width > 0 &&
               s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0';
      }).map(e => ({tag:e.tagName, role:e.getAttribute('role'),
                   text:(e.innerText || '').slice(0,120)}));
      return {width: innerWidth, bodyWidth: document.body.scrollWidth,
              documentWidth: document.documentElement.scrollWidth, controls, beyond, visibleBeyond};
    """)
    report[prefix + "_geometry"] = geometry
    check(prefix + "_no_document_horizontal_overflow",
          geometry["documentWidth"] <= geometry["width"] + 1 and not geometry["visibleBeyond"])
    # Flutter keeps a visibility:hidden text-measurement <p> off-screen;
    # its raw body.scrollWidth is diagnostic, not visible scrolling width.
    report["hidden_flutter_text_measurement_excluded"] = True
    for label in ("重新計算名單", "更新每日資料", "核對我的持倉"):
        matches = [r for r in geometry["controls"] if r["label"].strip() == label]
        check(prefix + "_" + label + "_inside_width",
              bool(matches) and all(r["left"] >= -1 and r["right"] <= geometry["width"] + 1 for r in matches))


try:
    with tempfile.TemporaryDirectory(prefix="quant_150_ui_") as directory:
        market = Path(directory) / "market.db"
        with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as source, market.open("wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copyfile(ROOT / "data_seed/sox.csv", Path(directory) / "sox.csv")
        metadata = Path(directory) / "ui_meta.json"
        env = dict(os.environ, APP_DATA_DIR=directory, JOURNAL_DATABASE_URL="", APP_PASSWORD="",
                   UNIVERSE_UI_PORT="8784", UNIVERSE_UI_META=str(metadata))
        with (HERE / "ui_server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(HERE / "preview.py")], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 30
                while True:
                    try:
                        if requests.get("http://127.0.0.1:8784", timeout=1).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    assert process.poll() is None and time.monotonic() < deadline, "preview startup"
                    time.sleep(.5)
                driver.get("http://127.0.0.1:8784")
                wait.until(lambda d: d.execute_script("return !!document.querySelector('flt-semantics-placeholder')"))
                driver.execute_script("document.querySelector('flt-semantics-placeholder').click()")
                wait.until(lambda d: "資料更新完成；" in body())
                check("first_startup_restores_one_bar", "更新 1" in body() and "已達目標 150" in body())
                tab("策略輪動")
                button("更新每日資料").click()
                wait.until(lambda d: "本日已保存 1 檔" in body() and "資料更新完成；" in body())
                check("second_update_reuses_daily_database_record", "未重複下載" in body() and "已達目標 151" in body())
                tab("個股分析")
                text = body()
                check("startup_current_151_no_failed_pending", "已達目標 151" in text and "失敗 0" in text and "待續抓 0" in text)
                check("stock_tab_pool_150_markets_and_date", "估算市值前 150 名（上市 126／上櫃 24）" in text and "2026-10-02" in text)
                check("stock_tab_overfit_disclaimer", "過擬合未排除" in text)
                driver.save_screenshot(str(HERE / "ui_stock_desktop.png"))
                report["stock_desktop_text"] = text
                tab("策略輪動")
                check("rotation_has_pool_before_computation", "估算市值前 150 名（上市 126／上櫃 24）" in body())
                check("recheck_disabled_before_model", driver.execute_script("return !!arguments[0].closest('[aria-disabled=\"true\"]')", button("核對我的持倉")))
                button("計算輪動名單").click()
                wait.until(lambda d: "輪動計算完成" in body())
                time.sleep(1)
                text = body()
                check("rotation_rankable_141_of_150", "最近換股日可排名 141／150 檔" in text)
                check("rotation_eight_stock_max_and_buffer", "模型最多 8 檔" in text and "不是持有 16 檔" in text)
                check("rotation_overfit_disclaimer", "過擬合未排除" in text)
                check("rotation_current_holdings_and_empty_book", "目前應持有 8/8 檔" in text and "帳本尚無持倉" in text)
                layout("desktop")
                driver.save_screenshot(str(HERE / "ui_rotation_desktop.png"))
                report["rotation_desktop_text"] = text
                driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 1500, "deviceScaleFactor": 1, "mobile": True})
                time.sleep(2)
                driver.save_screenshot(str(HERE / "ui_rotation_mobile.png"))
                layout("mobile")
                check("mobile_compute_and_recheck_reachable", button("重新計算名單").is_displayed() and button("核對我的持倉").is_displayed())
                driver.save_screenshot(str(HERE / "ui_rotation_mobile.png"))
                report["rotation_mobile_text"] = body()
                button("核對我的持倉").click()
                wait.until(lambda d: "目前應持有 8/8 檔" in body())
                check("mobile_recheck_retains_empty_book", "帳本尚無持倉" in body())
                tab("個股分析")
                check("mobile_tab_switch_retains_pool_and_warning", "估算市值前 150 名" in body() and "過擬合未排除" in body())
                driver.save_screenshot(str(HERE / "ui_stock_mobile.png"))
                tab("策略輪動")
                check("mobile_return_to_holdings_retains_model", "最近換股日可排名 141／150 檔" in body() and "目前應持有 8/8 檔" in body())
                # Use a new Chrome profile/session at 390px. Flet reconnects
                # an existing app session on refresh, preserving its status.
                driver.quit()
                mobile_options = webdriver.ChromeOptions()
                mobile_options.add_argument("--headless=new")
                mobile_options.add_argument("--window-size=390,1500")
                mobile_service = Service(r"C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe")
                mobile_service.creation_flags = subprocess.CREATE_NO_WINDOW
                driver = webdriver.Chrome(service=mobile_service, options=mobile_options)
                wait = WebDriverWait(driver, 60)
                driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 1500, "deviceScaleFactor": 1, "mobile": True})
                driver.get("http://127.0.0.1:8784")
                wait.until(lambda d: d.execute_script("return !!document.querySelector('flt-semantics-placeholder')"))
                driver.execute_script("document.querySelector('flt-semantics-placeholder').click()")
                wait.until(lambda d: "資料更新完成；" in body())
                check("fresh_mobile_startup_current_151", "已達目標 151" in body() and "估算市值前 150 名" in body())
                tab("策略輪動")
                button("計算輪動名單").click()
                wait.until(lambda d: "輪動計算完成" in body())
                time.sleep(1)
                layout("fresh_mobile")
                check("fresh_mobile_rank_and_holdings_visible", "最近換股日可排名 141／150 檔" in body() and "目前應持有 8/8 檔" in body())
                driver.save_screenshot(str(HERE / "ui_rotation_mobile_fresh.png"))
                report["fresh_mobile_text"] = body()
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                report["preview"] = meta
                check("only_one_controlled_source_call_across_sessions", meta["controlled_historical_gap"]["source_calls"] == ["2330"])
                check("daily_record_survives_fresh_browser_session", meta["startup_update"]["daily_skipped_symbols"] == ["2330"])
                check("formal_database_and_external_network_zero", all(meta["isolation"][key] == 0 for key in
                      ("formal_database_attempts", "outside_database_attempts", "external_network_attempts")))
                check("actual_model_rank_count_141_and_current_8", meta["model"]["latest_ranking_count"] == 141 and len(meta["model"]["current_holdings"]) == 8)
                with closing(sqlite3.connect(market)) as conn:
                    rows = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                            for table in ("trades", "cash_movements", "asset_history", "dca_plans", "corporate_actions")}
                report["disposable_journal_rows"] = rows
                check("disposable_journal_all_rows_zero", all(value == 0 for value in rows.values()))
                report["status"] = "passed"
            finally:
                process.terminate()
                process.wait(timeout=10)
except Exception as exc:
    report["status"] = "failed"
    report["error"] = str(exc)
    try:
        report["body_on_error"] = body()
        driver.save_screenshot(str(HERE / "ui_error.png"))
    except Exception:
        pass
    raise
finally:
    (HERE / "ui_browser_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "checks_count": len(checks), "checks": checks,
                      "error": report.get("error")}, ensure_ascii=True))
    driver.quit()
