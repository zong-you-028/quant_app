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
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.wheel_input import ScrollOrigin
from selenium.common.exceptions import StaleElementReferenceException

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
checks = []
report = {"checks": checks, "accounting_actions_clicked": False,
          "actual_startup_and_asset_history": True, "positions_are_synthetic_test_data": True,
          "browser_frontend_assets_allowed": True,
          "external_network_counter_scope": "Python app/data pipeline; browser Flet frontend assets excluded"}
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1440,1500")
service = Service(r"C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe")
service.creation_flags = subprocess.CREATE_NO_WINDOW
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 60, ignored_exceptions=(StaleElementReferenceException,))


def body():
    return driver.find_element(By.TAG_NAME, "body").text


def check(name, result):
    assert result, name
    checks.append(name)


def button(label):
    return wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='button' and normalize-space(.)='{label}']"))


def tab(label):
    locator = f"//*[@role='tab' and @aria-label='{label}']"
    def select(d):
        item = d.find_element(By.XPATH, locator)
        if item.get_attribute("aria-selected") == "true":
            return True
        item.click()
        return False
    wait.until(select)
    time.sleep(.8)
    return driver.find_element(By.XPATH, locator)


try:
    with tempfile.TemporaryDirectory(prefix="quant_asset_ui_") as directory:
        market = Path(directory) / "market.db"
        with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as source, market.open("wb") as target:
            shutil.copyfileobj(source, target)
        shutil.copyfile(ROOT / "data_seed/sox.csv", Path(directory) / "sox.csv")
        metadata = Path(directory) / "ui_meta.json"
        env = dict(os.environ, APP_DATA_DIR=directory, JOURNAL_DATABASE_URL="", APP_PASSWORD="",
                   UNIVERSE_UI_PORT="8785", UNIVERSE_UI_META=str(metadata))
        with (HERE / "ui_server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(HERE / "preview.py")], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 35
                while True:
                    try:
                        if requests.get("http://127.0.0.1:8785", timeout=1).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    assert process.poll() is None and time.monotonic() < deadline, "preview startup"
                    time.sleep(.5)
                driver.get("http://127.0.0.1:8785")
                wait.until(lambda d: d.execute_script("return !!document.querySelector('flt-semantics-placeholder')"))
                driver.execute_script("document.querySelector('flt-semantics-placeholder').click()")
                tab("投資紀錄")
                wait.until(lambda d: "總資產 3,100" in body())
                check("held_quote_refreshes_assets_before_batch_finishes", "已檢查" in body() and "資料更新完成；" not in body())
                check("auto_history_available_without_manual_snapshot", "展開總資產紀錄（2 筆）" in body())
                check("first_saved_quote_status_visible", "已自動更新 2026-10-08" in body())
                driver.save_screenshot(str(HERE / "ui_assets_partial_desktop.png"))
                report["partial_desktop_text"] = body()
                wait.until(lambda d: "總資產 3,200" in body() and "資料更新完成；" in body())
                check("second_held_quote_updates_latest_total", "總資產 3,200" in body())
                button("展開總資產紀錄（2 筆）").click()
                wait.until(lambda d: "每日自動 · 持股行情已達目標" in body())
                check("history_has_actual_quote_date", "2026-10-02" in body())
                check("previous_day_snapshot_preserved", "總資產 3,000" in body())
                driver.save_screenshot(str(HERE / "ui_assets_desktop.png"))
                report["final_desktop_text"] = body()
                tab("策略輪動")
                button("更新每日資料").click()
                wait.until(lambda d: "本日已保存 2 檔" in body() and "資料更新完成；" in body())
                tab("投資紀錄")
                check("same_day_repeat_keeps_two_days_only", "收合總資產紀錄（2 筆）" in body())
                check("same_day_repeat_keeps_correct_total", "總資產 3,200" in body())
                driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width":390,"height":1500,"deviceScaleFactor":1,"mobile":True})
                time.sleep(2)
                geometry = driver.execute_script("""
                  return {width: innerWidth, documentWidth: document.documentElement.scrollWidth,
                    visibleBeyond:[...document.querySelectorAll('*')].filter(e=>{
                      const r=e.getBoundingClientRect(),s=getComputedStyle(e);
                      return r.right>innerWidth+1&&r.width>0&&s.visibility!=='hidden'&&s.display!=='none'&&s.opacity!=='0';
                    }).map(e=>({tag:e.tagName,text:(e.innerText||'').slice(0,80)}))};
                """)
                report["mobile_geometry"] = geometry
                check("mobile_history_and_chart_have_no_horizontal_overflow", geometry["documentWidth"]<=391 and not geometry["visibleBeyond"])
                check("mobile_assets_and_daily_record_visible", "總資產 3,200" in body() and "已自動更新 2026-10-08" in body())
                # Flet uses its own scrollable viewport. Scroll the actual
                # investment tab until the chart/history controls enter view.
                for _ in range(5):
                    rect = driver.execute_script("return arguments[0].getBoundingClientRect().toJSON()", button("收合總資產紀錄（2 筆）"))
                    if 500 < rect["top"] < 1400:
                        break
                    ActionChains(driver).scroll_from_origin(ScrollOrigin.from_viewport(200,1200),0,550).perform()
                    time.sleep(.4)
                check("mobile_chart_and_history_controls_on_screen", 500 < rect["top"] < 1400)
                driver.save_screenshot(str(HERE / "ui_assets_mobile.png"))
                report["mobile_text"] = body()
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                report["preview"] = meta
                check("two_synthetic_quotes_downloaded_once", meta["source_calls"] == ["2330", "2317"])
                check("same_day_cache_avoids_repeated_download", meta["update_runs"][-1]["daily_skipped_symbols"] == ["2317", "2330"])
                check("formal_database_and_server_external_network_zero", all(meta["isolation"][key] == 0 for key in
                      ("formal_database_attempts", "outside_database_attempts", "external_network_attempts")))
                with closing(sqlite3.connect(market)) as conn:
                    rows = conn.execute("SELECT auto_day,total_assets,price_status FROM asset_history ORDER BY auto_day").fetchall()
                    trades = conn.execute("SELECT symbol,shares FROM trades ORDER BY symbol").fetchall()
                report["disposable_history_rows"] = rows
                check("daily_history_exact_two_rows", rows == [("2026-10-07",3000.,"latest"),("2026-10-08",3200.,"latest")])
                check("held_quote_intermediate_asset_value_committed", any(s["history"][0]["total_assets"]==3100. for s in meta["asset_saves"]))
                check("asset_refresh_does_not_change_positions", trades == [("2317",20.),("2330",10.)])
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
