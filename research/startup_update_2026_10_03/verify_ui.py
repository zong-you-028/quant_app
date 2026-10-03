"""Browser checks for real fresh-cache startup and simulated partial/failure UI."""
import gzip
from contextlib import closing
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
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1440,1300")
driver = webdriver.Chrome(service=Service(r"C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe"), options=options)
wait = WebDriverWait(driver, 30)


def body():
    return driver.find_element(By.TAG_NAME, "body").text


def check(name, ok):
    assert ok, name
    checks.append(name)


def button(label):
    return wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='button' and normalize-space(.)='{label}']"))


report = {"checks": checks, "journal_isolated": True, "external_downloads_forbidden": True}
try:
    for index, scenario in enumerate(("fresh", "partial")):
        with tempfile.TemporaryDirectory(prefix="quant_update_ui_") as directory:
            market = Path(directory) / "market.db"
            with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as src, market.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copyfile(ROOT / "data_seed/sox.csv", Path(directory) / "sox.csv")
            port = 8767 + index
            env = dict(os.environ, APP_DATA_DIR=directory, JOURNAL_DATABASE_URL="",
                       UPDATE_QA_SCENARIO=scenario, UPDATE_QA_PORT=str(port))
            with (HERE / f"{scenario}_server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(HERE / "preview.py")], cwd=ROOT,
                                           env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    deadline = time.monotonic() + 25
                    while True:
                        try:
                            if requests.get(f"http://127.0.0.1:{port}", timeout=1).status_code == 200:
                                break
                        except requests.RequestException:
                            pass
                        assert process.poll() is None and time.monotonic() < deadline, "preview startup"
                        time.sleep(.5)
                    driver.get(f"http://127.0.0.1:{port}")
                    wait.until(lambda d: d.execute_script("return !!document.querySelector('flt-semantics-placeholder')"))
                    driver.execute_script("document.querySelector('flt-semantics-placeholder').click()")
                    if scenario == "fresh":
                        wait.until(lambda d: "資料更新完成；" in body())
                        check("fresh_seed_startup_success_without_external_download", "已達目標 51" in body())
                        driver.save_screenshot(str(HERE / "ui_fresh_desktop.png"))
                        report["fresh_result"] = body()
                    else:
                        wait.until(lambda d: "2330 更新行情" in body())
                        check("shared_progress_visible_on_stock_tab", "已檢查 0/51 檔" in body() and "已用" in body())
                        check("analysis_disabled_during_update", button("分析這檔").get_attribute("aria-disabled") == "true" or driver.execute_script("return !!arguments[0].closest('[aria-disabled=\"true\"]')", button("分析這檔")))
                        driver.save_screenshot(str(HERE / "ui_progress_desktop.png"))
                        wait.until(lambda d: "待續抓 50" in body())
                        tab = driver.find_element(By.XPATH, "//*[@role='tab' and @aria-label='策略輪動']")
                        tab.click()
                        wait.until(lambda d: tab.get_attribute("aria-selected") == "true")
                        time.sleep(1)
                        button("繼續更新未完成資料").click()
                        wait.until(lambda d: "測試來源逾時" in body())
                        check("source_failure_visible", "失敗 1" in body())
                        driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 1350, "deviceScaleFactor": 1, "mobile": True})
                        time.sleep(1)
                        check("mobile_retry_reachable", button("重試未完成資料").is_displayed())
                        driver.save_screenshot(str(HERE / "ui_retry_mobile.png"))
                        button("重試未完成資料").click()
                        wait.until(lambda d: "資料更新完成；" in body())
                        check("retry_restores_success", "失敗 0" in body() and "待續抓 0" in body())
                        driver.save_screenshot(str(HERE / "ui_success_mobile.png"))
                    with closing(sqlite3.connect(market)) as conn:
                        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                                  for t in ("trades", "cash_movements", "asset_history", "dca_plans", "corporate_actions")}
                    check(f"{scenario}_journal_rows_remain_zero", all(n == 0 for n in counts.values()))
                    report[f"{scenario}_journal_rows"] = counts
                finally:
                    process.terminate()
                    process.wait(timeout=10)
    report["status"] = "passed"
except Exception as exc:
    report["status"] = "failed"
    report["error"] = str(exc)
    report["body_on_error"] = body()
    driver.save_screenshot(str(HERE / "ui_error.png"))
    raise
finally:
    (HERE / "ui_browser_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report.get("status"), "checks": checks, "error": report.get("error")}, ensure_ascii=True))
    driver.quit()
