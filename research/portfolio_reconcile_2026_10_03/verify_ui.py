"""Desktop/mobile comparison QA with synthetic book and zero journal records."""
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
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--window-size=1440,1400")
driver = webdriver.Chrome(service=Service(r"C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe"), options=options)
wait = WebDriverWait(driver, 35)


def body():
    return driver.find_element(By.TAG_NAME, "body").text


def check(name, ok):
    assert ok, name
    checks.append(name)


def button(label):
    return wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='button' and normalize-space(.)='{label}']"))


report = {"checks": checks, "positions_are_synthetic": True, "accounting_actions_clicked": False}
try:
    for index, scenario in enumerate(("real", "pending", "stale")):
        driver.execute_cdp_cmd("Emulation.clearDeviceMetricsOverride", {})
        with tempfile.TemporaryDirectory(prefix="quant_reconcile_ui_") as directory:
            market = Path(directory) / "market.db"
            with gzip.open(ROOT / "data_seed/market.db.gz", "rb") as src, market.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            shutil.copyfile(ROOT / "data_seed/sox.csv", Path(directory) / "sox.csv")
            metadata = Path(directory) / "qa_meta.json"
            port = 8770 + index
            env = dict(os.environ, APP_DATA_DIR=directory, JOURNAL_DATABASE_URL="",
                       RECONCILE_QA_SCENARIO=scenario, RECONCILE_QA_PORT=str(port), RECONCILE_QA_META=str(metadata))
            with (HERE / f"{scenario}_server.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, str(HERE / "preview.py")], cwd=ROOT,
                                           env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    deadline = time.monotonic() + 30
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
                    tab = wait.until(lambda d: d.find_element(By.XPATH, "//*[@role='tab' and @aria-label='策略輪動']"))
                    tab.click()
                    wait.until(lambda d: tab.get_attribute("aria-selected") == "true")
                    time.sleep(1)
                    disabled = driver.execute_script("return !!arguments[0].closest('[aria-disabled=\"true\"]')", button("核對我的持倉"))
                    check(f"{scenario}_recheck_disabled_before_model", disabled)
                    button("計算輪動名單").click()
                    wait.until(lambda d: "輪動計算完成" in body())
                    meta = json.loads(metadata.read_text(encoding="utf-8"))
                    check(f"{scenario}_eight_stock_rule_visible", "模型最多 8 檔" in body() and "不是持有 16 檔" in body())
                    if scenario == "real":
                        check("real_model_with_synthetic_book_has_keep_and_exit", "應保留（帳本已有） · 1" in body() and "TEST_OUT" in body())
                        driver.save_screenshot(str(HERE / "ui_comparison_desktop.png"))
                        button("核對我的持倉").click()
                        wait.until(lambda d: "應保留（帳本已有） · 0" in body())
                        check("book_recheck_changes_comparison", "待補入（目前名單尚缺） · 7" in body())
                    elif scenario == "pending":
                        check("pending_target_separate_from_current", "下一次目標（尚待執行）" in body() and "目前名單尚未切換" in body())
                        check("early_purchase_not_listed_for_current_exit", "已提前持有下一次目標" in body() and "依模型應退出（目前名單外） · 1" in body())
                    else:
                        check("stale_book_view_is_historical", "歷史模型名單" in body() and "這份對照不可當成即時買賣清單" in body())
                        check("stale_view_has_no_live_exit_header", "依模型應退出（目前名單外） ·" not in body())
                    driver.execute_cdp_cmd("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 1900, "deviceScaleFactor": 1, "mobile": True})
                    time.sleep(1)
                    check(f"{scenario}_mobile_recheck_button_reachable", button("核對我的持倉").is_displayed())
                    driver.save_screenshot(str(HERE / f"ui_{scenario}_mobile.png"))
                    report[f"{scenario}_text"] = body()
                    with closing(sqlite3.connect(market)) as conn:
                        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                                  for t in ("trades", "cash_movements", "asset_history", "dca_plans", "corporate_actions")}
                    check(f"{scenario}_journal_records_zero", all(n == 0 for n in counts.values()))
                    report[f"{scenario}_journal_counts"] = counts
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
