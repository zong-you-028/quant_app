"""Local offline preview only; never clicks refresh or accounting actions."""
import json
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


HERE = Path(__file__).resolve().parent
options = webdriver.ChromeOptions()
options.add_argument('--headless=new')
options.add_argument('--window-size=1440,1300')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(service=Service(
    r'C:\Users\USER\.cache\selenium\chromedriver\win64\153.0.8010.52\chromedriver.exe'), options=options)
wait = WebDriverWait(driver, 45)
result = {'checks': [], 'data_isolated': True, 'refresh_clicked': False,
          'accounting_actions_clicked': False}


def body():
    return driver.find_element(By.TAG_NAME, 'body').text


def check(name, condition):
    assert condition, name
    result['checks'].append(name)


def button(label):
    return wait.until(lambda d: d.find_element(By.XPATH, f"//*[@role='button' and normalize-space(.)='{label}']"))


def guide():
    return wait.until(lambda d: d.find_element(By.XPATH, "//*[@role='button' and normalize-space(.)='使用流程']"))


def tab(label):
    node = driver.find_element(By.XPATH, f"//*[@role='tab' and @aria-label='{label}']")
    node.click()
    wait.until(lambda d: node.get_attribute('aria-selected') == 'true')
    # Flutter keeps the departing panel's semantics during its animation.
    time.sleep(1)


try:
    driver.get('http://127.0.0.1:8766')
    wait.until(lambda d: d.execute_script("return !!document.querySelector('flt-semantics-placeholder')"))
    driver.execute_script("document.querySelector('flt-semantics-placeholder').click()")
    wait.until(lambda d: d.find_elements(By.XPATH, "//*[@role='tab' and @aria-label='策略輪動']"))
    check('one_model_header', '低換手多視窗模型' in body())
    check('overfitting_status_visible', '過擬合未排除' in body())
    check('no_legacy_model_choices', all(x not in body() for x in ('基準｜60 日動能', '趨勢穩定模型')))
    driver.save_screenshot(str(HERE / 'ui_desktop_initial.png'))

    tab('策略輪動')
    wait.until(lambda d: '使用流程' in body())
    # Wait for the semantic button; text appears before the tab animation ends.
    guide().click()
    wait.until(lambda d: '不扣現金' in body())
    check('usage_guide_opens', '實際成交後' in body() and '不會自動下單' in body())
    driver.save_screenshot(str(HERE / 'ui_usage_desktop.png'))
    guide().click()
    button('計算輪動名單').click()
    wait.until(lambda d: '排名分' in body() and '最多 8 檔' in body())
    check('rotation_computes_single_model', '低換手多視窗模型' in body() and '上期模型名單可保留至前 16 名' in body())
    check('model_status_card_is_not_duplicated', body().count('驗證狀態：') == 1)
    check('stale_result_marked', '名單僅供歷史檢視' in body())
    guarded = driver.find_elements(By.XPATH, "//*[@role='button' and (contains(normalize-space(.),'加入庫存') or contains(normalize-space(.),'續抱'))]")
    check('stale_recommendation_buttons_exist', len(guarded) > 0)
    for item in guarded:
        disabled = driver.execute_script("return !!arguments[0].closest('[aria-disabled=\"true\"]')", item)
        check('stale_recommendation_disabled', disabled)
    driver.save_screenshot(str(HERE / 'ui_rotation_desktop.png'))
    result['rotation_text'] = body()

    driver.execute_cdp_cmd('Emulation.setDeviceMetricsOverride', {
        'width': 390, 'height': 1500, 'deviceScaleFactor': 1, 'mobile': True})
    wait.until(lambda d: d.execute_script('return window.innerWidth') == 390)
    time.sleep(1)
    driver.save_screenshot(str(HERE / 'ui_rotation_mobile.png'))
    check('mobile_tabs_reachable', driver.find_element(By.XPATH, "//*[@role='tab' and @aria-label='個股分析']").is_displayed())
    tab('個股分析')
    wait.until(lambda d: d.find_elements(By.XPATH, "//*[@role='button' and normalize-space(.)='分析這檔']"))
    button('分析這檔').click()
    wait.until(lambda d: '排名分' in body() and '尚未分析' not in body())
    time.sleep(1)
    check('stock_computes_single_model', '低換手多視窗模型' in body())
    check('stock_stale_status_visible', '名單僅供歷史檢視' in body())
    driver.save_screenshot(str(HERE / 'ui_stock_mobile.png'))
    result['stock_text'] = body()
    result['status'] = 'passed'
except Exception as exc:
    result['status'] = 'failed'
    result['error'] = str(exc)
    result['body_on_error'] = body()
    result['roles_on_error'] = driver.execute_script("return [...document.querySelectorAll('[role]')].map(x=>({role:x.getAttribute('role'),label:x.getAttribute('aria-label'),text:x.textContent.slice(0,120)}))")
    driver.save_screenshot(str(HERE / 'ui_error.png'))
    raise
finally:
    (HERE / 'ui_browser_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': result.get('status'), 'checks': result['checks'], 'error': result.get('error')}, ensure_ascii=True))
    driver.quit()
