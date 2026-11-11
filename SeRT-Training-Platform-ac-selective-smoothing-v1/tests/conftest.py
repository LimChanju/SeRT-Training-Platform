"""실패한 Playwright 테스트에서 스크린샷 자동 저장"""
import os
import pytest


SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "..", "test-artifacts", "screenshots")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        page = item.funcargs.get("page")
        if page is not None:
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            name = item.nodeid.replace("/", "_").replace("::", "_").replace(" ", "_")
            path = os.path.join(SCREENSHOT_DIR, f"{name}.png")
            try:
                page.screenshot(path=path, full_page=True)
                print(f"\n[E2E] 실패 스크린샷 저장: {path}")
            except Exception as e:
                print(f"\n[E2E] 스크린샷 저장 실패: {e}")
