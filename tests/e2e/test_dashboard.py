"""
Playwright E2E 테스트 — DORA 메트릭 대시보드

시나리오:
1. dashboard.html 로드 → 제목 확인
2. 핵심 지표 4개(Lead Time, Deployment Frequency, Change Failure Rate, MTTR) 존재 확인
3. 차트 요소 렌더링 확인
4. 실패 시 스크린샷 자동 저장 (conftest.py 훅)
"""

import os
import pathlib
import pytest
from playwright.sync_api import Page, expect


DASHBOARD_PATH = (
    pathlib.Path(__file__).parent.parent.parent / "metrics" / "out" / "dashboard.html"
)


@pytest.fixture(scope="session")
def dashboard_url():
    if not DASHBOARD_PATH.exists():
        pytest.skip("dashboard.html not found — run compute_dora.py first")
    return DASHBOARD_PATH.as_uri()


def test_dashboard_loads(page: Page, dashboard_url: str):
    page.goto(dashboard_url)
    expect(page).to_have_title(lambda t: len(t) > 0)


def test_dora_metrics_present(page: Page, dashboard_url: str):
    page.goto(dashboard_url)
    body = page.locator("body")
    for metric in ["Lead Time", "Deployment Frequency", "Change Failure Rate", "MTTR"]:
        expect(body).to_contain_text(metric)


def test_chart_elements_rendered(page: Page, dashboard_url: str):
    page.goto(dashboard_url)
    # SVG 또는 canvas 요소 (차트 라이브러리 공통)
    charts = page.locator("svg, canvas")
    assert charts.count() > 0, "차트 요소가 없습니다"


def test_no_js_errors(page: Page, dashboard_url: str):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(dashboard_url)
    page.wait_for_timeout(500)
    assert errors == [], f"JavaScript 에러 발생: {errors}"
