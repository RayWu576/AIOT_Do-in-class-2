"""Browser integration and numeric parity checks; requires Playwright and Edge."""
import json
from pathlib import Path
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "dashboard/dist/data.json").read_text(encoding="utf-8"))
checks = json.loads((ROOT / "results/browser_prediction_checks.json").read_text(encoding="utf-8"))
assert len(data["models"]) == 25
assert len(data["comparison"]) == 5
for k in range(1, 6):
    rows = [m for m in data["models"] if m["k"] == k]
    assert len({tuple(m["features"]) for m in rows}) == 1, "Models must share inputs at each k"
for m in data["models"]:
    assert np.isclose(np.sqrt(np.mean([r["residual"] ** 2 for r in m["residuals"]])), m["rmse"])
    if m["method"] not in ("Random Forest", "Linear Regression"):
        assert all(c["p"] is None and c["se"] is None for c in m["coefficients"])
    if m["kind"] == "forest":
        assert len(m["trees"]) == 60 and np.isclose(sum(m["importances"]), 1)
qa = ROOT / "dashboard/qa"
qa.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge", headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 1100})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto((ROOT / "dashboard/dist/index.html").as_uri())
    page.wait_for_function("typeof window.startupPredict === 'function'")
    for check in checks:
        actual = page.evaluate("""c => {
          const m=window.DASHBOARD_DATA.models.find(m=>m.method===c.method&&m.k===c.k);
          return c.inputs.map(v=>window.startupPredict(m,v));
        }""", check)
        assert np.allclose(actual, check["expected"], rtol=1e-10, atol=1e-6), check
    for m in data["models"]:
        page.locator(f'#methods button[data-method="{m["method"]}"]').click()
        page.locator(f'#counts button[data-k="{m["k"]}"]').click()
        assert page.locator("#metrics .metric strong").first.inner_text() == f"${m['rmse']:,.2f}"
        assert page.locator("#residuals tr").count() == 10
        if m["kind"] == "forest":
            assert page.locator("#coefficients tr").count() == m["k"]
            assert "Random Forest" in page.locator("#equation").inner_text()
        else:
            assert page.locator("#coefficients tr").count() == m["k"] + 1
            if m["method"] != "Linear Regression":
                assert all(t == "—" for t in page.locator("#coefficients td:nth-child(5)").all_text_contents())
    page.locator('#methods button[data-method="Random Forest"]').click()
    page.locator('#counts button[data-k="5"]').click()
    page.locator("#number-0").fill("150000")
    page.locator("#state").select_option("Florida")
    page.locator("#save-scenario").click()
    assert page.locator("#scenarios .scenario").count() == 1
    with page.expect_download() as download:
        page.locator("#download").click()
    download.value.save_as(str(qa / "predictive-export.csv"))
    assert "Random Forest" in (qa / "predictive-export.csv").read_text(encoding="utf-8-sig")
    page.locator("#compare").uncheck()
    assert page.locator("#rmse-chart circle").count() == 5
    page.locator("#compare").check()
    page.locator("#sort").select_option("id")
    ids = [int(s.split("#")[1]) for s in page.locator("#residuals td:first-child").all_text_contents()]
    assert ids == sorted(ids)
    page.locator("#reset").click()
    page.screenshot(path=str(qa / "predictive-desktop.png"), full_page=True)
    page.locator("#theme").click()
    assert "dark" in page.locator("body").get_attribute("class")
    page.locator("#theme").click()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(qa / "predictive-mobile.png"), full_page=True)
    assert not errors, errors
    browser.close()
print("PASS: 25 models, shared inputs, 75 Python/browser predictions, real forest inference, model UI, statistics, CSV, scenarios, sorting, theme, mobile width; no JS errors")
