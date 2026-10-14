"""Numeric and browser integration checks for the second regression problem."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from playwright.sync_api import sync_playwright
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from housing_experiment import load_housing
from dashboard_export import refresh_asset_versions

bundle=json.loads((ROOT/'dashboard/dist/data.json').read_text(encoding='utf-8'))
assert set(bundle['datasets'])=={'startups','boston'}
D=bundle['datasets']['boston']
raw=pd.read_csv(ROOT/'datasets/boston_housing/HousingData.csv')
assert raw.shape==(506,14) and int(raw.isna().sum().sum())==120
X_train,X_test,y_train,y_test=load_housing()
assert (len(X_train),len(X_test))==(315,79)
assert D['metadata']['cleanSize']==394 and D['metadata']['unit']=='千美元'
assert len(D['models'])==65 and len(D['inputs'])==13
assert len({m['method'] for m in D['models']})==5
for m in D['models']:
    assert len(m['features'])==m['k'] and len(m['residuals'])==79
    reg=LinearRegression().fit(X_train[m['features']],y_train)
    assert np.allclose(reg.coef_,m['weights'])
    assert np.isclose(reg.intercept_,m['intercept'])
    assert np.allclose(reg.predict(X_test[m['features']]),[r['predicted'] for r in m['residuals']])
    assert np.isclose(np.sqrt(np.mean([r['residual']**2 for r in m['residuals']])),m['rmse'])
with sync_playwright() as p:
    b=p.chromium.launch(channel='msedge',headless=True)
    page=b.new_page(viewport={'width':1440,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'dashboard/dist/index.html').as_uri())
    assert page.locator('#dataset-select option').count()==2
    assert page.locator('#counts button').count()==5
    page.select_option('#dataset-select','boston')
    assert page.locator('#counts button').count()==13
    assert page.locator('#inputs input[type=number]').count()==13
    assert not page.locator('#state').is_visible()
    for m in D['models']:
        page.locator(f'#methods button[data-method="{m["method"]}"]').click()
        page.locator(f'#counts button[data-k="{m["k"]}"]').click()
        assert page.locator('#metrics .metric strong').first.inner_text()==f"{m['rmse']:,.2f} 千美元"
        expected=m['intercept']+sum(w*float(X_train[f].median()) for w,f in zip(m['weights'],m['features']))
        assert page.locator('#profit').inner_text()==f'{expected:,.2f} 千美元'
        assert page.locator('#coefficients tr').count()==m['k']+1
    assert page.locator('#rmse-chart circle').count()==65
    page.locator('#sort').select_option('id')
    assert page.locator('#residuals tr').count()==20
    first=page.locator('#residuals td:first-child').first.inner_text()
    page.locator('#next-page').click()
    assert page.locator('#residuals td:first-child').first.inner_text()!=first
    page.locator('#next-page').click();page.locator('#next-page').click()
    assert page.locator('#residuals tr').count()==19
    assert page.locator('#next-page').is_disabled()
    rm_index=next(i for i,f in enumerate(D['inputs']) if f['name']=='RM')
    old=page.locator('#profit').inner_text()
    page.locator(f'#number-{rm_index}').fill('8')
    assert page.locator('#profit').inner_text()!=old
    chas_index=next(i for i,f in enumerate(D['inputs']) if f['name']=='CHAS')
    page.locator(f'#number-{chas_index}').fill('2');page.locator('#save-scenario').click()
    assert page.locator('#scenarios .scenario').count()==1
    with page.expect_download() as dl:page.locator('#download').click()
    assert dl.value.suggested_filename.startswith('boston-')
    page.screenshot(path=str(ROOT/'dashboard/qa/boston-desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(ROOT/'dashboard/qa/boston-mobile.png'),full_page=True)
    page.select_option('#dataset-select','startups')
    assert page.locator('#counts button').count()==5 and page.locator('#state').is_visible()
    assert page.locator('#scenarios .scenario').count()==0
    assert '$' in page.locator('#profit').inner_text()
    assert 'Startup' in page.locator('#residuals td:first-child').first.inner_text()
    assert page.locator('#residuals tr').count()==10
    assert not errors,errors
    b.close()
print('PASS: source/cleaning, 65 Boston models and predictions, dataset switching, units, inputs, CSV, residual pagination and mobile; no JS errors')
