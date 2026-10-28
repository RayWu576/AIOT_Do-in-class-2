"""Browser upload E2E. Start uvicorn api:app on port 8000 before running."""
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
BASE=os.environ.get('REGRESSION_TEST_URL','http://127.0.0.1:8000')
qa=ROOT/'dashboard/qa';qa.mkdir(exist_ok=True)
rng=np.random.default_rng(71);n=60
label='floor area <img src=x onerror="window.__injected=1">'
category='zzz<img src=x onerror="window.__injected=1">'
frame=pd.DataFrame({label:rng.normal(size=n),'rooms':rng.normal(size=n),'distance':rng.uniform(0,5,n),
                    'region':['north','west',category]*20,'record_id':range(n),'constant':1})
frame['score']=5*frame[label]-2*frame.rooms+rng.normal(size=n)
frame.loc[[1,5,8],label]=np.nan;frame.loc[[4,7],'region']=np.nan
with tempfile.TemporaryDirectory() as directory,sync_playwright() as p:
    csv=Path(directory)/'custom.csv';frame.to_csv(csv,index=False)
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000});errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    assert page.goto(BASE,wait_until='networkidle').status==200
    page.locator('#upload-section summary').first.click()
    page.locator('#upload-file').set_input_files(csv)
    page.wait_for_function("!document.getElementById('upload-config').hidden",timeout=30000)
    assert page.locator('#upload-preview tbody tr').count()==10
    assert page.locator('#upload-preview img').count()==0
    page.select_option('#upload-target','score')
    assert not page.locator('#upload-features input').nth(4).is_checked()
    page.locator('.advanced-settings > summary').click()
    page.select_option('#upload-inner','2');page.select_option('#upload-outer','2')
    page.locator('#upload-counts').fill('1,2');page.locator('#upload-max-k').fill('2')
    page.locator('#upload-unit').fill('u<img src=x onerror=window.unitX=1>')
    page.locator('#analyze-upload').click()
    page.wait_for_function("document.getElementById('analysis-status').textContent.includes('分析完成') || (document.getElementById('upload-fields').disabled===false && document.getElementById('analysis-status').textContent.includes('failed'))",timeout=210000)
    assert '分析完成' in page.locator('#analysis-status').inner_text()
    assert page.locator('#dataset-select option').count()==3
    assert page.locator('#counts button').count()==2
    assert page.locator('#rmse-chart circle').count()==10
    assert not page.locator('#state').is_visible()
    assert page.locator('#inputs select[data-category]').count()==1
    assert page.locator('#metrics img').count()==0
    assert page.locator('#scatter img').count()==0
    assert not page.evaluate('Boolean(window.unitX)')
    assert page.locator('#coefficients img').count()==0
    assert page.locator('#equation img').count()==0
    assert not page.evaluate('Boolean(window.__injected)')
    page.locator('#methods button[data-method="Correlation"]').click();page.locator('#counts button[data-k="2"]').click()
    current=page.evaluate('window.regressionDashboard.current()')
    idx=next(i for i,col in enumerate(current['dataset']['inputs']) if col['name']==label)
    page.locator(f'#number-{idx}').fill('-2.5')
    catidx=next(i for i,col in enumerate(current['dataset']['inputs']) if col['name']=='region')
    page.select_option(f'#number-{catidx}',category)
    current=page.evaluate('window.regressionDashboard.current()')
    local=page.evaluate("c=>{const m=c.dataset.models.find(m=>m.method===c.method&&m.k===c.k);return window.regressionPredict(c.dataset,m,c.values)}",current)
    response=page.request.post(BASE+'/api/analyses/'+current['dataset']['jobId']+'/predict',data={'method':current['method'],'k':current['k'],'values':current['values']})
    assert response.ok,response.text()
    assert abs(response.json()['prediction']-local)<1e-6
    assert page.locator('#assumptions').is_visible()
    assert page.locator('#linearity-chart circle').count()>0
    assert page.locator('#qq-chart circle').count()>0
    with page.expect_download() as export:page.locator('#export-analysis').click()
    export.value.save_as(str(qa/'uploaded-analysis.json'))
    assert json.loads((qa/'uploaded-analysis.json').read_text(encoding='utf-8'))['metadata']['target']=='score'
    page.evaluate('window.scrollTo(0,0)');page.screenshot(path=str(qa/'upload-desktop.png'),full_page=True)
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    page.screenshot(path=str(qa/'upload-mobile.png'),full_page=True)
    page.select_option('#dataset-select','boston');assert page.locator('#counts button').count()==13
    page.select_option('#dataset-select','startups');assert page.locator('#state').is_visible()
    assert not errors,errors
    browser.close()
print('PASS: actual CSV upload, preview, arbitrary columns/categories, async analysis, negative values, browser/API prediction parity, XSS-safe labels, diagnostics, export, static fallback and mobile.')
