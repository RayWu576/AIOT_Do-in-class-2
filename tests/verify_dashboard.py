"""Verify the requested five feature-selection methods and browser results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'dashboard/dist/data.json').read_text(encoding='utf-8'))
expected={'Correlation','Mutual Information','RFE','RFECV','LASSO'}
assert {m['method'] for m in data['models']}==expected
assert len(data['models'])==25
assert set(pd.read_csv(ROOT/'results/model_comparison.csv')['Model'])==expected|{'Baseline'}
df=pd.read_csv(ROOT/'50_Startups.csv')
df.columns=df.columns.str.strip()
df=df.drop_duplicates().dropna()
df['State']=pd.Categorical(df['State'],categories=['California','Florida','New York'])
X=pd.get_dummies(df.drop(columns='Profit'),columns=['State'],drop_first=True,dtype=float)
X_train,X_test,y_train,y_test=train_test_split(X,df['Profit'],test_size=.2,random_state=0)
for m in data['models']:
    assert len(m['features'])==m['k']
    reg=LinearRegression().fit(X_train[m['features']],y_train)
    assert np.allclose(reg.coef_,m['weights'])
    pred=reg.predict(X_test[m['features']])
    assert np.allclose(pred,[r['predicted'] for r in m['residuals']])
    assert np.isclose(np.sqrt(np.mean((y_test-pred)**2)),m['rmse'])
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((ROOT/'dashboard/dist/index.html').as_uri())
    for m in data['models']:
        page.locator(f'#methods button[data-method="{m["method"]}"]').click()
        page.locator(f'#counts button[data-k="{m["k"]}"]').click()
        assert page.locator('#metrics .metric strong').first.inner_text()==f"${m['rmse']:,.2f}"
        assert page.locator('#coefficients tr').count()==m['k']+1
        values={'R&D Spend':100000,'Marketing Spend':150000,'Administration':85000,'State_Florida':0,'State_New York':0}
        predicted=m['intercept']+sum(w*values[f] for w,f in zip(m['weights'],m['features']))
        assert page.locator('#profit').inner_text()==f'${predicted:,.2f}'
    page.locator('#reset').click()
    page.locator('#number-0').fill('160000')
    page.locator('#save-scenario').click()
    assert page.locator('#scenarios .scenario').count()==1
    with page.expect_download() as dl:page.locator('#download').click()
    assert dl.value.suggested_filename.endswith('.csv')
    page.locator('#compare').uncheck()
    assert page.locator('#rmse-chart circle').count()==5
    page.locator('#theme').click()
    assert 'dark' in page.locator('body').get_attribute('class')
    page.set_viewport_size({'width':390,'height':844})
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
    assert not errors,errors
    browser.close()
print('PASS: five requested methods, 25 fitted subsets, Python metrics and browser predictions, CSV, scenarios, comparison, dark mode, and mobile width.')
