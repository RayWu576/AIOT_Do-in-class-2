"""FastAPI integration tests with a real isolated analysis process."""
import io
import time
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api import app


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app);cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):cls.client.__exit__(None,None,None)

    def fixture(self):
        rng=np.random.default_rng(8);a=rng.normal(size=48);b=rng.normal(size=48)
        return pd.DataFrame({'area':a,'rooms':b,'region':['a','b','c']*16,'price':4*a-b+rng.normal(size=48)})

    def test_health_preview_and_formats(self):
        self.assertEqual(self.client.get('/api/health').status_code,200)
        self.assertEqual(self.client.get('/').status_code,200)
        bad=self.client.post('/api/uploads',files={'file':('bad.txt',b'hello')})
        self.assertEqual(bad.status_code,422)
        buf=io.BytesIO();self.fixture().to_excel(buf,index=False)
        uploaded=self.client.post('/api/uploads',files={'file':('sample.xlsx',buf.getvalue())})
        self.assertEqual(uploaded.status_code,200)
        body=uploaded.json();self.assertEqual(len(body['overview']['preview']),10)
        self.assertIn('region',body['overview']['categorical'])
        self.assertEqual(self.client.delete('/api/uploads/'+body['session_id']).status_code,200)

    def test_end_to_end_analysis_predict_export(self):
        frame=self.fixture();frame.loc[1,'area']=np.nan
        upload=self.client.post('/api/uploads',files={'file':('data.csv',frame.to_csv(index=False).encode())}).json()
        req={'session_id':upload['session_id'],'config':{'target':'price','features':['area','rooms','region'],
            'inner_folds':2,'outer_folds':2,'feature_counts':[1,2],'max_seconds':180,'title':'Custom test'}}
        response=self.client.post('/api/analyses',json=req)
        self.assertEqual(response.status_code,202,response.text);job=response.json()['job_id']
        deadline=time.monotonic()+210
        while time.monotonic()<deadline:
            body=self.client.get('/api/analyses/'+job).json()
            if body['status'] in ('completed','failed'):break
            time.sleep(.2)
        self.assertEqual(body['status'],'completed',body)
        result=body['result'];m=result['models'][0]
        raw={'area':None,'rooms':1.2,'region':'NEW'}
        encoded={}
        for e in result['encoding']:
            v=raw[e['column']]
            encoded[e['name']]=float(v if v is not None else e['fill']) if e['kind']=='numeric' else float(str(v if v is not None else e['fill'])==e['category'])
        expected=m['intercept']+sum(w*encoded[f] for w,f in zip(m['weights'],m['features']))
        prediction=self.client.post('/api/analyses/'+job+'/predict',json={'method':m['method'],'k':m['k'],'values':raw})
        self.assertEqual(prediction.status_code,200,prediction.text)
        self.assertAlmostEqual(prediction.json()['prediction'],expected)
        self.assertEqual(self.client.get('/api/analyses/'+job+'/export').status_code,200)
        plot=self.client.get(f'/api/analyses/{job}/plots/{m["method"]}/{m["k"]}/qq.png')
        self.assertEqual(plot.status_code,200);self.assertTrue(plot.content.startswith(b'\x89PNG'))
        invalid=self.client.post('/api/analyses/'+job+'/predict',json={'method':m['method'],'k':m['k'],'values':{'bogus':1}})
        self.assertEqual(invalid.status_code,422)
        self.assertEqual(self.client.delete('/api/uploads/'+upload['session_id']).status_code,200)
        self.assertEqual(self.client.get('/api/analyses/'+job).status_code,404)

    def test_invalid_config_and_missing_session(self):
        self.assertEqual(self.client.get('/api/analyses/not-real').status_code,404)
        u=self.client.post('/api/uploads',files={'file':('data.csv',self.fixture().to_csv(index=False).encode())}).json()
        response=self.client.post('/api/analyses',json={'session_id':u['session_id'],'config':{'target':'price','inner_folds':99}})
        self.assertEqual(response.status_code,422)
        self.client.delete('/api/uploads/'+u['session_id'])


if __name__=='__main__':unittest.main()
