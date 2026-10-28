"""Reusable engine validation; run python -m unittest discover -s tests -p test_engine.py."""
import io
import json
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from regression.config import AnalysisConfig,AnalysisError
from regression.loading import load_table,inspect_table
from regression.preprocessing import make_preprocessor
from regression.engine import analyze,make_pipeline,selected_design


def fixture(p=4,n=60):
    rng=np.random.default_rng(14)
    X=pd.DataFrame(rng.normal(size=(n,p)),columns=[f'x{i}' for i in range(p)])
    X['answer']=3*X.x0-2*X.x1+rng.normal(size=n)
    return X


class EngineTests(unittest.TestCase):
    def cfg(self,**kw):
        options=dict(target='answer',feature_counts=[1,2],inner_folds=2,outer_folds=2,max_seconds=180)
        options.update(kw)
        return AnalysisConfig(**options)

    def test_schema_and_csv(self):
        df=fixture();df['group']=['a','b']*30;df['constant']=1;df['record_id']=range(60)
        df.loc[0,'x0']=np.nan
        info=inspect_table(load_table(df))
        self.assertIn('group',info['categorical']);self.assertIn('constant',info['constants'])
        self.assertIn('record_id',info['identifiers']);self.assertEqual(info['missing']['x0'],1)
        self.assertEqual(len(info['preview']),10)
        self.assertEqual(load_table(df.to_csv(index=False).encode(),'data.csv').shape,df.shape)

    def test_xlsx(self):
        buf=io.BytesIO();fixture().to_excel(buf,index=False,engine='openpyxl')
        self.assertEqual(load_table(buf.getvalue(),'data.xlsx').shape,(60,5))

    def test_fold_preprocessing_and_unknown_category(self):
        train=pd.DataFrame({'n':[1.,2.,np.nan,4.],'cat':['a','a',None,'b']}).replace({None:np.nan})
        prep=make_preprocessor(train);prep.fit(train)
        self.assertEqual(float(prep.named_transformers_['num'].statistics_[0]),2.)
        original=prep.named_transformers_['num'].statistics_.copy()
        prep.transform(pd.DataFrame({'n':[100000.],'cat':['UNSEEN']}))
        np.testing.assert_equal(prep.named_transformers_['num'].statistics_,original)
        self.assertNotIn('UNSEEN',prep.named_transformers_['cat'].named_steps['encoder'].categories_[0])

    def test_mixed_missing_and_numeric_parity(self):
        df=fixture();df['town']=['north','south','west']*20
        df.loc[[1,4,8],'x0']=np.nan;df.loc[[3,9],'town']=np.nan
        result,artifacts=analyze(df,self.cfg(),filename='mixed.csv')
        self.assertEqual(len(result['models']),10);self.assertEqual(len(result['comparison']),6)
        self.assertEqual(result['predictionMode'],'encoded-linear')
        self.assertIn('town',[v['name'] for v in result['inputs']])
        X=df.drop(columns='answer');_,xt,_,yt=train_test_split(X,df.answer,test_size=.2,random_state=0)
        for m in result['models']:
            model=artifacts[(m['method'],m['k'])]
            pred=model.predict(xt)
            np.testing.assert_allclose(pred,[r['predicted'] for r in m['residuals']])
            self.assertAlmostEqual(float(np.sqrt(np.mean((yt-pred)**2))),m['rmse'])
            self.assertEqual(len(m['features']),len(m['weights']))
        json.dumps(result,allow_nan=False)

    def test_twenty_and_thirtynine_inputs(self):
        for count in [20,39]:
            result,_=analyze(fixture(count),self.cfg(feature_counts=[1]),filename='many.csv')
            self.assertEqual(result['metadata']['maxFeatures'],count)
            self.assertEqual(len(result['models']),5)

    def test_builtin_files(self):
        for path,target in [('50_Startups.csv','Profit'),('datasets/boston_housing/HousingData.csv','MEDV')]:
            cfg=self.cfg(target=target,feature_counts=[1])
            result,_=analyze(path,cfg)
            self.assertEqual(result['metadata']['rawSize'],50 if target=='Profit' else 506)
            self.assertEqual(result['metadata']['cleanSize'],50 if target=='Profit' else 506)
            self.assertEqual(len(result['comparison']),6)

    def test_cv_never_fits_test_rows(self):
        from unittest.mock import patch
        import regression.engine as engine
        df=fixture();train,test=train_test_split(df,test_size=.2,random_state=0)
        original=engine.make_pipeline;fitted_indices=[]
        def factory(X,method,k,cfg):
            pipeline=original(X,method,k,cfg);fit=pipeline.fit
            def recorded(X,y,*args,**kw):
                fitted_indices.append(set(X.index));return fit(X,y,*args,**kw)
            pipeline.fit=recorded
            return pipeline
        with patch.object(engine,'make_pipeline',side_effect=factory):
            result,_=analyze(df,self.cfg(feature_counts=[1]))
        self.assertTrue(fitted_indices)
        self.assertTrue(all(rows.isdisjoint(set(test.index)) for rows in fitted_indices))
        self.assertTrue(any(len(rows)<len(train) for rows in fitted_indices))

    def test_valid_perfect_relation_and_singular_inference(self):
        rng=np.random.default_rng(43);x=rng.normal(size=40)
        df=pd.DataFrame({'x':x,'duplicate_x':x,'answer':3*x+2})
        result,_=analyze(df,self.cfg(feature_counts=[1,2]))
        self.assertEqual(len(result['models']),10)
        singular=next(m for m in result['models'] if m['k']==2)
        self.assertFalse(singular['diagnostics']['fullRank'])
        self.assertIsNone(singular['fP'])
        json.dumps(result,allow_nan=False)

    def test_rejections_and_limits(self):
        with self.assertRaises(AnalysisError):analyze(fixture(n=10),self.cfg())
        with self.assertRaises(AnalysisError):analyze(fixture(n=20),self.cfg(test_size=.4))
        with self.assertRaises(AnalysisError):analyze(fixture(),self.cfg(seed=-1))
        with self.assertRaises(AnalysisError):analyze(fixture().assign(answer='not numeric'),self.cfg())
        with self.assertRaises(AnalysisError):analyze(fixture().assign(answer=1),self.cfg())
        with self.assertRaises(AnalysisError):analyze(fixture(20),self.cfg(max_encoded_features=5,max_comparison_features=5))
        with self.assertRaises(AnalysisError):analyze(fixture(),self.cfg(max_workload=1))
        with self.assertRaises(AnalysisError):analyze(fixture(),self.cfg(inner_folds=9))
        with self.assertRaises(AnalysisError):analyze(fixture(),self.cfg(features=['answer','x0']))

    def test_monotonic_measurement_is_not_an_identifier(self):
        rng=np.random.default_rng(2)
        df=pd.DataFrame({'year':np.arange(1980,2020),'answer':np.arange(40)*.5+rng.normal(size=40)})
        self.assertNotIn('year',inspect_table(df)['identifiers'])
        result,_=analyze(df,self.cfg(feature_counts=[1]))
        self.assertEqual(result['inputs'][0]['name'],'year')

    def test_id_and_leakage(self):
        df=fixture();df['record_id']=range(len(df));df['answer_copy']=df.answer
        result,_=analyze(df,self.cfg(feature_counts=[1]))
        raw=[v['name'] for v in result['inputs']]
        self.assertNotIn('record_id',raw);self.assertNotIn('answer_copy',raw)
        self.assertTrue(result['warnings'])


if __name__=='__main__':unittest.main()
