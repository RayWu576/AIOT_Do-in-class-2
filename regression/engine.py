"""Reusable, bounded CRISP-DM engine. Every CV candidate fits raw training rows."""
import time
import math
import warnings
from datetime import datetime,timezone
from dataclasses import asdict
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold,train_test_split
from feature_selection import FixedCountSelector
from .config import AnalysisConfig,AnalysisError
from .loading import load_table,inspect_table,identifier_columns
from .preprocessing import make_preprocessor,encoded_metadata
from .evaluation import metrics,inference,finite

METHODS=['Correlation','Mutual Information','RFE','RFECV','LASSO']


class BoundedSelector(FixedCountSelector):
    def fit(self,X,y):
        requested=self.k
        self.k=min(requested,X.shape[1])
        try:return super().fit(X,y)
        finally:self.k=requested


def make_pipeline(X,method,k,cfg):
    steps=[('preprocessor',make_preprocessor(X)),('variance',VarianceThreshold())]
    if method!='Baseline':
        steps.append(('selector',BoundedSelector(method,k,cfg.inner_folds,cfg.seed,True)))
    steps.append(('regressor',LinearRegression()))
    return Pipeline(steps)


def selected_design(model,X):
    prep=model.named_steps['preprocessor'];variance=model.named_steps['variance']
    values=variance.transform(prep.transform(X))
    names=np.asarray(prep.get_feature_names_out())[variance.get_support()]
    if 'selector' in model.named_steps:
        selector=model.named_steps['selector'];values=selector.transform(values);names=names[selector.get_support()]
    return values,names.tolist()


def candidate_counts(p,cfg):
    maximum=min(p,cfg.max_comparison_features)
    if cfg.feature_counts is not None:
        if max(cfg.feature_counts)>maximum:raise AnalysisError(f'Feature counts exceed comparison limit {maximum}.')
        return sorted(set(cfg.feature_counts))
    if maximum<=cfg.max_comparisons:return list(range(1,maximum+1))
    return sorted(set(np.linspace(1,maximum,cfg.max_comparisons).round().astype(int).tolist()))


def analyze(source,config,filename=None):
    cfg=(AnalysisConfig(**config) if isinstance(config,dict) else config).validate()
    started=time.monotonic()
    def check():
        if time.monotonic()-started>cfg.max_seconds:raise AnalysisError('Analysis time limit exceeded. Reduce rows, CV folds or feature-count comparisons.')
    frame=load_table(source,filename)
    overview=inspect_table(frame)
    if len(frame)>cfg.max_rows:raise AnalysisError(f'Dataset exceeds {cfg.max_rows} row limit; sample it before analysis.')
    if cfg.target not in frame:raise AnalysisError('Target column does not exist.')
    if not pd.api.types.is_numeric_dtype(frame[cfg.target]):raise AnalysisError('Regression target must be numeric; nonnumeric values were detected.')
    frame=frame.drop_duplicates().dropna(subset=[cfg.target]).copy()
    if len(frame)<20:raise AnalysisError('At least 20 valid, nonduplicate target observations are required.')
    if frame[cfg.target].nunique()<5:raise AnalysisError('Choose a continuous numeric target with at least 5 distinct values.')
    chosen=cfg.features if cfg.features is not None else [c for c in frame if c!=cfg.target]
    if not chosen or len(chosen)!=len(set(chosen)) or any(c not in frame for c in chosen):
        raise AnalysisError('Select distinct existing predictor columns.')
    if cfg.target in chosen:raise AnalysisError('Target cannot also be a predictor.')
    ids=identifier_columns(frame.drop(columns=cfg.target))
    chosen=[c for c in chosen if c not in ids]
    notes=[f'Excluded identifier: {c}' for c in ids if c in (cfg.features or frame.columns.tolist())]
    if not chosen:raise AnalysisError('No predictors remain after excluding ID columns.')
    X=frame[chosen];y=frame[cfg.target]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=cfg.test_size,random_state=cfg.seed)
    if len(X_test)<4 or len(X_train)*(cfg.outer_folds-1)/cfg.outer_folds<cfg.inner_folds*3:
        raise AnalysisError('Not enough training observations for the requested nested CV folds.')
    minimum_outer=len(X_train)-math.ceil(len(X_train)/cfg.outer_folds)
    minimum_inner=minimum_outer-math.ceil(minimum_outer/cfg.inner_folds)
    if minimum_inner<max(4,cfg.inner_folds):
        raise AnalysisError('Too few observations in an inner training fold; reduce test fraction or CV folds.')
    # Exclude constants and exact target copies based solely on training rows.
    excluded=[]
    for c in X_train:
        values=X_train[c]
        named=(c.lower().startswith(cfg.target.lower()+'_') and any(t in c.lower() for t in ['copy','pred','future','result']))
        copy=False
        if pd.api.types.is_numeric_dtype(values) and values.notna().all() and values.nunique()>1:
            copy=bool(np.allclose(values.to_numpy(),y_train.to_numpy(),rtol=1e-12,atol=1e-12))
            if not copy and abs(values.corr(y_train))>1-1e-12:notes.append(f'Check possible target leakage: {c} is perfectly correlated on training data.')
        if values.nunique(dropna=True)<=1 or named or copy:excluded.append(c)
    chosen=[c for c in chosen if c not in excluded]
    notes.extend(f'Excluded constant or exact target leakage: {c}' for c in excluded)
    if not chosen:raise AnalysisError('No valid predictors remain after constant/leakage checks.')
    X_train=X_train[chosen];X_test=X_test[chosen]
    for c in X_train.select_dtypes(exclude='number'):
        if X_train[c].nunique()>cfg.max_encoded_features:
            raise AnalysisError(f'Categorical column {c} has too many levels. Exclude it or group its categories.')
    probe=make_preprocessor(X_train)
    probe_values=probe.fit_transform(X_train)
    if probe_values.shape[1]>cfg.max_encoded_features:
        raise AnalysisError(f'Encoding produces {probe_values.shape[1]} features; configured limit is {cfg.max_encoded_features}.')
    p=int(VarianceThreshold().fit(probe_values).get_support().sum())
    if not p:raise AnalysisError('All encoded predictors are constant.')
    counts=candidate_counts(p,cfg)
    workload=math.ceil((cfg.outer_folds+1)*cfg.inner_folds*len(counts)*(4+2*p/3+5*cfg.inner_folds)*max(1,len(X_train)/500))
    if workload>cfg.max_workload:raise AnalysisError(f'Estimated workload {workload} exceeds limit {cfg.max_workload}. Reduce folds or comparison counts.')
    check()
    def fit(method,k,xt,yt):
        check()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:return make_pipeline(xt,method,k,cfg).fit(xt,yt)
            except (ValueError,np.linalg.LinAlgError) as exc:
                raise AnalysisError(f'{method} cannot fit this training fold: {exc}') from exc
    def choose(method,xt,yt):
        options=[p] if method=='Baseline' else counts
        scores={k:[] for k in options}
        for train_idx,val_idx in KFold(cfg.inner_folds,shuffle=True,random_state=cfg.seed).split(xt):
            for k in options:
                model=fit(method,k,xt.iloc[train_idx],yt.iloc[train_idx])
                pred=model.predict(xt.iloc[val_idx])
                scores[k].append(metrics(yt.iloc[val_idx],pred,0)['rmse'])
        k=min(options,key=lambda q:np.mean(scores[q]))
        return k,float(np.mean(scores[k])),{key:float(np.mean(value)) for key,value in scores.items()}
    comparison=[];models=[];artifacts={}
    outer=KFold(cfg.outer_folds,shuffle=True,random_state=cfg.seed)
    for method in ['Baseline']+METHODS:
        outer_scores=[]
        for train_idx,val_idx in outer.split(X_train):
            xt,yt=X_train.iloc[train_idx],y_train.iloc[train_idx]
            k,_,_=choose(method,xt,yt)
            fitted=fit(method,k,xt,yt)
            outer_scores.append(metrics(y_train.iloc[val_idx],fitted.predict(X_train.iloc[val_idx]),0)['rmse'])
        best_k,tuning_score,candidate_scores=choose(method,X_train,y_train)
        fitted=fit(method,best_k,X_train,y_train)
        design,names=selected_design(fitted,X_train)
        test_metrics=metrics(y_test,fitted.predict(X_test),len(names))
        train_metrics=metrics(y_train,fitted.predict(X_train),len(names))
        comparison.append({'Model':method,'Features':len(names),'Selected Features':', '.join(names),
            'Requested Features':best_k,'CV RMSE':float(np.mean(outer_scores)),'CV Std':float(np.std(outer_scores)),
            'Tuning CV RMSE':tuning_score,'Train RMSE':train_metrics['rmse'],
            'Test RMSE':test_metrics['rmse'],'Test MAE':test_metrics['mae'],'Test MSE':test_metrics['mse'],
            'Test R2':test_metrics['r2'],'Test Adjusted R2':test_metrics['adjustedR2']})
        if method=='Baseline':
            artifacts[(method,best_k)]=fitted
            continue
        for k in counts:
            model=fit(method,k,X_train,y_train)
            artifacts[(method,k)]=model
            design,names=selected_design(model,X_train)
            pred_train=model.predict(X_train);pred_test=model.predict(X_test)
            order=np.argsort(y_train.index) if cfg.ordered_observations else np.arange(len(y_train))
            coefficients,f,fp,diagnostics=inference(design[order],np.asarray(y_train)[order],names,pred_train[order],cfg.ordered_observations)
            # Always export the actual sklearn coefficients, even for singular OLS designs.
            reg=model.named_steps['regressor']
            estimates=[float(reg.intercept_)]+reg.coef_.tolist()
            if not coefficients:coefficients=[dict(name=n,se=None,t=None,p=None,low=None,high=None,vif=None) for n in ['Intercept']+names]
            for c,value in zip(coefficients,estimates):c['estimate']=float(value)
            m=metrics(y_test,pred_test,len(names))
            residual=np.asarray(y_test)-pred_test
            models.append(dict(method=method,k=k,features=names,**m,
                trainRmse=metrics(y_train,pred_train,len(names))['rmse'],cvRmse=candidate_scores[k],
                intercept=float(reg.intercept_),weights=reg.coef_.tolist(),f=f,fP=fp,coefficients=coefficients,
                diagnostics=diagnostics,residuals=[dict(id=int(i)+1,actual=float(a),predicted=float(b),residual=float(e),
                    percent=finite(abs(e/a)*100) if abs(a)>1e-12 else None) for i,a,b,e in zip(y_test.index,y_test,pred_test,residual)]))
    comparison.sort(key=lambda r:r['CV RMSE'])
    for i,row in enumerate(comparison):row['Rank']=i+1
    inputs,encoding=encoded_metadata(probe,X_train)
    ranges={name:dict(min=float(probe_values[:,i].min()),max=float(probe_values[:,i].max())) for i,name in enumerate(probe.get_feature_names_out())}
    recommended=next(r for r in comparison if r['Model']!='Baseline')
    mapping={e['name']:e['column'] for e in encoding}
    for m in models:m['originalFeatures']=list(dict.fromkeys(mapping[n] for n in m['features']))
    payload=dict(generatedAt=datetime.now(timezone.utc).isoformat(),trainSize=len(y_train),testSize=len(y_test),
        seed=cfg.seed,models=models,comparison=comparison,ranges=ranges,inputs=inputs,encoding=encoding,
        featureCounts=counts,defaultMethod=recommended['Model'],defaultK=recommended['Requested Features'],
        metadata=dict(id='uploaded',title=cfg.title,problem=f'{cfg.title}: {cfg.target}',target=cfg.target,targetLabel=cfg.target,
            unit=cfg.unit,unitPrefix='',rawSize=overview['samples'],cleanSize=len(frame),sampleLabel='Row',
            maxFeatures=p,source=filename or str(source) if not isinstance(source,pd.DataFrame) else cfg.title,
            description='Uploaded dataset regression analysis.',fieldLabels={c:c for c in chosen},
            missingPolicy='Predictor imputation and one-hot encoding are fitted within each training fold; missing targets and duplicate rows are removed.',
            targetNote='Units are configured by the dataset owner; predictions are point estimates.'),
        overview=overview,warnings=notes,config=asdict(cfg),workload=workload,elapsedSeconds=time.monotonic()-started,
        predictionMode='encoded-linear',baselineRmse=next(r['Test RMSE'] for r in comparison if r['Model']=='Baseline'))
    return payload,artifacts
