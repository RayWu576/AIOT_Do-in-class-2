"""Run Boston Housing with the same five feature-selection workflows."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from feature_selection import FixedCountSelector, TunedCountSelector, RFECVSelector, LassoSelector
from dashboard_export import export_dashboard

ROOT=Path(__file__).resolve().parent
METHODS=["Correlation","Mutual Information","RFE","RFECV","LASSO"]


def load_housing():
    frame=pd.read_csv(ROOT/'datasets/boston_housing/HousingData.csv')
    frame.columns=frame.columns.str.strip().str.upper()
    if 'MEDV' not in frame or frame.shape[1]!=14:
        raise ValueError('Expected MEDV target and 13 Boston Housing input features')
    frame=frame.apply(pd.to_numeric,errors='raise').drop_duplicates().dropna()
    return train_test_split(frame.drop(columns='MEDV'),frame.MEDV,test_size=.2,random_state=0)


def run_housing():
    output=ROOT/'results/boston_housing'
    output.mkdir(parents=True,exist_ok=True)
    X_train,X_test,y_train,y_test=load_housing()
    print(f"\nBOSTON HOUSING: clean={len(y_train)+len(y_test)}, train={len(y_train)}, test={len(y_test)}",flush=True)
    outer=KFold(5,shuffle=True,random_state=0)
    models={
        'Baseline':Pipeline([('regressor',LinearRegression())]),
        **{m:Pipeline([('selector',TunedCountSelector(m,4)),('regressor',LinearRegression())])
           for m in ['Correlation','Mutual Information','RFE']},
        'RFECV':Pipeline([('selector',RFECVSelector(4)),('regressor',LinearRegression())]),
        'LASSO':Pipeline([('selector',LassoSelector(4)),('regressor',LinearRegression())])
    }
    rows,curve_rows,selected=[],[],[]
    fitted_models={}
    for name,pipeline in models.items():
        scores=-cross_val_score(clone(pipeline),X_train,y_train,cv=outer,scoring='neg_root_mean_squared_error')
        fitted=clone(pipeline).fit(X_train,y_train)
        fitted_models[name]=fitted
        feats=X_train.columns.tolist() if name=='Baseline' else X_train.columns[fitted.named_steps['selector'].get_support()].tolist()
        train_pred,test_pred=fitted.predict(X_train),fitted.predict(X_test)
        p,n=len(feats),len(y_test)
        r2=r2_score(y_test,test_pred)
        rows.append({'Model':name,'Features':p,'CV RMSE':float(scores.mean()),'CV Std':float(scores.std()),
                     'Train RMSE':float(np.sqrt(mean_squared_error(y_train,train_pred))),
                     'Test MAE':float(mean_absolute_error(y_test,test_pred)),
                     'Test MSE':float(mean_squared_error(y_test,test_pred)),
                     'Test RMSE':float(np.sqrt(mean_squared_error(y_test,test_pred))),
                     'Test R2':float(r2),'Test Adjusted R2':float(1-(1-r2)*(n-1)/(n-p-1))})
        selected.append({'Model':name,'Selected Features':', '.join(feats),'Feature Count':p})
        print(f"Boston {name}: nested CV RMSE={scores.mean():.4f}, features={p}",flush=True)
    for method in METHODS:
        for k in range(1,X_train.shape[1]+1):
            selector=FixedCountSelector(method,k,4).fit(X_train,y_train)
            feats=X_train.columns[selector.selected_indices_].tolist()
            fitted=LinearRegression().fit(X_train[feats],y_train)
            pred=fitted.predict(X_test[feats])
            curve_rows.append({'Model':method,'Features':k,'Selected Features':', '.join(feats),
                               'Test RMSE':float(np.sqrt(mean_squared_error(y_test,pred))),
                               'Test R2':float(r2_score(y_test,pred))})
    comparison=pd.DataFrame(rows).sort_values('CV RMSE').reset_index(drop=True)
    comparison.insert(0,'Rank',range(1,len(rows)+1))
    curves=pd.DataFrame(curve_rows)
    comparison.to_csv(output/'model_comparison.csv',index=False)
    curves.to_csv(output/'feature_count_metrics.csv',index=False)
    pd.DataFrame(selected).to_csv(output/'selected_features.csv',index=False)
    best_name=comparison.iloc[0].Model
    best=fitted_models[best_name]
    best_pred=best.predict(X_test)
    pd.DataFrame({'Sample ID':y_test.index+1,'Actual MEDV':y_test,'Predicted MEDV':best_pred,
                  'Residual':y_test-best_pred}).to_csv(output/'best_model_predictions.csv',index=False)
    joblib.dump({'model':best,'model_name':best_name,'feature_names':X_train.columns.tolist()},output/'best_housing_model.pkl')
    metadata=json.loads((ROOT/'datasets/boston_housing/metadata.json').read_text(encoding='utf-8'))
    export_dashboard(curves,X_train,X_test,y_train,y_test,comparison,dataset_id='boston',metadata=metadata)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for method,marker in zip(METHODS,['o','s','^','D','v']):
        points=curves[curves.Model==method]
        for ax,metric in zip(axes,['Test RMSE','Test R2']):
            ax.plot(points.Features,points[metric],marker=marker,label=method)
            ax.set(xlabel='Number of features',ylabel=metric,title='Boston Housing — '+metric)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'model_comparison.png',dpi=200)
    plt.close(fig)
    print('BOSTON HOUSING COMPLETED: '+best_name,flush=True)


if __name__=='__main__':
    run_housing()
