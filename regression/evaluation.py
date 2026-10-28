"""Finite JSON-safe regression metrics and assumption diagnostics."""
import warnings
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def finite(value):
    try:return float(value) if np.isfinite(value) else None
    except (TypeError,ValueError):return None


def metrics(y,pred,p):
    n=len(y);r2=r2_score(y,pred) if n>1 else None
    return dict(rmse=finite(np.sqrt(mean_squared_error(y,pred))),mae=finite(mean_absolute_error(y,pred)),
                mse=finite(mean_squared_error(y,pred)),r2=finite(r2),
                adjustedR2=finite(1-(1-r2)*(n-1)/(n-p-1)) if n>p+1 and r2 is not None else None)


def inference(values,y,names,pred,ordered=False):
    design=sm.add_constant(values,has_constant='add')
    residual=np.asarray(y)-pred
    full_rank=np.linalg.matrix_rank(design)==design.shape[1]
    enough=len(y)>design.shape[1]
    coefficients=[];f=fp=None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            fit=sm.OLS(y,design).fit()
            f=finite(fit.fvalue) if enough and full_rank else None
            fp=finite(fit.f_pvalue) if enough and full_rank else None
            ci=np.asarray(fit.conf_int())
            for i,name in enumerate(['Intercept']+names):
                coefficients.append(dict(name=name,estimate=finite(fit.params[i]),
                    se=finite(fit.bse[i]) if enough and full_rank else None,
                    t=finite(fit.tvalues[i]) if enough and full_rank else None,
                    p=finite(fit.pvalues[i]) if enough and full_rank else None,
                    low=finite(ci[i,0]) if enough and full_rank else None,
                    high=finite(ci[i,1]) if enough and full_rank else None,
                    vif=finite(variance_inflation_factor(design,i)) if i>0 and design.shape[1]>2 else (1. if i else None)))
        except (ValueError,np.linalg.LinAlgError,ZeroDivisionError):pass
        normal=None
        if len(residual)>=3 and np.std(residual)>1e-12:
            # Shapiro is only evaluated on up to 5000 deterministic training residuals.
            sw=stats.shapiro(residual[:5000]);normal=dict(statistic=finite(sw.statistic),p=finite(sw.pvalue),samples=min(5000,len(residual)))
        bp=None
        if full_rank and enough and design.shape[1]>1:
            try:
                values_bp=het_breuschpagan(residual,design)
                bp=dict(statistic=finite(values_bp[0]),p=finite(values_bp[1]))
            except (ValueError,np.linalg.LinAlgError):pass
        qq=[]
        if len(residual)>=3:
            q=stats.probplot(residual,dist='norm',fit=False)
            qq=[dict(theoretical=float(a),observed=float(b)) for a,b in zip(*q)]
    diagnostics=dict(linearity=dict(fitted=np.asarray(pred).tolist(),residuals=residual.tolist()),
        durbinWatson=finite(durbin_watson(residual)) if ordered else None,
        independenceNote='User declared meaningful row order.' if ordered else 'Not interpreted: observations were randomly split; no meaningful serial order declared.',
        shapiro=normal,breuschPagan=bp,qq=qq,fullRank=bool(full_rank),
        note='Exploratory post-selection OLS statistics; no correction for feature selection.' if enough and full_rank else 'Insufficient residual degrees of freedom or singular design: undefined inference is null.')
    return coefficients,f,fp,diagnostics
