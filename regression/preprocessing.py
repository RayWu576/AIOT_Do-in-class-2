"""Training-fold-only preprocessing and readable encoded feature metadata."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from .config import AnalysisError


def make_preprocessor(X):
    numeric=X.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in X if c not in numeric]
    parts=[]
    if numeric: parts.append(('num',SimpleImputer(strategy='median',keep_empty_features=True),numeric))
    if categorical:
        parts.append(('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
                                      ('encoder',OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False))]),categorical))
    return ColumnTransformer(parts,verbose_feature_names_out=False)


def encoded_metadata(preprocessor, X):
    names=preprocessor.get_feature_names_out().tolist()
    encoding=[]
    inputs=[]
    for group,transformer,columns in preprocessor.transformers_:
        if group=='remainder':continue
        if group=='num':
            for name,fill in zip(columns,transformer.statistics_):
                values=X[name].dropna()
                fill=float(fill)
                lo=float(values.min()) if len(values) else fill
                hi=float(values.max()) if len(values) else fill
                step=float(10**np.floor(np.log10(max(hi-lo,1e-6)/1000)))
                encoding.append(dict(name=name,column=name,kind='numeric',fill=fill))
                inputs.append(dict(name=name,label=name,kind='numeric',default=fill,min=lo,max=hi,step=step,binary=False))
        else:
            imputer=transformer.named_steps['imputer'];encoder=transformer.named_steps['encoder']
            encoded_names=iter(encoder.get_feature_names_out(columns))
            for i,(name,categories,fill) in enumerate(zip(columns,encoder.categories_,imputer.statistics_)):
                inputs.append(dict(name=name,label=name,kind='categorical',default=str(fill),options=[str(v) for v in categories]))
                dropped=encoder.drop_idx_[i]
                for j,category in enumerate(categories):
                    if j==dropped:continue
                    encoding.append(dict(name=next(encoded_names),column=name,kind='category',category=str(category),fill=str(fill)))
    # Order metadata exactly like ColumnTransformer's encoded columns.
    if len(set(names))!=len(names):
        raise AnalysisError('Encoded column names collide; rename ambiguous input columns.')
    lookup={e['name']:e for e in encoding}
    return inputs,[lookup[n] for n in names]
