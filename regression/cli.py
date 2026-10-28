"""CLI: python -m regression.cli data.csv --target Y --output new-directory."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from .config import AnalysisConfig
from .engine import analyze
from .artifacts import diagnostic_png


def main():
    p=argparse.ArgumentParser(description='Reusable CRISP-DM regression engine')
    p.add_argument('dataset');p.add_argument('--target',required=True)
    p.add_argument('--features',nargs='+');p.add_argument('--test-size',type=float,default=.2)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--inner-folds',type=int,default=3)
    p.add_argument('--outer-folds',type=int,default=3);p.add_argument('--counts',type=int,nargs='+')
    p.add_argument('--max-features',type=int,default=60);p.add_argument('--max-seconds',type=int,default=180)
    p.add_argument('--title',default='Uploaded dataset');p.add_argument('--unit',default='target units')
    p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():p.error('Output directory already exists; choose a new directory to preserve prior results.')
    cfg=AnalysisConfig(target=args.target,features=args.features,test_size=args.test_size,seed=args.seed,
        inner_folds=args.inner_folds,outer_folds=args.outer_folds,feature_counts=args.counts,
        max_encoded_features=args.max_features,max_comparison_features=min(12,args.max_features),max_seconds=args.max_seconds,title=args.title,unit=args.unit)
    result,models=analyze(args.dataset,cfg)
    out.mkdir(parents=True)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    pd.DataFrame(result['comparison']).to_csv(out/'model_comparison.csv',index=False)
    joblib.dump(models,out/'models.joblib')
    recommended=next(m for m in result['models'] if m['method']==result['defaultMethod'] and m['k']==result['defaultK'])
    for kind in ('linearity','qq'):(out/(kind+'.png')).write_bytes(diagnostic_png(recommended,kind))
    print(out/'analysis.json')


if __name__=='__main__':main()
