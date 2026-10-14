"""Export real fitted models for the browser dashboard (no server required)."""
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def export_dashboard(curves, X_train, X_test, y_train, y_test, comparison):
    models = []
    for _, row in curves.iterrows():
        features = row["Selected Features"].split(", ")
        fit = LinearRegression().fit(X_train[features], y_train)
        predicted = fit.predict(X_test[features])
        design = sm.add_constant(X_train[features].astype(float), has_constant="add")
        ols = sm.OLS(y_train, design).fit()
        intervals = np.asarray(ols.conf_int())
        coefficients = []
        for i, name in enumerate(["Intercept"] + features):
            coefficients.append(dict(
                name=name, estimate=float(ols.params.iloc[i]),
                se=float(ols.bse.iloc[i]), t=float(ols.tvalues.iloc[i]),
                p=float(ols.pvalues.iloc[i]), low=float(intervals[i, 0]),
                high=float(intervals[i, 1]),
                vif=None if i == 0 else float(variance_inflation_factor(design.to_numpy(), i))
            ))
        k, n = len(features), len(y_test)
        residuals = np.asarray(y_test) - predicted
        models.append(dict(
            method=row["Model"], k=k, features=features,
            rmse=float(row["Test RMSE"]), r2=float(row["Test R2"]),
            adjustedR2=float(1 - (1 - row["Test R2"]) * (n - 1) / (n - k - 1)),
            mae=float(mean_absolute_error(y_test, predicted)),
            f=float(ols.fvalue), fP=float(ols.f_pvalue),
            intercept=float(fit.intercept_), weights=[float(v) for v in fit.coef_],
            coefficients=coefficients,
            residuals=[dict(id=int(index) + 1, actual=float(actual), predicted=float(pred),
                            residual=float(error), percent=float(abs(error / actual) * 100))
                       for index, actual, pred, error in zip(y_test.index, y_test, predicted, residuals)]
        ))
    payload = dict(
        generatedAt=datetime.now(timezone.utc).isoformat(), trainSize=len(y_train),
        testSize=len(y_test), seed=0, models=models,
        comparison=comparison.to_dict(orient="records"),
        ranges={name: dict(min=float(X_train[name].min()), max=float(X_train[name].max()))
                for name in X_train.columns}
    )
    output = Path(__file__).resolve().parent / "dashboard" / "dist"
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    (output / "data.json").write_text(encoded, encoding="utf-8")
    (output / "data.js").write_text("window.DASHBOARD_DATA = " + encoded + ";\n", encoding="utf-8")
    refresh_asset_versions(output)
    print("Interactive dashboard data exported:", output)


def refresh_asset_versions(output=None):
    """Match asset URLs to their content so browsers cannot reuse old JS/data."""
    output = Path(output) if output is not None else Path(__file__).resolve().parent / "dashboard" / "dist"
    index = output / "index.html"
    if not index.exists():
        return
    versions = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()[:12]
                for name in ("app.js", "data.js", "styles.css")}
    text = index.read_text(encoding="utf-8")
    text = re.sub(
        r'((?:src|href)=")(app\.js|data\.js|styles\.css)(?:\?v=[^"]*)?(")',
        lambda match: f'{match[1]}{match[2]}?v={versions[match[2]]}{match[3]}', text
    )
    index.write_text(text, encoding="utf-8")
