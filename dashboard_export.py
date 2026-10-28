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


def export_dashboard(curves, X_train, X_test, y_train, y_test, comparison, dataset_id="startups", metadata=None):
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
    metadata = dict(metadata or {
        "id": "startups", "title": "50 Startups", "problem": "50 Startups：公司利潤預測",
        "target": "Profit", "targetLabel": "公司利潤", "unit": "美元", "unitPrefix": "$",
        "rawSize": 50, "sampleLabel": "Startup", "source": "50_Startups.csv",
        "description": "從研發、行銷、行政投入與所在地，探索新創公司的利潤預測。",
        "missingPolicy": "移除重複與缺失資料列，再進行固定 80% / 20% 分割。",
        "targetNote": "Profit 單位為美元。",
        "fieldLabels": {"R&D Spend": "研發投入", "Marketing Spend": "行銷投入", "Administration": "行政支出"}
    })
    metadata["cleanSize"] = len(y_train) + len(y_test)
    metadata["maxFeatures"] = X_train.shape[1]
    inputs = []
    for name in X_train.columns:
        if name.startswith("State_"):
            continue
        col = X_train[name]
        span = float(col.max() - col.min())
        binary = col.nunique() <= 2 and set(col.unique()).issubset({0., 1.})
        integer = bool(np.allclose(col, np.round(col)))
        step = 1. if binary or integer else float(10 ** np.floor(np.log10(max(span, 1e-9) / 1000)))
        inputs.append(dict(name=name, label=metadata.get("fieldLabels", {}).get(name, name),
                           default=float(col.median()), min=float(col.min()), max=float(col.max()),
                           step=step, binary=binary))
    recommended = next(row for row in comparison.to_dict(orient="records") if row["Model"] != "Baseline")
    payload.update(metadata=metadata, inputs=inputs,
                   defaultMethod=recommended["Model"], defaultK=int(recommended["Features"]))
    existing = json.loads((output / "data.json").read_text(encoding="utf-8")) if (output / "data.json").exists() else {}
    datasets = dict(existing.get("datasets", {}))
    datasets[dataset_id] = payload
    startup = datasets.get("startups", payload)
    bundle = {**startup, "schemaVersion": 3, "datasets": datasets}
    encoded = json.dumps(bundle, ensure_ascii=False, allow_nan=False)
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
                for name in ("app.js", "data.js", "styles.css", "upload.js") if (output / name).exists()}
    text = index.read_text(encoding="utf-8")
    text = re.sub(
        r'((?:src|href)=")(app\.js|data\.js|styles\.css|upload\.js)(?:\?v=[^"]*)?(")',
        lambda match: f'{match[1]}{match[2]}?v={versions[match[2]]}{match[3]}', text
    )
    index.write_text(text, encoding="utf-8")
