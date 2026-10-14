"""Export actual fitted linear and forest models for static browser inference."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def export_dashboard(curves, fitted_models, X_train, X_test, y_train, y_test, comparison, best_name):
    models, checks = [], []
    for _, row in curves.iterrows():
        method, k = row["Model"], int(row["Features"])
        fitted = fitted_models[(method, k)]
        features = X_train.columns[fitted.named_steps["selector"].get_support()].tolist()
        estimator = fitted.named_steps["regressor"]
        predicted = fitted.predict(X_test)
        design = sm.add_constant(X_train[features].astype(float), has_constant="add")
        vif = [float(variance_inflation_factor(design.to_numpy(), i + 1)) for i in range(k)]
        forest = method == "Random Forest"
        coefficients, trees, weights, importances = [], [], [], []
        intercept = None
        if forest:
            importances = estimator.feature_importances_.tolist()
            for tree in estimator.estimators_:
                t = tree.tree_
                trees.append(dict(left=t.children_left.tolist(), right=t.children_right.tolist(),
                                  feature=t.feature.tolist(), threshold=t.threshold.tolist(),
                                  value=t.value[:, 0, 0].tolist()))
        else:
            scaler = fitted.named_steps["scaler"]
            weights = (estimator.coef_ / scaler.scale_).tolist()
            intercept = float(estimator.intercept_ - np.dot(weights, scaler.mean_))
            ols = sm.OLS(y_train, design).fit() if method == "Linear Regression" else None
            for i, (name, value) in enumerate(zip(["Intercept"] + features, [intercept] + weights)):
                coefficients.append(dict(
                    name=name, estimate=float(value),
                    se=float(ols.bse.iloc[i]) if ols is not None else None,
                    t=float(ols.tvalues.iloc[i]) if ols is not None else None,
                    p=float(ols.pvalues.iloc[i]) if ols is not None else None,
                    low=float(ols.conf_int().iloc[i, 0]) if ols is not None else None,
                    high=float(ols.conf_int().iloc[i, 1]) if ols is not None else None,
                    vif=None if i == 0 else vif[i - 1]
                ))
        residual = np.asarray(y_test) - predicted
        models.append(dict(
            method=method, kind="forest" if forest else "linear", k=k, features=features,
            rmse=float(row["Test RMSE"]), r2=float(row["Test R2"]), mae=float(row["Test MAE"]),
            trainRmse=float(row["Train RMSE"]), cvRmse=float(row["Tuning CV RMSE"]),
            adjustedR2=None, intercept=intercept, weights=weights, coefficients=coefficients,
            importances=importances, trees=trees, parameters=json.loads(row["Best Parameters"]),
            residuals=[dict(id=int(idx) + 1, actual=float(actual), predicted=float(pred),
                            residual=float(error), percent=float(abs(error / actual) * 100))
                       for idx, actual, pred, error in zip(y_test.index, y_test, predicted, residual)]
        ))
        examples = X_test.iloc[:3]
        checks.append(dict(method=method, k=k, inputs=examples.to_dict(orient="records"),
                           expected=fitted.predict(examples).tolist()))
    payload = dict(schemaVersion=2, generatedAt=datetime.now(timezone.utc).isoformat(),
                   trainSize=len(y_train), testSize=len(y_test), seed=0, models=models,
                   recommended=best_name, comparison=comparison.to_dict(orient="records"),
                   ranges={f: dict(min=float(X_train[f].min()), max=float(X_train[f].max())) for f in X_train})
    root = Path(__file__).resolve().parent
    output = root / "dashboard" / "dist"
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    (output / "data.json").write_text(encoded, encoding="utf-8")
    (output / "data.js").write_text("window.DASHBOARD_DATA = " + encoded + ";\n", encoding="utf-8")
    (root / "results" / "browser_prediction_checks.json").write_text(json.dumps(checks), encoding="utf-8")
    # The winning model may have a different k than any other family.
    winning_k = int(comparison.loc[comparison["Model"] == best_name, "Features"].iloc[0])
    winning = next(m for m in models if m["method"] == best_name and m["k"] == winning_k)
    pd.DataFrame(winning["coefficients"], columns=["name", "estimate", "se", "t", "p", "low", "high", "vif"]).to_csv(
        root / "results" / "best_model_coefficients.csv", index=False)
    pd.DataFrame({"Feature": winning["features"], "VIF": [float(variance_inflation_factor(
        sm.add_constant(X_train[winning["features"]], has_constant="add").to_numpy(), i + 1
    )) for i in range(winning_k)]}).to_csv(root / "results" / "vif_results.csv", index=False)
    pd.DataFrame({"Feature": winning["features"],
                  "Importance": winning["importances"] if winning["kind"] == "forest" else [None] * winning_k}).to_csv(
        root / "results" / "best_model_importances.csv", index=False)
    print("Dashboard exports complete: 25 real predictive models", flush=True)
