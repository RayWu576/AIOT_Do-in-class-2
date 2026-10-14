"""Compare five predictive regressors using identical data and nested CV.

Run: python main.py
Fixed-k plots use a common training-only Pearson ranking. Nested CV tunes
both k and model hyperparameters without consulting the held-out test set.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dashboard_export import export_dashboard

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results"
RANDOM_STATE = 0
INNER_FOLDS, OUTER_FOLDS = 4, 5
SCORING = "neg_root_mean_squared_error"
METHODS = ["Linear Regression", "Ridge", "Lasso", "Elastic Net", "Random Forest"]


def prepare_data():
    df = pd.read_csv(ROOT / "50_Startups.csv")
    df.columns = df.columns.str.strip()
    columns = ["R&D Spend", "Administration", "Marketing Spend", "State", "Profit"]
    if not set(columns).issubset(df.columns):
        raise ValueError("Missing dataset columns")
    df = df[columns].drop_duplicates().dropna().copy()
    states = ["California", "Florida", "New York"]
    if not df["State"].isin(states).all():
        raise ValueError("Unexpected State category")
    df["State"] = pd.Categorical(df["State"], categories=states)
    X = pd.get_dummies(df.drop(columns="Profit"), columns=["State"], drop_first=True, dtype=float)
    return train_test_split(X, df["Profit"], test_size=.2, random_state=RANDOM_STATE)


def make_search(method, k=None):
    estimators = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=100000),
        "Elastic Net": ElasticNet(max_iter=100000),
        "Random Forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        )
    }
    grids = {
        "Linear Regression": {},
        "Ridge": {"regressor__alpha": [.01, .1, 1, 10, 100]},
        "Lasso": {"regressor__alpha": [1, 10, 100, 1000, 5000]},
        "Elastic Net": {"regressor__alpha": [.01, .1, 1, 10], "regressor__l1_ratio": [.2, .8]},
        "Random Forest": {"regressor__max_depth": [3, None]}
    }
    pipeline = Pipeline([
        ("selector", SelectKBest(f_regression, k=5 if k is None else k)),
        ("scaler", "passthrough" if method == "Random Forest" else StandardScaler()),
        ("regressor", estimators[method])
    ])
    grid = dict(grids[method])
    if k is None:
        grid["selector__k"] = [1, 2, 3, 4, 5]
    return GridSearchCV(
        pipeline, grid, scoring=SCORING,
        cv=KFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1, error_score="raise"
    )


def metrics(fitted, X_train, X_test, y_train, y_test):
    pred = fitted.predict(X_test)
    return {"Train RMSE": float(np.sqrt(mean_squared_error(y_train, fitted.predict(X_train)))),
            "Test MAE": float(mean_absolute_error(y_test, pred)),
            "Test MSE": float(mean_squared_error(y_test, pred)),
            "Test RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "Test R2": float(r2_score(y_test, pred))}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_data()
    feature_names = X_train.columns.to_numpy()
    outer = KFold(OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    comparisons, curves, fitted_curves, fitted_models = [], [], {}, {}
    print("50 STARTUPS: FIVE PREDICTIVE MODELS", flush=True)
    print(f"Train={len(y_train)}, Test={len(y_test)}, seed={RANDOM_STATE}", flush=True)
    for method in METHODS:
        print(f"\nModel: {method} — nested CV", flush=True)
        search = make_search(method)
        outer_rmse = -cross_val_score(clone(search), X_train, y_train, cv=outer, scoring=SCORING)
        search.fit(X_train, y_train)
        fitted = search.best_estimator_
        fitted_models[method] = fitted
        feats = feature_names[fitted.named_steps["selector"].get_support()].tolist()
        comparisons.append({
            "Model": method, "Features": len(feats), "Selected Features": ", ".join(feats),
            "CV RMSE": float(outer_rmse.mean()), "CV Std": float(outer_rmse.std()),
            "Best Parameters": json.dumps(search.best_params_),
            **metrics(fitted, X_train, X_test, y_train, y_test)
        })
        for k in range(1, 6):
            fixed = make_search(method, k).fit(X_train, y_train)
            fixed_model = fixed.best_estimator_
            fitted_curves[(method, k)] = fixed_model
            features = feature_names[fixed_model.named_steps["selector"].get_support()].tolist()
            curves.append({"Model": method, "Features": k,
                           "Selected Features": ", ".join(features),
                           "Tuning CV RMSE": float(-fixed.best_score_),
                           "Best Parameters": json.dumps(fixed.best_params_),
                           **metrics(fixed_model, X_train, X_test, y_train, y_test)})
        print(f"Nested CV RMSE={outer_rmse.mean():.2f}; selected features={feats}", flush=True)

    comparison = pd.DataFrame(comparisons).sort_values("CV RMSE").reset_index(drop=True)
    comparison.insert(0, "Rank", np.arange(1, len(comparison) + 1))
    curve_df = pd.DataFrame(curves)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    curve_df.to_csv(OUTPUT_DIR / "feature_count_metrics.csv", index=False)
    comparison[["Model", "Selected Features", "Features"]].rename(
        columns={"Features": "Feature Count"}
    ).to_csv(OUTPUT_DIR / "selected_features.csv", index=False)
    best_name = comparison.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    best_features = feature_names[best_model.named_steps["selector"].get_support()].tolist()
    predictions = best_model.predict(X_test)
    pd.DataFrame({"Sample ID": y_test.index + 1, "Actual Profit": y_test,
                  "Predicted Profit": predictions, "Residual": y_test - predictions}).to_csv(
        OUTPUT_DIR / "best_model_predictions.csv", index=False)
    joblib.dump(dict(model=best_model, model_name=best_name, feature_names=list(feature_names),
                     selected_features=best_features), OUTPUT_DIR / "best_startup_model.pkl")
    joblib.dump(fitted_curves, OUTPUT_DIR / "dashboard_models.pkl")
    export_dashboard(curve_df, fitted_curves, X_train, X_test, y_train, y_test, comparison, best_name)

    colors = ["#5585ef", "#09ad87", "#e7a43c", "#9c68db", "#e16a87"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for method, color, marker in zip(METHODS, colors, ["o", "s", "^", "D", "v"]):
        rows = curve_df[curve_df["Model"] == method]
        for ax, metric, title in zip(axes, ["Test RMSE", "Test R2"], ["Test RMSE", "Test R-squared"]):
            ax.plot(rows["Features"], rows[metric], label=method, color=color, marker=marker, linewidth=2)
            ax.set(title=f"{title} by Number of Features — 5 Predictive Models", xlabel="Number of Features", ylabel=metric)
            ax.set_xticks(range(1, 6))
            ax.grid(alpha=.25, linestyle="--")
            ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, predictions)
    limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(limits, limits, "r--")
    ax.set(xlabel="Actual Profit", ylabel="Predicted Profit", title=f"Actual vs Predicted — {best_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)
    new = pd.DataFrame({"R&D Spend": [150000], "Administration": [120000],
                        "Marketing Spend": [250000], "State_Florida": [0.], "State_New York": [0.]})
    print("\nFINAL MODEL COMPARISON\n" + comparison.round(4).to_string(index=False), flush=True)
    print(f"Best by nested CV: {best_name}", flush=True)
    print(f"New startup prediction: {best_model.predict(new[X_train.columns])[0]:.2f}", flush=True)
    print("CRISP-DM PROJECT COMPLETED!", flush=True)


if __name__ == "__main__":
    main()
