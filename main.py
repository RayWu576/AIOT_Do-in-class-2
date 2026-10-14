
# ============================================================
# 50 STARTUPS - MULTIPLE LINEAR REGRESSION
# CRISP-DM + 5 FEATURE SELECTION METHODS
#
# M0: Baseline
# M1: Exhaustive Feature Selection
# M2: Sequential Forward Selection
# M3: RFECV
# M4: Lasso Feature Selection
# M5: Sequential Backward Selection
# ============================================================

import itertools
import warnings
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.stats import shapiro
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    clone
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SequentialFeatureSelector,
    RFECV
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

RANDOM_STATE = 42
INNER_FOLDS = 4
OUTER_FOLDS = 5

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: BUSINESS UNDERSTANDING
# ============================================================

print("\n" + "=" * 65)
print("STEP 1: BUSINESS UNDERSTANDING")
print("=" * 65)

print("Objective: Predict startup profit.")
print("Task: Multiple Linear Regression")
print("Target: Profit")
print("Compare five feature selection algorithms.")


# ============================================================
# STEP 2: DATA UNDERSTANDING
# ============================================================

print("\n" + "=" * 65)
print("STEP 2: DATA UNDERSTANDING")
print("=" * 65)

df = pd.read_csv("50_Startups.csv")
df.columns = df.columns.str.strip()

required_columns = [
    "R&D Spend",
    "Administration",
    "Marketing Spend",
    "State",
    "Profit"
]

if not set(required_columns).issubset(df.columns):
    raise ValueError("Missing required dataset columns.")

df = df[required_columns].copy()

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

print("\nCorrelation Matrix:")
print(
    df.select_dtypes(include=np.number)
    .corr()
    .round(3)
)

# Correlation visualization
corr = df.select_dtypes(include=np.number).corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=35, ha="right")
ax.set_yticklabels(corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(
            j, i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.colorbar(im, ax=ax)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation.png", dpi=200)
plt.show()


# ============================================================
# STEP 3: DATA PREPARATION
# ============================================================

print("\n" + "=" * 65)
print("STEP 3: DATA PREPARATION")
print("=" * 65)

df = df.drop_duplicates().dropna()

# Explicit categories ensure stable dummy columns.
states = ["California", "Florida", "New York"]

if not df["State"].isin(states).all():
    raise ValueError("Unexpected State category.")

df["State"] = pd.Categorical(
    df["State"],
    categories=states
)

# One-hot encoding using known categories.
# California becomes the reference category.
X = pd.get_dummies(
    df.drop(columns="Profit"),
    columns=["State"],
    drop_first=True,
    dtype=float
)

y = df["Profit"]

feature_names = X.columns.tolist()

print("\nEncoded Features:")
for i, name in enumerate(feature_names, start=1):
    print(f"{i}. {name}")

print("\nTotal Features:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# NOTE:
# One-hot mapping is fixed in advance.
# Feature selection and model fitting occur
# inside cross-validation training folds.

inner_cv = KFold(
    n_splits=INNER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

outer_cv = KFold(
    n_splits=OUTER_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

SCORING = "neg_root_mean_squared_error"


# ============================================================
# FEATURE SELECTION HELPER
# ============================================================

def subset_columns(X, indices):
    """Select columns by integer positions."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, indices]

    return np.asarray(X)[:, indices]


class BaseSubsetSelector(
    BaseEstimator,
    TransformerMixin
):
    selected_indices_: np.ndarray
    n_features_in_: int

    def transform(self, X):
        return subset_columns(
            X,
            self.selected_indices_
        )

    def get_support(self):
        mask = np.zeros(
            self.n_features_in_,
            dtype=bool
        )
        mask[self.selected_indices_] = True
        return mask


# ============================================================
# M1: EXHAUSTIVE FEATURE SELECTION
# ============================================================

class ExhaustiveSelector(BaseSubsetSelector):
    """
    Test every nonempty feature subset.
    Select the subset with lowest CV RMSE.
    """

    def __init__(self, cv=4):
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        best_score = -np.inf
        best_subset = None

        for k in range(1, self.n_features_in_ + 1):
            for subset in itertools.combinations(
                range(self.n_features_in_), k
            ):
                X_subset = subset_columns(X, list(subset))

                scores = cross_val_score(
                    LinearRegression(),
                    X_subset,
                    y,
                    cv=cv,
                    scoring=SCORING
                )

                mean_score = scores.mean()

                if mean_score > best_score:
                    best_score = mean_score
                    best_subset = subset

        self.selected_indices_ = np.array(
            best_subset,
            dtype=int
        )

        return self


# ============================================================
# M2 / M5: FORWARD AND BACKWARD SELECTION
# ============================================================

class TunedSequentialSelector(BaseSubsetSelector):
    """
    Tune the number of selected features
    using training-only cross-validation.
    """

    def __init__(self, direction="forward", cv=4):
        self.direction = direction
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        best_score = -np.inf
        best_indices = None

        # The full-feature model is also a candidate.
        full_score = cross_val_score(
            LinearRegression(),
            X,
            y,
            cv=cv,
            scoring=SCORING
        ).mean()

        best_score = full_score
        best_indices = np.arange(self.n_features_in_)

        for k in range(1, self.n_features_in_):

            selector = SequentialFeatureSelector(
                estimator=LinearRegression(),
                n_features_to_select=k,
                direction=self.direction,
                scoring=SCORING,
                cv=cv
            )

            selector.fit(X, y)

            indices = np.where(
                selector.get_support()
            )[0]

            X_subset = subset_columns(X, indices)

            score = cross_val_score(
                LinearRegression(),
                X_subset,
                y,
                cv=cv,
                scoring=SCORING
            ).mean()

            if score > best_score:
                best_score = score
                best_indices = indices

        self.selected_indices_ = np.asarray(
            best_indices,
            dtype=int
        )

        return self


# ============================================================
# M3: RECURSIVE FEATURE ELIMINATION WITH CV
# ============================================================

class RFECVSelector(BaseSubsetSelector):

    def __init__(self, cv=4):
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        cv = KFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        # Standardization makes coefficient
        # magnitudes comparable during RFE.
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        selector = RFECV(
            estimator=LinearRegression(),
            step=1,
            min_features_to_select=1,
            cv=cv,
            scoring=SCORING
        )

        selector.fit(X_scaled, y)

        self.selected_indices_ = np.where(
            selector.support_
        )[0]

        return self


# ============================================================
# M4: LASSO FEATURE SELECTION
# ============================================================

class LassoSelector(BaseSubsetSelector):

    def __init__(self, cv=4):
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X)

        lasso = LassoCV(
            cv=KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=RANDOM_STATE
            ),
            max_iter=100000,
            random_state=RANDOM_STATE
        )

        lasso.fit(X_scaled, y)

        coefficients = np.abs(lasso.coef_)

        indices = np.where(
            coefficients > 1e-7
        )[0]

        # Ensure at least one feature is selected.
        if len(indices) == 0:
            indices = np.array([
                np.argmax(coefficients)
            ])

        self.selected_indices_ = indices

        return self


# ============================================================
# STEP 4: MODELING
# ============================================================

print("\n" + "=" * 65)
print("STEP 4: MODELING")
print("=" * 65)

models = {
    "Baseline": Pipeline([
        ("regressor", LinearRegression())
    ]),

    "Exhaustive": Pipeline([
        ("selector", ExhaustiveSelector(cv=INNER_FOLDS)),
        ("regressor", LinearRegression())
    ]),

    "Forward Selection": Pipeline([
        ("selector", TunedSequentialSelector(
            direction="forward",
            cv=INNER_FOLDS
        )),
        ("regressor", LinearRegression())
    ]),

    "RFECV": Pipeline([
        ("selector", RFECVSelector(cv=INNER_FOLDS)),
        ("regressor", LinearRegression())
    ]),

    "Lasso": Pipeline([
        ("selector", LassoSelector(cv=INNER_FOLDS)),
        ("regressor", LinearRegression())
    ]),

    "Backward Selection": Pipeline([
        ("selector", TunedSequentialSelector(
            direction="backward",
            cv=INNER_FOLDS
        )),
        ("regressor", LinearRegression())
    ])
}


# ============================================================
# STEP 5: EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("STEP 5: MODEL EVALUATION")
print("=" * 65)

def calculate_metrics(y_true, y_pred, p):
    n = len(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if n > p + 1:
        adj_r2 = 1 - (
            (1 - r2) * (n - 1) / (n - p - 1)
        )
    else:
        adj_r2 = np.nan

    return mae, mse, rmse, r2, adj_r2


results = []
selected_features_dict = {}
fitted_models = {}
test_predictions = {}

for name, pipeline in models.items():

    print("\n" + "-" * 55)
    print("Model:", name)
    print("-" * 55)

    # Nested CV:
    # Outer folds evaluate the complete workflow.
    # Inner folds perform feature selection.
    cv_scores = cross_val_score(
        clone(pipeline),
        X_train,
        y_train,
        cv=outer_cv,
        scoring=SCORING,
        n_jobs=1
    )

    cv_rmse = -cv_scores.mean()
    cv_std = (-cv_scores).std()

    # Fit on the complete training set
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    fitted_models[name] = fitted

    # Extract selected features
    if name == "Baseline":
        selected = feature_names
    else:
        mask = fitted.named_steps[
            "selector"
        ].get_support()

        selected = list(
            np.array(feature_names)[mask]
        )

    selected_features_dict[name] = selected

    p = len(selected)

    # Predictions
    y_train_pred = fitted.predict(X_train)
    y_test_pred = fitted.predict(X_test)

    test_predictions[name] = y_test_pred

    train_metrics = calculate_metrics(
        y_train,
        y_train_pred,
        p
    )

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
        p
    )

    results.append({
        "Model": name,
        "Features": p,
        "CV RMSE": cv_rmse,
        "CV Std": cv_std,
        "Train RMSE": train_metrics[2],
        "Test MAE": test_metrics[0],
        "Test MSE": test_metrics[1],
        "Test RMSE": test_metrics[2],
        "Test R2": test_metrics[3],
        "Test Adjusted R2": test_metrics[4]
    })

    print("Selected Features:", selected)
    print("Number of Features:", p)
    print("Nested CV RMSE:", round(cv_rmse, 4))
    print("Test RMSE:", round(test_metrics[2], 4))
    print("Test R2:", round(test_metrics[3], 4))


# ============================================================
# MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(results)

# Rank by training-only nested CV RMSE
comparison_df = comparison_df.sort_values(
    "CV RMSE",
    ascending=True
).reset_index(drop=True)

comparison_df.insert(
    0,
    "Rank",
    np.arange(1, len(comparison_df) + 1)
)

print("\n" + "=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

print(
    comparison_df.round(4).to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_DIR / "model_comparison.csv",
    index=False
)

selected_df = pd.DataFrame([
    {
        "Model": name,
        "Selected Features": ", ".join(features),
        "Feature Count": len(features)
    }
    for name, features in selected_features_dict.items()
])

selected_df.to_csv(
    OUTPUT_DIR / "selected_features.csv",
    index=False
)

print("\n===== SELECTED FEATURES =====")
print(selected_df.to_string(index=False))


# ============================================================
# BEST MODEL
# ============================================================

best_name = comparison_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

print("\nBEST MODEL BY NESTED CV:")
print(best_name)

print("\nSelected Features:")
print(selected_features_dict[best_name])

print("\nTest Performance:")
best_row = comparison_df.iloc[0]

print("MAE:", best_row["Test MAE"])
print("MSE:", best_row["Test MSE"])
print("RMSE:", best_row["Test RMSE"])
print("R2:", best_row["Test R2"])


# ============================================================
# VISUALIZATION: MODEL COMPARISON
# ============================================================

plt.figure(figsize=(11, 6))

plt.bar(
    comparison_df["Model"],
    comparison_df["CV RMSE"]
)

plt.ylabel("Nested CV RMSE")
plt.title("Feature Selection Method Comparison")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=200
)

plt.show()


# ============================================================
# ACTUAL VS PREDICTED - BEST MODEL
# ============================================================

best_test_pred = best_model.predict(X_test)

plt.figure(figsize=(7, 5))

plt.scatter(
    y_test,
    best_test_pred,
    alpha=0.8
)

minimum = min(y_test.min(), best_test_pred.min())
maximum = max(y_test.max(), best_test_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "r--"
)

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title(f"Actual vs Predicted - {best_name}")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=200
)
plt.show()


# ============================================================
# REGRESSION ASSUMPTION TESTS
# ============================================================

print("\n" + "=" * 65)
print("REGRESSION ASSUMPTION CHECKS - BEST MODEL")
print("=" * 65)

# Diagnostics use training data.
train_pred = best_model.predict(X_train)
residuals = y_train.to_numpy() - train_pred

selected_features = selected_features_dict[best_name]

X_selected = X_train[selected_features].astype(float)

X_design = sm.add_constant(
    X_selected,
    has_constant="add"
)

# ------------------------------------------------------------
# 1. LINEARITY
# ------------------------------------------------------------

print("\n1. LINEARITY")

plt.figure(figsize=(7, 5))
plt.scatter(train_pred, residuals)
plt.axhline(0, color="red", linestyle="--")

plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "linearity.png",
    dpi=200
)
plt.show()

print("Inspect for systematic nonlinear patterns.")


# ------------------------------------------------------------
# 2. INDEPENDENCE
# ------------------------------------------------------------

print("\n2. INDEPENDENCE")

dw = durbin_watson(residuals)

print("Durbin-Watson:", round(dw, 4))
print("Values near 2 suggest little serial correlation.")
print("Meaningful ordering is required for interpretation.")


# ------------------------------------------------------------
# 3. NORMALITY
# ------------------------------------------------------------

print("\n3. NORMALITY")

stat, p_value = shapiro(residuals)

print("Shapiro-Wilk p-value:", round(p_value, 4))

if p_value > 0.05:
    print("Normality not rejected.")
else:
    print("Evidence against normality.")

stats = sm.ProbPlot(residuals)
stats.qqplot(line="45")

plt.title("Q-Q Plot")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "normality.png",
    dpi=200
)
plt.show()


# ------------------------------------------------------------
# 4. HOMOSCEDASTICITY
# ------------------------------------------------------------

print("\n4. HOMOSCEDASTICITY")

bp = het_breuschpagan(
    residuals,
    X_design
)

print("Breusch-Pagan p-value:", round(bp[1], 4))

if bp[1] > 0.05:
    print("Constant variance not rejected.")
else:
    print("Evidence of nonconstant variance.")


# ------------------------------------------------------------
# 5. MULTICOLLINEARITY
# ------------------------------------------------------------

print("\n5. MULTICOLLINEARITY")

vif_values = []

for i, feature in enumerate(selected_features):
    vif = variance_inflation_factor(
        X_design.to_numpy(),
        i + 1
    )

    vif_values.append({
        "Feature": feature,
        "VIF": vif
    })

vif_df = pd.DataFrame(vif_values)

print(vif_df.to_string(index=False))

vif_df.to_csv(
    OUTPUT_DIR / "vif_results.csv",
    index=False
)


# ============================================================
# REGRESSION COEFFICIENTS
# ============================================================

print("\n===== BEST MODEL COEFFICIENTS =====")

best_regressor = best_model.named_steps["regressor"]

coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": best_regressor.coef_
})

print("Intercept:", best_regressor.intercept_)
print(coef_df.to_string(index=False))

coef_df.to_csv(
    OUTPUT_DIR / "best_model_coefficients.csv",
    index=False
)


# ============================================================
# STEP 6: DEPLOYMENT
# ============================================================

print("\n" + "=" * 65)
print("STEP 6: DEPLOYMENT")
print("=" * 65)

new_startup = pd.DataFrame({
    "R&D Spend": [150000],
    "Administration": [120000],
    "Marketing Spend": [250000],
    "State_Florida": [0.0],
    "State_New York": [0.0]
})

# California is the reference state.
new_startup = new_startup[feature_names]

prediction = best_model.predict(new_startup)

print("\nNew Startup:")
print(new_startup)

print("\nPredicted Profit:", round(prediction[0], 2))

joblib.dump(
    {
        "model": best_model,
        "feature_names": feature_names,
        "selected_features": selected_features,
        "model_name": best_name
    },
    OUTPUT_DIR / "best_startup_model.pkl"
)

predictions_df = pd.DataFrame({
    "Actual Profit": y_test.to_numpy(),
    "Predicted Profit": best_test_pred,
    "Residual": y_test.to_numpy() - best_test_pred
})

predictions_df.to_csv(
    OUTPUT_DIR / "best_model_predictions.csv",
    index=False
)

print("\nAll results saved to:", OUTPUT_DIR.resolve())
print("\nCRISP-DM PROJECT COMPLETED!")
