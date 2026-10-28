"""Shared training-only feature selectors for startup and housing experiments."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, RFECV, RFE
from sklearn.pipeline import Pipeline
RANDOM_STATE = 0
SCORING = "neg_root_mean_squared_error"

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
# FIXED-K SELECTION: CORRELATION, MUTUAL INFORMATION, RFE, LASSO
# ============================================================

class FixedCountSelector(BaseSubsetSelector):
    """Fit a feature selection method on training data for a specified k.

    LASSO fixed-k curves rank absolute standardized LassoCV coefficients.
    Zero-coefficient ties retain input column order. RFECV curves use the
    fixed-k RFE elimination path; RFECV itself selects k by cross-validation.
    """

    def __init__(self, method="Correlation", k=1, cv=4, random_state=0, strict_cv=False):
        self.method = method
        self.k = k
        self.cv = cv
        self.random_state = random_state
        self.strict_cv = strict_cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        if not 1 <= self.k <= self.n_features_in_:
            raise ValueError("k must be between 1 and the number of features.")
        values = np.asarray(X, dtype=float)
        if self.k == self.n_features_in_:
            self.selected_indices_ = np.arange(self.n_features_in_)
            return self

        if self.method == "Correlation":
            centered = values - values.mean(axis=0)
            target = np.asarray(y) - np.mean(y)
            denominator = np.sqrt((centered ** 2).sum(axis=0) * (target ** 2).sum())
            scores = np.divide(
                np.abs(centered.T @ target), denominator,
                out=np.zeros(self.n_features_in_), where=denominator > 0
            )
        elif self.method == "Mutual Information":
            # The two one-hot State columns are discrete features.
            discrete = np.array([np.unique(values[:, i]).size <= 2 for i in range(values.shape[1])])
            scores = mutual_info_regression(
                values, y, discrete_features=discrete, random_state=self.random_state
            )
        elif self.method in ("RFE", "RFECV"):
            scaled = StandardScaler().fit_transform(values)
            selector = RFE(LinearRegression(), n_features_to_select=self.k, step=1)
            selector.fit(scaled, y)
            self.selected_indices_ = np.flatnonzero(selector.support_)
            return self
        elif self.method == "LASSO" and self.strict_cv:
            # Scale independently inside every Lasso tuning fold.
            scale=float(np.std(y)) or 1.
            search=GridSearchCV(
                Pipeline([('scaler',StandardScaler()),('lasso',Lasso(max_iter=20000))]),
                {'lasso__alpha':scale*np.array([.0001,.001,.01,.1,1.])},
                cv=KFold(self.cv,shuffle=True,random_state=self.random_state),
                scoring=SCORING,error_score='raise'
            ).fit(values,y)
            scores=np.abs(search.best_estimator_.named_steps['lasso'].coef_)
        elif self.method == "LASSO":
            scaled = StandardScaler().fit_transform(values)
            lasso = LassoCV(
                cv=KFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE),
                max_iter=100000, random_state=RANDOM_STATE
            ).fit(scaled, y)
            scores = np.abs(lasso.coef_)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self.selected_indices_ = np.argsort(-scores, kind="stable")[:self.k]
        return self


class TunedCountSelector(BaseSubsetSelector):
    """Choose k using training CV, refitting selection inside every fold."""

    def __init__(self, method="Correlation", cv=4):
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        cv = KFold(n_splits=self.cv, shuffle=True, random_state=RANDOM_STATE)
        best_score = -np.inf
        for k in range(1, self.n_features_in_ + 1):
            candidate = Pipeline([
                ("selector", FixedCountSelector(self.method, k, self.cv)),
                ("regressor", LinearRegression())
            ])
            score = cross_val_score(candidate, X, y, cv=cv, scoring=SCORING).mean()
            if score > best_score:
                best_score = score
                self.best_k_ = k
        fitted = FixedCountSelector(self.method, self.best_k_, self.cv).fit(X, y)
        self.selected_indices_ = fitted.selected_indices_
        return self


# ============================================================
# M4: RECURSIVE FEATURE ELIMINATION WITH CV
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
# M5: LASSO FEATURE SELECTION
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
