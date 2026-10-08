"""Tabular risk classifiers with a unified interface.

All models: fit(X, y) / predict(X) / predict_proba(X) / save(path) / load(path).
XGBoost is preferred; falls back to sklearn GradientBoostingClassifier when
xgboost is not installed. Includes an automatic model-comparison helper.
"""
import os
import pickle

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, average_precision_score, f1_score,
                             precision_score, recall_score, roc_curve)

try:
    import xgboost as _xgb
    _XGB_AVAILABLE = True
except Exception:
    _xgb = None
    _XGB_AVAILABLE = False

XGBOOST_AVAILABLE = _XGB_AVAILABLE


def _metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(average_precision_score(y_true, y_proba)),
    }


class TabularModel:
    name = "base"

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        raise NotImplementedError

    def feature_importance(self):
        """Return dict feature -> importance (fallback-safe)."""
        return {}

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class XGBoostModel(TabularModel):
    name = "xgboost"

    def __init__(self, seed=42):
        self.seed = seed
        if _XGB_AVAILABLE:
            self.model = _xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.06,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=seed, n_jobs=-1, eval_metric="logloss",
            )
            self.backend = "xgboost"
        else:
            self.model = GradientBoostingClassifier(random_state=seed)
            self.backend = "sklearn-gbm-fallback"

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        try:
            imp = self.model.feature_importances_
        except Exception:
            return {}
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, map(float, imp)), key=lambda kv: kv[1], reverse=True))


class RandomForestModel(TabularModel):
    name = "random_forest"

    def __init__(self, seed=42):
        self.seed = seed
        self.model = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        try:
            imp = self.model.feature_importances_
        except Exception:
            return {}
        names = self.feature_names_ or [f"f{i}" for i in range(len(imp))]
        return dict(sorted(zip(names, map(float, imp)), key=lambda kv: kv[1], reverse=True))


class LogisticRegressionModel(TabularModel):
    name = "logistic_regression"

    def __init__(self, seed=42):
        self.seed = seed
        self.model = LogisticRegression(max_iter=2000, random_state=seed)

    def fit(self, X, y):
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self):
        try:
            coef = np.abs(self.model.coef_[0])
        except Exception:
            return {}
        names = self.feature_names_ or [f"f{i}" for i in range(len(coef))]
        return dict(sorted(zip(names, map(float, coef)), key=lambda kv: kv[1], reverse=True))


MODEL_CLASSES = {
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "logistic_regression": LogisticRegressionModel,
}


def compare_models(X_train, y_train, X_test, y_test, seed=42):
    """Train every tabular model and return {name: {metrics, model}}."""
    results = {}
    for name, cls in MODEL_CLASSES.items():
        model = cls(seed=seed).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        results[name] = {"metrics": _metrics(np.asarray(y_test), proba), "model": model}
    return results
