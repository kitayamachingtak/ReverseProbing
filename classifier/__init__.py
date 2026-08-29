from .data import Dataset
from . import xgboost_clf, catboost_clf, logistic_clf, importance

CLASSIFIERS = {
    "XGBoost": xgboost_clf,
    "CatBoost": catboost_clf,
    "LogisticRegression": logistic_clf,
}