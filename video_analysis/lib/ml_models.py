"""
This module contains the Machine Learning models that have been evaluated and the differente parametrization options.
"""

from . import config

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier

random_state = config.my_random_state

# Dynamic detection of GPU for XGBoost
try:
    import xgboost as xgb
    import numpy as np

    test_model = xgb.XGBClassifier(device="gpu", n_estimators=1)
    test_model.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    xgb_devices = ["gpu"]
except Exception:
    xgb_devices = ["cpu"]

models = {
    "LogisticRegression": LogisticRegression,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "XGBClassifier": XGBClassifier,
    "LDA": LinearDiscriminantAnalysis,
}

model_parameter_rules = {
    LogisticRegression: [
        {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [10000],
            "random_state": [random_state],
        },
        {
            "C": [0.1, 1],
            "penalty": ["l1"],
            "solver": ["saga"],
            "max_iter": [10000],
            "random_state": [random_state],
        },
    ],
    KNeighborsClassifier: [
        {
            "n_neighbors": [3, 7, 11],
            "weights": ["uniform", "distance"],
            "metric": ["minkowski"],
            "p": [1, 2],
        }
    ],
    RandomForestClassifier: [
        {
            "n_estimators": [100, 200],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 4],
            "max_depth": [None, 10],
            "random_state": [random_state],
        }
    ],
    DecisionTreeClassifier: [
        {
            "criterion": ["gini", "entropy"],
            "splitter": ["best"],
            "max_features": ["sqrt", None],
            "class_weight": ["balanced"],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 4],
            "max_depth": [None, 10],
            "random_state": [random_state],
        }
    ],
    AdaBoostClassifier: [
        {
            "n_estimators": [50, 100],
            "learning_rate": [0.5, 1],
            "random_state": [random_state],
        }
    ],
    XGBClassifier: [
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.1, 0.3],
            "max_depth": [4, 6],
            "device": xgb_devices,
            "random_state": [random_state],
        }
    ],
    LinearDiscriminantAnalysis: [
        {"solver": ["svd"]},
        {"solver": ["lsqr"], "shrinkage": ["auto", 0.1, 0.5]},
    ],
}
