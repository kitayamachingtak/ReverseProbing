#Logistic regression.

#Linear models are sensitive to feature scale, so the classifier is wrapped in a pipeline with StandardScaler. 

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .search import run_grid

NAME = "LogisticRegression"


def configs(scale_pos_weight):
    scaled = {0: 1.0, 1: scale_pos_weight}   # matches the tree models' weighting
    return [
        {"name": "l2_scaled_weight",
         "penalty": "l2", "C": 1.0, "solver": "liblinear", "class_weight": scaled},
        {"name": "l2_strong_reg",
         "penalty": "l2", "C": 0.1, "solver": "liblinear", "class_weight": scaled},
        {"name": "l1_sparse",
         "penalty": "l1", "C": 0.1, "solver": "liblinear", "class_weight": scaled},
        {"name": "l2_builtin_balanced",
         "penalty": "l2", "C": 1.0, "solver": "lbfgs", "class_weight": "balanced",
         "max_iter": 3000},
    ]


def build(config):
    config = dict(config)
    max_iter = config.pop("max_iter", 2000)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=max_iter, random_state=42, **config)),
    ])


def train(data):
    return run_grid(NAME, build, configs(data.scale_pos_weight), data)