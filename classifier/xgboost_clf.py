#XGBoost. 
import xgboost as xgb
from .search import run_grid

NAME = "XGBoost"


def configs(scale_pos_weight):
    return [
        {"name": "balanced",
         "max_depth": 5, "learning_rate": 0.05, "n_estimators": 200,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "scale_pos_weight": scale_pos_weight},
        {"name": "deep",
         "max_depth": 7, "learning_rate": 0.03, "n_estimators": 300,
         "subsample": 0.8, "colsample_bytree": 0.8,
         "scale_pos_weight": scale_pos_weight},
        {"name": "conservative",
         "max_depth": 4, "learning_rate": 0.03, "n_estimators": 300,
         "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,
         "scale_pos_weight": scale_pos_weight},
        {"name": "aggressive",
         "max_depth": 6, "learning_rate": 0.05, "n_estimators": 250,
         "subsample": 0.9, "colsample_bytree": 0.9,
         "scale_pos_weight": scale_pos_weight * 1.5},
    ]


def train(data):
    build = lambda cfg: xgb.XGBClassifier(**cfg, random_state=42, tree_method="hist")
    return run_grid(NAME, build, configs(data.scale_pos_weight), data)