#CatBoost. 
from catboost import CatBoostClassifier
from .search import run_grid

NAME = "CatBoost"


def configs(scale_pos_weight):
    return [
        {"name": "balanced",
         "depth": 6, "learning_rate": 0.05, "iterations": 250, "l2_leaf_reg": 3,
         "scale_pos_weight": scale_pos_weight},
        {"name": "deep",
         "depth": 8, "learning_rate": 0.03, "iterations": 300, "l2_leaf_reg": 5,
         "scale_pos_weight": scale_pos_weight},
        {"name": "conservative",
         "depth": 5, "learning_rate": 0.03, "iterations": 300, "l2_leaf_reg": 7,
         "min_data_in_leaf": 20, "scale_pos_weight": scale_pos_weight},
        {"name": "bayesian",
         "depth": 6, "learning_rate": 0.05, "iterations": 250,
         "bootstrap_type": "Bayesian", "bagging_temperature": 0.5,
         "scale_pos_weight": scale_pos_weight},
    ]


def train(data):
    build = lambda cfg: CatBoostClassifier(**cfg, random_state=42, verbose=0,
                                           task_type="CPU")
    return run_grid(NAME, build, configs(data.scale_pos_weight), data)