"""Shared constants and loaders for the plotting scripts.

Everything is read from outputs/, which train_classifiers.py writes:

    outputs/metrics_{model}_{config}_{dataset}.json
        {"XGBoost": {"f1":..., "auprc":..., "auc":..., ...}, "CatBoost": {...}}

    outputs/importance_{model}_{config}_{dataset}.json
        {"XGBoost": {"importance_type": "gain",
                     "importance": {"delta_energy": 12.3, ...}}, ...}

Baseline scores are not produced by that script, so they live in a separate
file you fill in once:

    outputs/baselines.json
        {"SE Probes": {"mimic": {"Mistral-7B": 0.0474, ...},
                       "generated": {...}}, ...}
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

OUT_DIR = cfg.OUT_DIR
FIG_DIR = cfg.FIG_DIR

MODELS = cfg.DISPLAY_NAMES
DISPLAY_TO_KEY = cfg.DISPLAY_TO_KEY
MODEL_COLORS = cfg.MODEL_COLORS
CONFIGS = list(cfg.FEATURE_CONFIGS)
DATASETS = list(cfg.DATA_FILES)
DATASET_TITLES = cfg.DATASET_TITLES
MAX_FEATURES = {cfg.MODELS[k]["display"]: cfg.MODELS[k]["max_features"]
                for k in cfg.MODEL_ORDER}

PALETTES = {
    "teal": ["#f4f4f2", "#cfe0e4", "#8fb8c4", "#4d8497", "#255a72", "#11324a"],
    "ember": ["#f6f3ef", "#e8d5c4", "#d6a88c", "#bc7255", "#94422f", "#5c1f18"],
    "purple": ["#f7f5f9", "#ded6e8", "#bda7ce", "#9273ac", "#67488a", "#3b2159"],
}

# ---------------------------------------------------------------- families

FAMILIES = ["Context Contrast", "Neighbourhood", "Attention", "Hidden states",
            "Ranking / semantic", "Logit", "Lexical", "Medical Knowledge"]

SIGNALS = ["Distributional", "Semantic", "Internal", "Medical"]

FAMILY_TO_SIGNAL = {
    "Context Contrast": "Distributional",
    "Neighbourhood": "Distributional",
    "Attention": "Internal",
    "Hidden states": "Internal",
    "Ranking / semantic": "Semantic",
    "Logit": "Semantic",
    "Lexical": "Medical",
    "Medical Knowledge": "Medical",
}


def family_of(feature):
    """Map a raw feature name to one of FAMILIES.

    The order matters: rank_top* belongs to ranking, but rank_with_bhc and
    rank_without_bhc are context-contrast features.
    """
    f = feature
    if f.startswith(("neighbor_", "isolation_", "relative_isolation",
                     "medical_density")):
        return "Neighbourhood"
    if f.startswith(("attn_", "rollout_")):
        return "Attention"
    if f.startswith(("hidden_", "layer_")):
        return "Hidden states"
    if f.startswith(("rank_top", "in_top", "max_sim", "avg_sim", "top3_sim",
                     "sim_std", "semantic_rank")):
        return "Ranking / semantic"
    if f.startswith(("is_medical", "ner_", "medical_confidence")):
        return "Medical Knowledge"
    if f.startswith(("freq", "idf", "rarity", "baseline_")):
        return "Lexical"
    if f.startswith(("delta_", "kl_", "js_", "pmi", "npmi", "cpmi",
                     "topk_over", "topk_jac", "rank_with", "rank_without",
                     "bhc_", "ctx_", "total_info", "halluc_", "patient_",
                     "context_reliance", "prob_dominance", "prob_vs")):
        return "Context Contrast"
    return "Logit"


def signal_of(feature):
    return FAMILY_TO_SIGNAL[family_of(feature)]


# ---------------------------------------------------------------- loading

def _path(kind, model, config, dataset):
    key = DISPLAY_TO_KEY.get(model, model)
    return os.path.join(OUT_DIR, f"{kind}_{key}_{config}_{dataset}.json")


def load_metrics(model, config, dataset):
    path = _path("metrics", model, config, dataset)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_importance(model, config, dataset, classifier):
    path = _path("importance", model, config, dataset)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        blob = json.load(f)
    return blob.get(classifier)


def load_baselines():
    path = os.path.join(OUT_DIR, "baselines.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It holds baseline AUPRC per method, dataset "
            "and model; see the module docstring for the format.")
    with open(path) as f:
        return json.load(f)


def group_importance(importance, groups="family", top_n=20):
    """Aggregate one importance vector into families or signal categories.

    Returns percentages of TOTAL importance carried by the top-n features of
    each group, so the values sum to the top-n coverage rather than to 100.
    """
    values = importance["importance"]
    total = sum(values.values()) or 1.0
    top = sorted(values.items(), key=lambda kv: -kv[1])[:top_n]

    keys = FAMILIES if groups == "family" else SIGNALS
    fn = family_of if groups == "family" else signal_of

    out = {k: 0.0 for k in keys}
    for feature, value in top:
        out[fn(feature)] += 100.0 * value / total
    return out


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
