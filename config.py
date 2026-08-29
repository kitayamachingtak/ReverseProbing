import os


DATA_DIR = "mimic"                # source datasets
SCRIPTS_DIR = "different_feature_numbers"
FEATURE_DIR = "features_by_script"
OUT_DIR = "outputs"               # metrics and importance json
FIG_DIR = "figures"

DATA_FILES = {
    "mimic": "hallucinations_mimic_di.jsonl",
    "generated": "hallucinations_generated_di.jsonl",
}


MODELS = {
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "display": "Mistral-7B", "max_features": 454, "color": "#D65F5F",
    },
    "biomistral-7b": {
        "hf_id": "BioMistral/BioMistral-7B",
        "display": "BioMistral-7B", "max_features": 454, "color": "#4878CF",
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "display": "Llama3.1-8B", "max_features": 454, "color": "#B47CC7",
    },
    "openbiollm-8b": {
        "hf_id": "aaditya/Llama3-OpenBioLLM-8B",
        "display": "OpenBioLLM-8B", "max_features": 454, "color": "#55A868",
    },
    "llama3.1-70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "display": "Llama3.1-70B", "max_features": 886, "color": "#C4AD66",
    },
    "openbiollm-70b": {
        "hf_id": "aaditya/Llama3-OpenBioLLM-70B",
        "display": "OpenBioLLM-70B", "max_features": 886, "color": "#77BEDB",
    },
}

MODEL_ORDER = ["mistral-7b", "biomistral-7b", "llama3.1-8b",
               "openbiollm-8b", "llama3.1-70b", "openbiollm-70b"]

DISPLAY_NAMES = [MODELS[k]["display"] for k in MODEL_ORDER]
DISPLAY_TO_KEY = {MODELS[k]["display"]: k for k in MODEL_ORDER}
MODEL_COLORS = {MODELS[k]["display"]: MODELS[k]["color"] for k in MODEL_ORDER}


FEATURE_CONFIGS = {
    "93": "collect_token_features_93",
    "120": "collect_token_features_120",
    "204": "collect_token_features_204",
    "max": None,                  # resolved per model, see max_module()
}

MAX_MODULES = {454: "collect_token_features_454",
               886: "collect_token_features_886"}

NEIGHBOR_WINDOWS = {"93": [2, 3, 5], "120": [2, 3, 5, 7],
                    "204": [2, 3, 5, 7], "max": [2, 3, 5, 7, 9]}

EXPECTED_FEATURES = {"93": 93, "120": 120, "204": 204}

EMBED_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
SEED = 42
TRAIN_RATIO = 0.8

DATASET_TITLES = {"mimic": "MIMIC-DI (human-written)",
                  "generated": "Generated-DI"}



#     export HF_HUB_CACHE=/path/to/huggingface
# For a model outside that cache, set MODEL_PATH_<KEY>, upper case with dots
# and dashes replaced by underscores:
#     export MODEL_PATH_LLAMA3_1_70B=/path/to/snapshot


def model_path(key):
    env = "MODEL_PATH_" + key.upper().replace("-", "_").replace(".", "_")
    return os.environ.get(env) or MODELS[key]["hf_id"]


def max_module(key):
    return MAX_MODULES[MODELS[key]["max_features"]]


def module_for(key, config):
    return max_module(key) if config == "max" else FEATURE_CONFIGS[config]


def expected_features(key, config):
    return EXPECTED_FEATURES.get(config, MODELS[key]["max_features"])


def data_path(dataset):
    return os.path.join(DATA_DIR, DATA_FILES[dataset])


def feature_paths(key, config, dataset):
    d = os.path.join(FEATURE_DIR, key)
    tag = f"features_{config}"
    return (os.path.join(d, f"{tag}_train_{dataset}.pkl"),
            os.path.join(d, f"{tag}_test_{dataset}.pkl"))
