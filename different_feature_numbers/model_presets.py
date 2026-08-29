# 8B reference: 32 layers, 32 heads. 70B: 80 layers, 64 heads.
PRESETS = {
    32: {
        "93": {
            "target_layers": [8, 16, 24, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
        },
        "120": {
            "target_layers": [2, 6, 10, 14, 18, 22, 26, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
        },
        "204": {
            "target_layers": [2, 6, 10, 14, 18, 22, 26, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
            # three bands of four consecutive layers -> 11 adjacent pairs
            "attention_layers": (list(range(0, 4)) + list(range(14, 18))
                                 + list(range(28, 32))),
            "drift_heads": [0, 8, 16, 24],
            "rollout_checkpoints": [3, 17, 31],       # last layer of each band
        },
    },
    80: {
        "93": {
            "target_layers": [20, 40, 60, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
        },
        "120": {
            "target_layers": [5, 15, 25, 35, 45, 55, 65, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
        },
        "204": {
            "target_layers": [5, 15, 25, 35, 45, 55, 65, -1],
            "uncertainty_heads": [(15, 24), (7, 16), (31, 8),
                                  (23, 0), (23, 28), (23, 9)],
            "attention_layers": (list(range(0, 4)) + list(range(35, 39))
                                 + list(range(76, 80))),
            "drift_heads": [0, 16, 32, 48],
            "rollout_checkpoints": [3, 38, 79],
        },
    },
}

EXPECTED_FEATURES = {"93": 93, "120": 120, "204": 204}


def apply_preset(config, model, script, override=None):
    """Fill CONFIG with indices matching the loaded model, then validate."""
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    if n_layers not in PRESETS:
        raise ValueError(
            f"no preset for a {n_layers}-layer model; add one to PRESETS "
            "rather than reusing indices from another depth")

    config.update(PRESETS[n_layers][script])
    if override:
        config.update(override)

    _validate(config, n_layers, n_heads, script)
    print(f"  preset: {n_layers} layers, {n_heads} heads")
    print(f"    target_layers     {config['target_layers']}")
    if "attention_layers" in config:
        print(f"    attention_layers  {len(config['attention_layers'])} layers, "
              f"{len(config['attention_layers']) - 1} adjacent pairs")
        print(f"    drift_heads       {config['drift_heads']}")
        print(f"    rollout           {config['rollout_checkpoints']}")
    return config


def _validate(config, n_layers, n_heads, script):
    def resolve(layers):
        return [n_layers - 1 if l == -1 else l for l in layers]

    target = resolve(config["target_layers"])
    if max(target) >= n_layers:
        raise ValueError(f"target_layers {target} exceed {n_layers} layers")
    if max(target) < n_layers - 2:
        raise ValueError(
            f"target_layers {target} stop at layer {max(target)} of "
            f"{n_layers}; the deepest layers would never be sampled")

    for layer, head in config["uncertainty_heads"]:
        if layer >= n_layers:
            raise ValueError(f"uncertainty head layer {layer} >= {n_layers}")
        if head >= n_heads:
            raise ValueError(f"uncertainty head index {head} >= {n_heads}")

    if script != "204":
        return

    attn = config["attention_layers"]
    if max(attn) >= n_layers:
        raise ValueError(f"attention_layers exceed {n_layers} layers")
    if len(attn) != 12:
        raise ValueError(
            f"attention_layers has {len(attn)} entries; the 204 set assumes "
            "three bands of four, giving 11 adjacent pairs")
    if max(config["drift_heads"]) >= n_heads:
        raise ValueError(f"drift_heads exceed {n_heads} heads")
    if len(config["rollout_checkpoints"]) != 3:
        raise ValueError("rollout_checkpoints should hold one layer per band")


def check_feature_count(features, script, n_layers):
    """Warn if a feature vector does not have the expected length."""
    expected = EXPECTED_FEATURES.get(script)
    if expected is None or not features:
        return
    n = len(features[0])
    if n != expected:
        print(f"  WARNING: {n} features per token, expected {expected} "
              f"(script {script}, {n_layers} layers)")
    else:
        print(f"  {n} features per token")
