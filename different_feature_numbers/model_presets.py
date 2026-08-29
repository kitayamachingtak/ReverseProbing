# 8B: 32 layers, 32 heads.  70B: 80 layers, 64 heads.
HEADS_8B = [(15, 24), (7, 16), (31, 8), (23, 0), (23, 28), (23, 9)]
HEADS_70B = [(17, 24), (7, 16), (77, 8), (57, 0), (57, 28), (57, 9)]
HEADS_70B_204 = [(37, 24), (17, 16), (77, 8), (57, 0), (57, 28), (57, 9)]

PRESETS = {
    32: {
        "93": {
            "target_layers": [8, 16, 24, -1],
            "uncertainty_heads": HEADS_8B,
        },
        "120": {
            "target_layers": [2, 6, 10, 14, 18, 22, 26, -1],
            "uncertainty_heads": HEADS_8B,
        },
        "204": {
            "target_layers": [2, 6, 10, 14, 18, 22, 26, -1],
            "uncertainty_heads": HEADS_8B,
            "attention_layers": (list(range(0, 4)) + list(range(14, 18))
                                 + list(range(28, 32))),
            "drift_heads": [0, 8, 16, 24],
            "rollout_checkpoints": [3, 17, 31],
        },
    },
    80: {
        "93": {
            "target_layers": [20, 40, 60, -1],
            "uncertainty_heads": HEADS_70B,
        },
        "120": {
            "target_layers": [5, 15, 25, 35, 45, 55, 65, -1],
            "uncertainty_heads": HEADS_70B,
        },
        "204": {
            "target_layers": [5, 15, 25, 35, 45, 55, 65, -1],
            "uncertainty_heads": HEADS_70B_204,
            "attention_layers": (list(range(0, 8)) + list(range(35, 43))
                                 + list(range(70, 80))),
            "drift_heads": [0, 16, 32, 48],
            "rollout_checkpoints": [3, 17, 31],
        },
    },
}


def apply_preset(config, model, script, override=None):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    if n_layers not in PRESETS:
        raise ValueError(
            f"no preset for a {n_layers}-layer model; add one to PRESETS "
            "rather than reusing indices from another depth")
    if script not in PRESETS[n_layers]:
        raise ValueError(f"no preset for script {script!r} at {n_layers} layers")

    config.update(PRESETS[n_layers][script])
    if override:
        config.update(override)

    _check(config, n_layers, n_heads, script)
    _describe(config, n_layers, n_heads, script)
    return config


def _check(config, n_layers, n_heads, script):
    target = [n_layers - 1 if l == -1 else l for l in config["target_layers"]]
    if max(target) >= n_layers:
        raise ValueError(f"target_layers {target} exceed {n_layers} layers")
    if max(target) < n_layers - 2:
        raise ValueError(
            f"target_layers {target} stop at layer {max(target)} of {n_layers}; "
            "the deepest layers would never be sampled, which usually means "
            "indices from a shallower model were reused")

    for layer, head in config["uncertainty_heads"]:
        if layer >= n_layers:
            raise ValueError(f"uncertainty head layer {layer} >= {n_layers}")
        if head >= n_heads:
            raise ValueError(f"uncertainty head index {head} >= {n_heads}")

    if script != "204":
        return

    if max(config["attention_layers"]) >= n_layers:
        raise ValueError(f"attention_layers exceed {n_layers} layers")
    if max(config["drift_heads"]) >= n_heads:
        raise ValueError(f"drift_heads exceed {n_heads} heads")
    if max(config["rollout_checkpoints"]) >= n_layers:
        raise ValueError(f"rollout_checkpoints exceed {n_layers} layers")


def _describe(config, n_layers, n_heads, script):
    print(f"  preset: {n_layers} layers, {n_heads} heads, script {script}")
    print(f"    target_layers      {config['target_layers']}")
    print(f"    uncertainty_heads  {len(config['uncertainty_heads'])} pairs "
          f"-> {3 * len(config['uncertainty_heads'])} attention snapshots")

    if script != "204":
        return

    attn = config["attention_layers"]
    n_drift = (len(attn) - 1) * len(config["drift_heads"])
    n_rollout = 3 * len(config["rollout_checkpoints"])
    print(f"    attention_layers   {len(attn)} layers -> {len(attn) - 1} pairs")
    print(f"    drift_heads        {config['drift_heads']} "
          f"-> {n_drift} drift features")
    print(f"    rollout            {config['rollout_checkpoints']} "
          f"-> {n_rollout} rollout features")


def report_feature_count(features, script, n_layers):
    if not features:
        return
    n = len(features[0])
    print(f"  {n} features per token (script {script}, {n_layers} layers)")
    return n
