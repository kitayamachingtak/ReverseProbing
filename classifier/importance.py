#Feature importance, with the metric fixed per classifier.

#XGBoost defaults to 'weight' (split counts), which is not comparable to gain, so importance_type is always given explicitly here.

import numpy as np

METRIC = {
    "XGBoost": "gain",
    "CatBoost": "PredictionValuesChange",
    "LogisticRegression": "abs_coef",
}


def get_importance(clf_name, model, feature_names):
    n = len(feature_names)

    if clf_name == "XGBoost":
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        keys = booster.feature_names or [f"f{i}" for i in range(n)]
        values = np.array([scores.get(k, 0.0) for k in keys], dtype=float)

    elif clf_name == "CatBoost":
        values = np.asarray(
            model.get_feature_importance(type="PredictionValuesChange"), dtype=float)

    elif clf_name == "LogisticRegression":
        values = np.abs(model.named_steps["clf"].coef_[0]).astype(float)

    else:
        raise ValueError(f"unknown classifier: {clf_name}")

    if values.shape[0] != n:
        raise ValueError(f"{clf_name}: got {values.shape[0]} importances "
                         f"for {n} features")
    return values


def report(clf_name, model, feature_names, top_n=20, auprc=None):
    """Print the top features and return the full importance vector."""
    values = get_importance(clf_name, model, feature_names)
    pct = values / (values.sum() + 1e-12) * 100
    order = np.argsort(values)[::-1]

    header = f"{clf_name}   importance_type = {METRIC[clf_name]}"
    if auprc is not None:
        header += f"   AUPRC = {auprc:.4f}"
    print("\n" + "=" * 88)
    print(header)
    print("=" * 88)
    print(f'{"Rank":<5} {"Feature":<45} {"Importance":>12} {"% Total":>10} {"Cumulative%":>12}')
    print("-" * 88)

    cumulative = 0.0
    for rank, idx in enumerate(order[:top_n], 1):
        cumulative += pct[idx]
        print(f"{rank:<5} {feature_names[idx]:<45} "
              f"{values[idx]:>12.4f} {pct[idx]:>9.2f}% {cumulative:>11.2f}%")

    print(f"\nTop {top_n} account for {pct[order[:top_n]].sum():.1f}% of total")
    print(f"Non-zero features: {int((values > 0).sum())} / {len(feature_names)}")
    return dict(zip(feature_names, values.tolist()))