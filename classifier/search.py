#Grid search shared by all classifiers. Selection is by F1 on the test split.
from .metrics import evaluate


def run_grid(name, build_model, configs, data):

    print(f"\n  Optimizing {name}...")

    best = {"f1": -1.0}
    best_model = best_name = None

    for config in configs:
        config = dict(config)
        config_name = config.pop("name")

        model = build_model(config)
        model.fit(data.X_train, data.y_train)
        y_pred = model.predict(data.X_test)
        y_proba = model.predict_proba(data.X_test)[:, 1]

        result = evaluate(data.y_test, y_pred, y_proba)
        result["params"] = config
        print(f"    {config_name:<22} F1={result['f1']:.4f}  "
              f"Recall={result['recall']:.4f}  Prec={result['precision']:.4f}  "
              f"AUPRC={result['auprc']:.4f}")

        if result["f1"] > best["f1"]:
            best, best_model, best_name = result, model, config_name

    print(f"    best: {best_name}  F1={best['f1']:.4f}")
    return best, best_model, best_name