#python train_classifiers.py --model llama3.1-70b --config max --dataset mimic
#python train_classifiers.py --model mistral-7b

import argparse, json, os
from classifier import CLASSIFIERS, Dataset, importance

FEATURE_DIR = "features_by_script"
OUT_DIR = "outputs"


def paths(model, config, dataset):
    d = os.path.join(FEATURE_DIR, model)
    tag = f"features_{config}"
    return (os.path.join(d, f"{tag}_train_{dataset}.pkl"),
            os.path.join(d, f"{tag}_test_{dataset}.pkl"))


def run(model, config, dataset, classifiers, top_n):
    train_pkl, test_pkl = paths(model, config, dataset)
    if not (os.path.exists(train_pkl) and os.path.exists(test_pkl)):
        print(f"skip {model}/{config}/{dataset}: features not found")
        return None

    print("\n" + "#" * 88)
    print(f"# {model} | {config} | {dataset}")
    print("#" * 88)

    data = Dataset(train_pkl, test_pkl)
    print(data.summary())

    results, importances = {}, {}
    for name in classifiers:
        result, model_obj, config_name = CLASSIFIERS[name].train(data)
        result["best_config"] = config_name
        results[name] = result
        importances[name] = {
            "importance_type": importance.METRIC[name],
            "importance": importance.report(
                name, model_obj, data.feature_names, top_n, result["auprc"]),
        }

    print("\n" + "=" * 72)
    print(f'{"Classifier":<22} {"F1":>8} {"Recall":>8} {"Prec":>8} '
          f'{"AUC-ROC":>10} {"AUPRC":>8}')
    print("-" * 72)
    for name, r in results.items():
        print(f'{name:<22} {r["f1"]:>8.4f} {r["recall"]:>8.4f} '
              f'{r["precision"]:>8.4f} {r["auc"]:>10.4f} {r["auprc"]:>8.4f}')

    stem = f"{model}_{config}_{dataset}"
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"metrics_{stem}.json"), "w") as f:
        json.dump({k: {m: v for m, v in r.items() if m != "params"}
                   for k, r in results.items()}, f, indent=2)
    with open(os.path.join(OUT_DIR, f"importance_{stem}.json"), "w") as f:
        json.dump(importances, f)
    print(f"\nwrote outputs/metrics_{stem}.json and outputs/importance_{stem}.json")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", nargs="*", default=["93", "120", "204", "max"])
    ap.add_argument("--dataset", nargs="*", default=["mimic", "generated"])
    ap.add_argument("--classifiers", nargs="*", default=list(CLASSIFIERS),
                    choices=list(CLASSIFIERS))
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    for config in args.config:
        for dataset in args.dataset:
            run(args.model, config, dataset, args.classifiers, args.top_n)


if __name__ == "__main__":
    main()