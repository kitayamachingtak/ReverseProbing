import argparse
import importlib
import json
import os
import pickle
import sys

import numpy as np

import config as cfg

sys.path.insert(0, cfg.SCRIPTS_DIR)


def load_model(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"loading {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map="auto", torch_dtype=torch.float16,
        trust_remote_code=True, attn_implementation="eager")
    model.config.output_attentions = True
    print(f"  {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads")
    return model, tokenizer


def split_data(path):
    data = [json.loads(line) for line in open(path)]
    order = np.random.RandomState(cfg.SEED).permutation(len(data))
    k = int(cfg.TRAIN_RATIO * len(data))
    return [data[i] for i in order[:k]], [data[i] for i in order[k:]]


def run_one(mod, config, model, tokenizer, embed_model, dataset):
    from model_presets import apply_preset

    path = cfg.data_path(dataset)
    train_data, test_data = split_data(path)
    print(f"  train {len(train_data)}  test {len(test_data)}")

    mod.CONFIG["data_file"] = path
    mod.CONFIG["neighbor_windows"] = cfg.NEIGHBOR_WINDOWS[config]
    apply_preset(mod.CONFIG, model, script=config)

    token_freq, token_idf = mod.build_vocabulary_stats(train_data, tokenizer)
    shared = [model, tokenizer, embed_model, token_freq, token_idf, {}]

    if hasattr(mod, "build_medical_cache"):
        from medical_detector import MedicalTermDetector
        detector = MedicalTermDetector(
            meddra_dir=mod.CONFIG.get("meddra_dir"),
            use_umls_linker=mod.CONFIG.get("use_umls_linker", False),
            spacy_model=mod.CONFIG.get("spacy_model"))
        train = mod.process_dataset(
            train_data, *shared, mod.build_medical_cache(train_data, detector), "TRAIN")
        test = mod.process_dataset(
            test_data, *shared, mod.build_medical_cache(test_data, detector), "TEST")
    else:
        train = mod.process_dataset(train_data, *shared, "TRAIN")
        test = mod.process_dataset(test_data, *shared, "TEST")
    return train, test


def save(path, features, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"features": features, "labels": labels,
                     "feature_names": list(features[0].keys()) if features else []}, f)
    print(f"  saved {len(features)} vectors -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=cfg.MODEL_ORDER)
    ap.add_argument("--config", nargs="*", default=list(cfg.FEATURE_CONFIGS),
                    choices=list(cfg.FEATURE_CONFIGS))
    ap.add_argument("--dataset", nargs="*", default=list(cfg.DATA_FILES),
                    choices=list(cfg.DATA_FILES))
    ap.add_argument("--force", action="store_true", help="rerun existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="list pending jobs only")
    args = ap.parse_args()

    todo = [(c, d) for c in args.config for d in args.dataset
            if args.force or not all(
                os.path.exists(p) for p in cfg.feature_paths(args.model, c, d))]
    skipped = len(args.config) * len(args.dataset) - len(todo)

    print(f"model={args.model}  todo={len(todo)}  skipped={skipped}")
    for c, d in todo:
        print(f"  - {c} / {d}")
    if args.dry_run or not todo:
        return

    model, tokenizer = load_model(cfg.model_path(args.model))
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(cfg.EMBED_MODEL)

    for i, (config, dataset) in enumerate(todo, 1):
        mod_name = cfg.module_for(args.model, config)
        print(f"\n[{i}/{len(todo)}] {args.model} | {config} | {dataset}  ({mod_name})")

        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)

        (train_f, train_l), (test_f, test_l) = run_one(
            mod, config, model, tokenizer, embed_model, dataset)

        n = len(train_f[0]) if train_f else 0
        expected = cfg.expected_features(args.model, config)
        if n != expected:
            print(f"  WARNING: got {n} features, expected {expected}")

        train_out, test_out = cfg.feature_paths(args.model, config, dataset)
        save(train_out, train_f, train_l)
        save(test_out, test_f, test_l)

    print("\ndone")


if __name__ == "__main__":
    main()
