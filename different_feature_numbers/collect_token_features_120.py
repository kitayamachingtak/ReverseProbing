#!/usr/bin/env python3

# Token-level feature collection (120-feature configuration).
# Extends the 93-feature set: medical detection replaced with scispaCy NER + UMLS/MedDRA lookup (is_medical, ner_entity_type, medical_confidence), and the neighbourhood window set extended to {2, 3, 5, 7}. 


import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import math

from medical_detector import MedicalTermDetector


from model_presets import apply_preset, check_feature_count


CONFIG = {
    'data_file': 'hallucinations_mimic_di.jsonl',

    'train_output': 'train_token_features.pkl',
    'test_output':  'test_token_features.pkl',

    'gen_model_name': 'meta-llama/Llama-3.1-8B-Instruct',

    'use_multi_gpu': True,
    'max_memory_per_gpu': None,

    'embed_model_name': 'emilyalsentzer/Bio_ClinicalBERT',

    'max_samples': None,
    'train_test_split': 0.8,
    'random_seed': 42,

    'neighbor_windows': [2, 3, 5, 7],
    'top_k_values': [5, 10, 20],


    'meddra_dir': None,
    'use_umls_linker': False,
    'spacy_model': None,
}


def build_medical_cache(data_list: list, detector: MedicalTermDetector) -> dict:
    print("  Running document-level medical NER (one-time)...")
    cache = {}
    for idx, data in enumerate(data_list):
        full_text  = data['text'] + "\n" + data['summary']
        cache[idx] = detector.analyze_document(full_text)
        if (idx + 1) % 50 == 0:
            print(f"    NER progress: {idx+1}/{len(data_list)}")
    print(f"  Medical NER cache built for {len(data_list)} documents")
    return cache


_ENTITY_TYPE_MAP = {
    'O': 0,
    'VOCAB': 1,
    'CHEMICAL': 2,
    'DISEASE': 3,
    'GENE_OR_GENE_PRODUCT': 4,
    'CANCER': 5,
    'CELL_TYPE': 6,
    'ORGAN': 7,
    'TISSUE': 8,
    'ORGANISM_SUBSTANCE': 9,
    'SIMPLE_CHEMICAL': 10,
    'PATHOLOGICAL_FORMATION': 11,
}

def _encode_entity_type(etype: str) -> int:
    return _ENTITY_TYPE_MAP.get(etype, 99)


def get_baseline_logit_simple(token_id, model, tokenizer):
    bos       = tokenizer.bos_token_id if tokenizer.bos_token_id else 1
    input_ids = torch.tensor([[bos, token_id]]).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits  = outputs.logits[0, 0, :]
    probs     = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return {
        'prob':    probs[token_id].item(),
        'entropy': -(probs * log_probs).sum().item(),
    }


def build_vocabulary_stats(data_list, tokenizer):
    print("  Building vocabulary statistics...")
    token_freq         = Counter()
    doc_contains_token = {}

    for data in data_list:
        token_ids = tokenizer.encode(data['summary'], add_special_tokens=False)
        token_freq.update(token_ids)
        for tid in set(token_ids):
            doc_contains_token[tid] = doc_contains_token.get(tid, 0) + 1

    total_docs = len(data_list)
    token_idf  = {
        tid: math.log(total_docs / (df + 1))
        for tid, df in doc_contains_token.items()
    }
    return token_freq, token_idf


def extract_all_token_features_single_forward(
    bhc_text, summary_text,
    gen_model, tokenizer, embed_model,
    token_freq, token_idf, baseline_cache,
    medical_annotations: dict,
):
    summary_token_ids = tokenizer.encode(summary_text, add_special_tokens=False)

    MAX_SUMMARY_TOKENS = 1024
    if len(summary_token_ids) > MAX_SUMMARY_TOKENS:
        print(f"    Warning: summary too long ({len(summary_token_ids)} tokens), truncating")
        summary_token_ids = summary_token_ids[:MAX_SUMMARY_TOKENS]
        summary_text      = tokenizer.decode(summary_token_ids)

    full_text    = f"Brief Hospital Course:\n{bhc_text}\n\nSummary:\n{summary_text}"
    bhc_prefix   = f"Brief Hospital Course:\n{bhc_text}\n\nSummary:\n"
    bhc_length   = tokenizer(bhc_prefix, return_tensors="pt")["input_ids"].shape[1]
    full_inputs  = tokenizer(full_text,  return_tensors="pt").to(gen_model.device)

    summary_only   = f"Summary:\n{summary_text}"
    summary_inputs = tokenizer(summary_only, return_tensors="pt").to(gen_model.device)
    summary_start  = tokenizer("Summary:\n", return_tensors="pt")["input_ids"].shape[1]

    print("    Forward pass with BHC...")
    with torch.no_grad():
        outputs_with = gen_model(
            full_inputs["input_ids"],
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )
        logits_with        = outputs_with.logits[0, bhc_length-1 : bhc_length-1+len(summary_token_ids)]
        hidden_states_with = outputs_with.hidden_states
        attentions_with    = outputs_with.attentions

    print("    Forward pass without BHC...")
    with torch.no_grad():
        outputs_without = gen_model(summary_inputs["input_ids"], return_dict=True)
        logits_without  = outputs_without.logits[
            0, summary_start-1 : summary_start-1+len(summary_token_ids)
        ]

    print(f"    Extracting features for {len(summary_token_ids)} tokens...")

    features_list = []
    total_tokens  = sum(token_freq.values()) if token_freq else 1
    num_layers    = len(hidden_states_with)

    for i in range(len(summary_token_ids)):
        token_id  = summary_token_ids[i]
        token_str = tokenizer.decode([token_id]).strip().lower()
        features  = {}

        logits    = logits_with[i]
        probs     = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        entropy    = -(probs * log_probs).sum().item()
        vocab_size = len(logits)
        features['entropy']            = entropy
        features['normalized_entropy'] = entropy / np.log(vocab_size)
        features['max_prob']           = probs.max().item()
        features['current_prob']       = probs[token_id].item()

        top2 = torch.topk(probs, k=2).values
        features['margin'] = (top2[0] - top2[1]).item()
        features['ratio']  = (top2[0] / (top2[1] + 1e-10)).item()

        features['topk_cumulative'] = torch.topk(probs, k=10).values.sum().item()

        sorted_probs, _ = torch.sort(probs, descending=True)
        n     = len(sorted_probs)
        index = torch.arange(1, n + 1, dtype=torch.float32).to(probs.device)
        features['gini'] = (
            2 * torch.sum(index * sorted_probs) / (n * torch.sum(sorted_probs)) - (n + 1) / n
        ).item()

        features['perplexity'] = torch.exp(-log_probs[token_id]).item()
        max_logit = logits.max()
        features['energy'] = -(
            max_logit + torch.log(torch.sum(torch.exp(logits - max_logit)))
        ).item()

        logits_no    = logits_without[i]
        probs_no     = torch.nn.functional.softmax(logits_no, dim=-1)
        log_probs_no = torch.nn.functional.log_softmax(logits_no, dim=-1)
        entropy_no   = -(probs_no * log_probs_no).sum().item()

        features['delta_prob']    = probs[token_id].item() - probs_no[token_id].item()
        features['delta_entropy'] = entropy - entropy_no
        features['delta_energy']  = features['energy'] - (
            -(logits_no.max() + torch.log(torch.sum(torch.exp(logits_no - logits_no.max())))).item()
        )

        for k in CONFIG['top_k_values']:
            topk_probs, topk_indices = torch.topk(probs, k=k)

            rank = 0
            for r, tid in enumerate(topk_indices):
                if tid == token_id:
                    rank = r + 1
                    break
            features[f'rank_top{k}'] = rank
            features[f'in_top{k}']   = float(rank > 0)

            if token_str and token_str.strip():
                topk_tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]
                try:
                    orig_emb     = embed_model.encode([token_str],  show_progress_bar=False)
                    topk_embs    = embed_model.encode(topk_tokens,  show_progress_bar=False)
                    similarities = cosine_similarity(orig_emb, topk_embs)[0]

                    features[f'max_sim_top{k}']  = float(np.max(similarities))
                    features[f'avg_sim_top{k}']  = float(np.mean(similarities))
                    features[f'top3_sim_top{k}'] = float(np.mean(similarities[:min(3, len(similarities))]))
                    features[f'sim_std_top{k}']  = float(np.std(similarities))

                    sem_rank = 0
                    for r, sim in enumerate(similarities):
                        if sim > 0.7:
                            sem_rank = r + 1
                            break
                    features[f'semantic_rank_top{k}'] = sem_rank
                except Exception:
                    features[f'max_sim_top{k}']       = 0.0
                    features[f'avg_sim_top{k}']       = 0.0
                    features[f'top3_sim_top{k}']      = 0.0
                    features[f'sim_std_top{k}']       = 0.0
                    features[f'semantic_rank_top{k}'] = 0
            else:
                features[f'max_sim_top{k}']       = 0.0
                features[f'avg_sim_top{k}']       = 0.0
                features[f'top3_sim_top{k}']      = 0.0
                features[f'sim_std_top{k}']       = 0.0
                features[f'semantic_rank_top{k}'] = 0

        for window in CONFIG['neighbor_windows']:
            neighbor_probs = []
            medical_count  = 0

            for j in range(max(0, i - window), min(len(summary_token_ids), i + window + 1)):
                if j == i or j >= len(logits_with):
                    continue
                n_tid  = summary_token_ids[j]
                n_dist = torch.nn.functional.softmax(logits_with[j], dim=-1)
                neighbor_probs.append(n_dist[n_tid].item())

                n_str = tokenizer.decode([n_tid]).strip().lower()
                if n_str in medical_annotations:
                    medical_count += 1

            neighbor_avg = np.mean(neighbor_probs) if neighbor_probs else features['current_prob']
            neighbor_std = np.std(neighbor_probs)  if neighbor_probs else 0.0

            window_size = min(len(summary_token_ids), i + window + 1) - max(0, i - window)
            features[f'neighbor_avg_w{window}']       = neighbor_avg
            features[f'neighbor_std_w{window}']       = neighbor_std
            features[f'isolation_w{window}']          = features['current_prob'] - neighbor_avg
            features[f'relative_isolation_w{window}'] = (
                (features['current_prob'] - neighbor_avg) / (neighbor_std + 1e-10)
            )
            features[f'medical_density_w{window}']   = (
                medical_count / window_size if window_size > 0 else 0.0
            )

        ann = medical_annotations.get(token_str)
        if ann:
            features['is_medical']         = 1.0
            features['ner_entity_type']    = _encode_entity_type(ann.get('entity_type', 'O'))
            source = ann.get('source', 'vocab')
            features['medical_confidence'] = {
                'ner+vocab': 1.0, 'ner': 0.85, 'vocab': 0.6
            }.get(source, 0.5)
        else:
            features['is_medical']         = 0.0
            features['ner_entity_type']    = 0
            features['medical_confidence'] = 0.0

        if token_freq:
            freq = token_freq.get(token_id, 0)
            features['freq']            = freq
            features['freq_normalized'] = freq / total_tokens
            features['freq_log']        = math.log(freq + 1)
            features['idf']             = token_idf.get(token_id, math.log(len(token_freq)))
            features['rarity']          = 1.0 / (freq + 1)

        if token_id not in baseline_cache:
            baseline_cache[token_id] = get_baseline_logit_simple(token_id, gen_model, tokenizer)
        baseline = baseline_cache[token_id]
        features['baseline_prob']    = baseline['prob']
        features['baseline_entropy'] = baseline['entropy']

        actual_idx = bhc_length - 1 + i

        for layer_idx in CONFIG['target_layers']:
            if layer_idx == -1:
                layer_idx = num_layers - 1
            if layer_idx < num_layers and actual_idx < hidden_states_with[layer_idx].shape[1]:
                h = hidden_states_with[layer_idx][0, actual_idx, :]
                features[f'hidden_norm_l{layer_idx}'] = torch.norm(h).item()
                features[f'hidden_mean_l{layer_idx}'] = h.mean().item()
                features[f'hidden_std_l{layer_idx}']  = h.std().item()

        layers = [l if l != -1 else num_layers - 1 for l in CONFIG['target_layers']]
        for idx in range(len(layers) - 1):
            l1, l2 = layers[idx], layers[idx + 1]
            if l1 < num_layers and l2 < num_layers and actual_idx < hidden_states_with[l1].shape[1]:
                h1 = hidden_states_with[l1][0, actual_idx, :]
                h2 = hidden_states_with[l2][0, actual_idx, :]
                features[f'layer_change_l{l1}_to_l{l2}'] = torch.norm(h2 - h1).item()
                features[f'layer_cosine_l{l1}_to_l{l2}'] = (
                    torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0)).item()
                )

        if attentions_with:
            for layer_idx, head_idx in CONFIG['uncertainty_heads']:
                if layer_idx < len(attentions_with) and actual_idx < attentions_with[layer_idx].shape[2]:
                    attn      = attentions_with[layer_idx][0, head_idx, actual_idx, :]
                    attn_dist = attn / (attn.sum() + 1e-10)
                    features[f'attn_entropy_l{layer_idx}_h{head_idx}'] = (
                        -(attn_dist * torch.log(attn_dist + 1e-10)).sum().item()
                    )
                    features[f'attn_to_bhc_l{layer_idx}_h{head_idx}'] = attn[:bhc_length].sum().item()
                    features[f'attn_max_l{layer_idx}_h{head_idx}']    = attn.max().item()

        for key, value in list(features.items()):
            if isinstance(value, float):
                if np.isinf(value):
                    features[key] = 1e10 if value > 0 else -1e10
                elif np.isnan(value):
                    features[key] = 0.0

        features_list.append(features)

    return features_list


def process_dataset(data_list, gen_model, tokenizer, embed_model,
                    token_freq, token_idf, baseline_cache,
                    medical_cache, dataset_name):

    print(f"\n[Processing {dataset_name} set...]")
    all_features, all_labels = [], []

    for idx, data in enumerate(data_list):
        print(f"  Sample {idx+1}/{len(data_list)}")

        medical_annotations = medical_cache.get(idx, {})

        features_list = extract_all_token_features_single_forward(
            data['text'], data['summary'],
            gen_model, tokenizer, embed_model,
            token_freq, token_idf, baseline_cache,
            medical_annotations,
        )

        summary_tokens = tokenizer.encode(data['summary'], add_special_tokens=False)
        token_labels   = np.zeros(len(summary_tokens))

        for label_info in data.get('labels', []):
            start_char  = label_info['start']
            end_char    = label_info['end']
            token_start = len(tokenizer.encode(data['summary'][:start_char], add_special_tokens=False))
            token_end   = len(tokenizer.encode(data['summary'][:end_char],   add_special_tokens=False))
            token_labels[token_start:token_end] = 1

        for i, feat in enumerate(features_list):
            if i < len(token_labels):
                all_features.append(feat)
                all_labels.append(token_labels[i])

    return all_features, all_labels


def main():
    print("=" * 80)
    print("OPTIMIZED FEATURE COLLECTION - scispaCy + UMLS/MedDRA medical detection")
    print("=" * 80)

    print("\n[1] Loading generation model...")
    n_gpus = torch.cuda.device_count()
    print(f"  Detected {n_gpus} GPUs")

    if CONFIG['use_multi_gpu'] and n_gpus > 1:
        device_map = "auto"
        max_memory = (
            {i: CONFIG['max_memory_per_gpu'] for i in range(n_gpus)}
            if CONFIG['max_memory_per_gpu'] else None
        )
    else:
        device_map = {"": 0}
        max_memory = None

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['gen_model_name'])
    gen_model = AutoModelForCausalLM.from_pretrained(
        CONFIG['gen_model_name'],
        device_map=device_map,
        max_memory=max_memory,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    apply_preset(CONFIG, gen_model, script="120")

    embed_model = SentenceTransformer(CONFIG['embed_model_name'])
    print("Generation and embedding models loaded")

    print("\n[2] Loading medical term detector...")
    detector = MedicalTermDetector(
        meddra_dir      = CONFIG['meddra_dir'],
        use_umls_linker = CONFIG['use_umls_linker'],
        spacy_model     = CONFIG['spacy_model'],
    )

    print("\n[3] Loading data...")
    all_data = []
    with open(CONFIG['data_file'], 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
    if CONFIG['max_samples']:
        all_data = all_data[:CONFIG['max_samples']]
    print(f"  Loaded {len(all_data)} samples")

    print(f"\n[4] Splitting ({CONFIG['train_test_split']*100:.0f}% / {(1-CONFIG['train_test_split'])*100:.0f}%)...")
    np.random.seed(CONFIG['random_seed'])
    indices     = np.random.permutation(len(all_data))
    split_point = int(CONFIG['train_test_split'] * len(all_data))
    train_data  = [all_data[i] for i in indices[:split_point]]
    test_data   = [all_data[i] for i in indices[split_point:]]
    print(f"  Train: {len(train_data)}  Test: {len(test_data)}")

    print("\n[5] Building vocabulary stats (train only)...")
    token_freq, token_idf = build_vocabulary_stats(train_data, tokenizer)
    print(f"  Vocab size: {len(token_freq)}")

    print("\n[6] Building medical annotation cache...")
    print("  Train NER")
    train_medical_cache = build_medical_cache(train_data, detector)
    print("  Test NER")
    test_medical_cache  = build_medical_cache(test_data,  detector)

    baseline_cache = {}

    print("\n[7] Collecting TRAIN features...")
    train_features, train_labels = process_dataset(
        train_data, gen_model, tokenizer, embed_model,
        token_freq, token_idf, baseline_cache,
        train_medical_cache, "TRAIN",
    )
    print(f"\n[8] Saving to {CONFIG['train_output']}")
    with open(CONFIG['train_output'], 'wb') as f:
        pickle.dump({
            'features':      train_features,
            'labels':        train_labels,
            'feature_names': list(train_features[0].keys()) if train_features else [],
            'config':        CONFIG,
        }, f)
    print(f"{len(train_features)} train vectors | pos: {int(np.sum(train_labels))} ({np.mean(train_labels)*100:.2f}%)")

    print("\n[9] Collecting TEST features...")
    test_features, test_labels = process_dataset(
        test_data, gen_model, tokenizer, embed_model,
        token_freq, token_idf, baseline_cache,
        test_medical_cache, "TEST",
    )
    print(f"\n[10] Saving to {CONFIG['test_output']}")
    with open(CONFIG['test_output'], 'wb') as f:
        pickle.dump({
            'features':      test_features,
            'labels':        test_labels,
            'feature_names': list(test_features[0].keys()) if test_features else [],
            'config':        CONFIG,
        }, f)
    print(f"{len(test_features)} test vectors | pos: {int(np.sum(test_labels))} ({np.mean(test_labels)*100:.2f}%)")

    print(f"\n{'='*80}\nCOMPLETED\n{'='*80}")


if __name__ == "__main__":
    main()
