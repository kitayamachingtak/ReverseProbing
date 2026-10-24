# Reverse Probing

Token-level uncertainty quantification for clinical summaries. 

Paper: *Reverse Probing: Supervised Token-level Uncertainty Quantification for
Large Language Models in Clinical Text*

## Setup
 
```bash
pip install -r requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
```
 
The scispaCy NER model is a separate download and is not on PyPI, so any
one of them will run, but the paper's numbers use `en_ner_bc5cdr_md`.
 
UMLS and MedDRA both need a
licence.
 
Tested with Python 3.10, CUDA 12, `numpy<2`. numpy 2.x breaks scispaCy 0.6.2.
 
This work accesses the internal activation of LLMs, so API is not useful here. Models here is resolved through the normal Hugging Face cache. 
 
```bash
export HF_HUB_CACHE=/path/to/huggingface
```
 
For a model outside that cache, set `MODEL_PATH_<KEY>` instead, upper case with
dots and dashes replaced by underscores:
 
```bash
export MODEL_PATH_LLAMA3_1_70B=/path/to/snapshot
```


### Data

Both datasets come from
[Hegselmann et al. (2025)](https://physionet.org/content/ann-pt-summ/) and are
derived from MIMIC-IV-Note. Access requires PhysioNet credentialing. Place the two files in `mimic/`:

```
mimic/hallucinations_mimic_di.jsonl
mimic/hallucinations_generated_di.jsonl
```

## Files

Three stages, in order. 

```bash
# 1. extract features 
python extract_features.py --model mistral-7b

# 2. train classifiers and dump metrics + feature importance
python train_classifiers.py --model mistral-7b

# 3. redraw every figure from outputs/
bash make_figures.sh
```

To reproduce the whole paper, loop over the six models:

```bash
for m in mistral-7b biomistral-7b llama3.1-8b openbiollm-8b \
         llama3.1-70b openbiollm-70b; do
    python extract_features.py --model "$m"
    python train_classifiers.py --model "$m"
done
bash make_figures.sh
```

## Layout

```
config.py                      
extract_features.py            
train_classifiers.py           
make_figures.sh               

different_feature_numbers/     feature extraction, one script per feature count
    model_presets.py           config
    medical_detector.py        medical entity detection
    collect_token_features_93.py
    collect_token_features_120.py
    collect_token_features_204.py
    collect_token_features_454.py    7-8B models
    collect_token_features_886.py     70B models

classifier/                    training and evaluation
    data.py                  
    metrics.py                
    search.py                  hyperparameter grid, selection by F1
    xgboost_clf.py            
    catboost_clf.py
    logistic_clf.py
    importance.py             

draw_image/                 
    common.py                  
    fig_auprc_bars.py          AUPRC across base LLMs
    fig_feature_curve.py       F1 against feature count
    fig_importance_heatmap.py  importance per family, all three classifiers

mimic/                         put datasets here
features_by_script/            collected features
outputs/                      with baseline results
figures/                      
```

### baselines.json

Baseline scores are not produced by this repository, so `outputs/baselines.json`
is committed directly. 
Baselines are adaptations of
published methods, described in Section 3.4 of the paper.

## Figures

```bash
python draw_image/fig_auprc_bars.py
python draw_image/fig_feature_curve.py --metric f1
python draw_image/fig_importance_heatmap.py --classifier XGBoost
python draw_image/fig_importance_heatmap.py --classifier CatBoost
python draw_image/fig_importance_heatmap.py --classifier LogisticRegression \
    --groups signal --config max
```

`--groups family` gives the eight feature families, `--groups signal` collapses
them into the four signal categories used in the paper. Cells show the share of
total importance carried by the top 20 features of each group, so a column sums
to the top-20 coverage rather than to 100 percent.

## Notes

Splits are at the case level, with each BHC-summary pair kept whole, 80/20,
seed 42. Both datasets are heavily imbalanced: unsupported tokens are 7.33% of
MIMIC-DI and 2.05% of Generated-DI, which is why AUPRC rather than AUCROC is
the primary metric throughout.


## Citation

```bibtex
@inproceedings{xiao-etal-2026-reverse,
    title = "Reverse Probing: Supervised Token-level Uncertainty Quantification for Large Language Models in Clinical Text",
    author = "Xiao, Bushi  and
      Soni, Sarvesh  and
      Wang, Daisy Zhe",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2026",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-emnlp.407/",
    pages = "8599--8616",
    ISBN = "979-8-89176-462-0",
    abstract = "As large language models are increasingly deployed for clinical text, ensuring they can reliably signal their own uncertainty becomes critical. Most existing uncertainty quantification (UQ) methods are designed for open-domain generation and cannot localize uncertainty at the token or span level in long clinical text. We propose Reverse Probing, the first uncertainty quantification framework specialized for clinical summarization, which estimates token-level uncertainty directly from pre-existing clinical summaries. Rather than sampling new outputs, Reverse Probing treats the text as a probe into the model{'}s internal state, extracting uncertainty features from four categories of signals. We evaluate Reverse Probing on two expert-annotated clinical datasets and outperform eight adapted baselines, achieving up to $4\times$ higher AUPRC while running over $15\times$ faster than sampling-based methods. Feature analysis reveals that the log-logit difference induced by the source record and the neighborhood context are the most consistent predictors across all models. This study offers interpretable insights into how models internally respond to unsupported clinical content."
}
```

