# CrossHal: Cross-Lingual and Cross-Domain Hallucination Detection in Arabic LLMs

> **Extending HalluShift to evaluate cross-lingual and cross-domain generalization of hallucination detection across six Arabic large language models.**

---

## Overview

CrossHal investigates whether internal-state hallucination signals — hidden state trajectories, attention patterns, and token probability features — generalize across languages and domains in Arabic LLMs. We extend the HalluShift framework to evaluate six models on three datasets under nine experimental conditions spanning monolingual, cross-lingual, cross-domain, and combined transfer settings.

A central finding is that cross-lingual transfer is governed by two independent factors: **class separability** within each language and **language alignment** across languages. Models like Qwen2.5-14B achieve near-perfect language alignment but near-zero class separability, while ALLaM-7B achieves within-language separability but fails to align across languages — both fail cross-lingual transfer for opposite reasons.

---

## Models

| Model | Parameters | HuggingFace ID |
|---|---|---|
| ALLaM-7B | 7B | `ALLaM-AI/ALLaM-7B-Instruct-preview` |
| Aya-23-8B | 8B | `CohereLabs/aya-23-8B` |
| Mistral-8B | 8B | `mistralai/Ministral-8B-Instruct-2410` |
| Phi4-mini | ~4B | `microsoft/Phi-4-mini-instruct` |
| Qwen2.5-14B | 14B | `Qwen/Qwen2.5-14B-Instruct` |
| Silma-9B | 9B | `silma-ai/SILMA-9B-Instruct-v1.0` |

---

## Datasets

| Dataset | Language | Domain | Size (train / test) | Label Source |
|---|---|---|---|---|
| TruthfulQA EN | English | Factual QA | 613 / 204 | GPT-4o-as-judge |
| TruthfulQA AR | Arabic | Factual QA (translated) | 613 / 204 | GPT-4o-as-judge |
| HalluScore (AraHalluQA) | Arabic | Diverse Arabic QA | varies | GPT-4o-as-judge |

TruthfulQA is stratified 75/25 by question category. AraHalluQA is loaded from pre-split files.

---

## Experiments

Nine transfer conditions are evaluated for each model:

| Exp | Train | Test | Type |
|---|---|---|---|
| 1 | TQA EN | TQA EN | Monolingual |
| 2 | TQA AR | TQA AR | Monolingual |
| 3 | HS | HS | Monolingual |
| 4 | TQA EN | TQA AR | Cross-lingual |
| 5 | TQA EN | HS | Cross-lingual + domain |
| 6 | HS | TQA EN | Cross-lingual + domain |
| 7 | TQA AR | HS | Cross-domain |
| 8 | HS | TQA AR | Cross-domain |
| 9 | TQA AR | TQA EN | Cross-lingual |

---

## Pipeline

```
Data Preparation
    ↓
LLM Generation (greedy decoding, max 128 tokens)
    ↓
Internal State Feature Extraction
  ├── Hidden state Wasserstein distance per layer pair (averaged across tokens)
  ├── Hidden state cosine similarity per layer pair
  ├── Attention Wasserstein + cosine per layer pair
  └── Token probability features:
        norm_entropy_max/min, low_prob_count_max/min,
        mean_grad_max/min, p25_max, p50_max, p75_max
    ↓
GPT-4o-as-Judge Hallucination Labelling
    ↓
Feature Engineering + Normalisation
  ├── Same-language experiments: StandardScaler fit on train
  └── Cross-lingual/domain experiments: QuantileTransformer fit independently on train and test
    ↓
HalluShiftNet Classifier Training (9 experiments × 6 models)
    ↓
Evaluation: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC
```

Features are saved as `.parquet` files — LLM inference only needs to run once. All classifiers train directly from the parquet files.

---

## Classifier: HalluShiftNet

A lightweight 3-layer MLP trained on the extracted features:

```
Input features → FC(64) → ReLU → Dropout(0.3)
              → FC(32) → ReLU → Dropout(0.3)
              → FC(1)  → Sigmoid
```

**Training:** BCEWithLogitsLoss + pos_weight for class imbalance, Adam optimiser (lr=1e-3), threshold tuning on validation split to maximise F1.

**Three variants reported:**
- `no_imbalance` — no imbalance handling, threshold=0.5
- `pos_weight` — pos_weight only, threshold=0.5
- `pos_weight_tuned` — pos_weight + tuned threshold (primary metric)

---

## Repository Structure

```
CrossHal/
│
├── extract_features.py       # LLM inference + feature extraction + GPT-4o labelling
├── functions.py              # Core HalluShift feature utilities
├── prepare_data.py           # Dataset loading and splitting
│
├── classifier.py             # HalluShiftNet baseline
├── crosslingual_eval.py      # Run any of the 9 experiments
│
├── trajectory_all_comparisons.py   # Layer-wise trajectory figures (6 PDFs)
├── combined_trajectory_figure.py   # Combined 3×6 trajectory figure
├── cross_domain_2x6.py             # 2×6 cross-domain figure
├── layerwise_3x6.py                # 3×6 layer-wise analysis figure
│
├── tsne_6x6.py               # 6×6 t-SNE (rows: per-dataset label, language)
├── tsne_4x6.py               # 4×6 t-SNE
├── tsne_2x6.py               # 2×6 t-SNE
│
├── kde_distributions.py      # KDE distributions (top 4 features, all models)
├── kde_union_features.py     # KDE with union of top features per model
├── sankey_all_models.py      # Sankey flow diagrams (EN → AraHalluQA)
│
└── results_table.tex         # LaTeX results table
    results_table_colored.tex # LaTeX table with top-3 coloring
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/crosshal.git
cd crosshal

pip install torch transformers openai pandas numpy scipy \
            scikit-learn matplotlib plotly pyarrow openpyxl
```

For feature extraction (requires GPU):

```bash
pip install accelerate bitsandbytes
```

---

## Usage

### 1. Feature Extraction (one-time, requires GPU + API keys)

```python
python extract_features.py \
    --model_name   phi4_mini \
    --dataset_name truthfulqa \
    --csv_path     ./data/truthfulqa_en_train.csv \
    --output_dir   ./results/tqa_en_train
```

Set your keys in `extract_features.py`:
```python
HF_TOKEN   = "your_huggingface_token"
OPENAI_KEY = "your_openai_key"
```

Supported model names: `allam_7B`, `aya_8B`, `mistral_8B`, `phi4_mini`, `qwen2.5_14B`, `silma_9B`

Supported dataset names: `truthfulqa`, `truthfulqa_ar`, `arahalluqa`

### 2. Run an Experiment

```python
python crosslingual_eval.py \
    --experiment     4 \
    --train_features ./results/tqa_en_train/truthfulqa_phi4_mini_features.parquet \
    --test_features  ./results/tqa_ar_test/truthfulqa_ar_phi4_mini_features.parquet \
    --train_meta     ./results/tqa_en_train/meta.json \
    --output_dir     ./results/exp4_en_to_ar
```

Or from Python:

```python
from classifier import train_crosslingual_model
import pandas as pd

train_df = pd.read_parquet('results/tqa_en_train/truthfulqa_phi4_mini_features.parquet')
test_df  = pd.read_parquet('results/tqa_ar_test/truthfulqa_ar_phi4_mini_features.parquet')

_, results = train_crosslingual_model(
    train_df   = train_df,
    test_df    = test_df,
    num_layers = 32,
    epochs     = 50,
    save_dir   = './results/exp4_en_to_ar',
)
print(results)
```

---

## Google Drive Data Structure

```
hallushift/
├── data/
│   ├── truthfulqa_en_train.csv
│   ├── truthfulqa_en_test.csv
│   ├── truthfulqa_ar_train.csv
│   ├── truthfulqa_ar_test.csv
│   ├── arahalluqa_train.csv
│   └── arahalluqa_test.csv
│
├── results_allam_latest/
│   ├── tqa_en_train_allam/truthfulqa_allam_7B_features.parquet
│   ├── tqa_en_test_allam/truthfulqa_allam_7B_features.parquet
│   ├── tqa_ar_train_allam/truthfulqa_ar_allam_7B_features.parquet
│   ├── tqa_ar_test_allam/truthfulqa_ar_allam_7B_features.parquet
│   ├── arahalluqa_train_allam/arahalluqa_allam_7B_features.parquet
│   └── arahalluqa_test_allam/arahalluqa_allam_7B_features.parquet
│
├── results_aya_latest/       (same structure, suffix _aya)
├── results_mistral_latest/   (same structure, suffix _mistral)
├── results_phi4_mini_latest/ (same structure, suffix _phi)
├── results_qwen2.5_latest/   (same structure, suffix _qwen)
└── results_silma_latest/     (same structure, suffix _silma)
```
