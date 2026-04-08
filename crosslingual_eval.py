"""
crosslingual_eval.py – Run any of the six HalluShift experiments.

Experiments
-----------
  1  TruthfulQA EN train   →  TruthfulQA EN test      (EN baseline)
  2  TruthfulQA AR train   →  TruthfulQA AR test      (AR baseline)
  3  AraHalluQA train      →  AraHalluQA test         (AraHalluQA baseline)
  4  TruthfulQA EN train   →  TruthfulQA AR test      (cross-lingual: EN → AR)
  5  TruthfulQA EN train   →  AraHalluQA test         (cross-lingual: EN → AraHalluQA)
  6  AraHalluQA train      →  TruthfulQA EN test      (cross-lingual: AraHalluQA → EN)

All six experiments use the same train_crosslingual_model function with
separate train and test feature files — no internal random splitting.
The scaler is always fit on the training set only and applied to the test set.

Usage examples
--------------
# Experiment 1 — TruthfulQA EN baseline
python crosslingual_eval.py \\
    --experiment     1 \\
    --train_features ./results/tqa_en_train/truthfulqa_llama2_7B_features.parquet \\
    --test_features  ./results/tqa_en_test/truthfulqa_llama2_7B_features.parquet \\
    --train_meta     ./results/tqa_en_train/meta.json \\
    --output_dir     ./results/exp1_tqa_en

# Experiment 2 — TruthfulQA AR baseline
python crosslingual_eval.py \\
    --experiment     2 \\
    --train_features ./results/tqa_ar_train/truthfulqa_ar_llama2_7B_features.parquet \\
    --test_features  ./results/tqa_ar_test/truthfulqa_ar_llama2_7B_features.parquet \\
    --train_meta     ./results/tqa_ar_train/meta.json \\
    --output_dir     ./results/exp2_tqa_ar

# Experiment 3 — AraHalluQA baseline
python crosslingual_eval.py \\
    --experiment     3 \\
    --train_features ./results/arahalluqa_train/arahalluqa_llama2_7B_features.parquet \\
    --test_features  ./results/arahalluqa_test/arahalluqa_llama2_7B_features.parquet \\
    --train_meta     ./results/arahalluqa_train/meta.json \\
    --output_dir     ./results/exp3_arahalluqa

# Experiment 4 — EN train → AR test
python crosslingual_eval.py \\
    --experiment     4 \\
    --train_features ./results/tqa_en_train/truthfulqa_llama2_7B_features.parquet \\
    --test_features  ./results/tqa_ar_test/truthfulqa_ar_llama2_7B_features.parquet \\
    --train_meta     ./results/tqa_en_train/meta.json \\
    --output_dir     ./results/exp4_en_to_ar

# Experiment 5 — EN train → AraHalluQA test
python crosslingual_eval.py \\
    --experiment     5 \\
    --train_features ./results/tqa_en_train/truthfulqa_llama2_7B_features.parquet \\
    --test_features  ./results/arahalluqa_test/arahalluqa_llama2_7B_features.parquet \\
    --train_meta     ./results/tqa_en_train/meta.json \\
    --output_dir     ./results/exp5_en_to_arahalluqa

# Experiment 6 — AraHalluQA train → EN test
python crosslingual_eval.py \\
    --experiment     6 \\
    --train_features ./results/arahalluqa_train/arahalluqa_llama2_7B_features.parquet \\
    --test_features  ./results/tqa_en_test/truthfulqa_llama2_7B_features.parquet \\
    --train_meta     ./results/arahalluqa_train/meta.json \\
    --output_dir     ./results/exp6_arahalluqa_to_en
"""

import os
import json
import argparse

import pandas as pd

import classifier

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "1": {
        "name":        "Experiment 1 — TruthfulQA EN Baseline",
        "train_label": "TruthfulQA EN train (75%)",
        "test_label":  "TruthfulQA EN test (25%)",
        "type":        "same_language",
    },
    "2": {
        "name":        "Experiment 2 — TruthfulQA AR Baseline",
        "train_label": "TruthfulQA AR train (75%)",
        "test_label":  "TruthfulQA AR test (25%)",
        "type":        "same_language",
    },
    "3": {
        "name":        "Experiment 3 — AraHalluQA Baseline",
        "train_label": "AraHalluQA train",
        "test_label":  "AraHalluQA test",
        "type":        "same_language",
    },
    "4": {
        "name":        "Experiment 4 — Cross-Lingual: TruthfulQA EN → TruthfulQA AR",
        "train_label": "TruthfulQA EN train (75%)",
        "test_label":  "TruthfulQA AR test (25%)",
        "type":        "cross_lingual",
    },
    "5": {
        "name":        "Experiment 5 — Cross-Lingual: TruthfulQA EN → AraHalluQA",
        "train_label": "TruthfulQA EN train (75%)",
        "test_label":  "AraHalluQA test",
        "type":        "cross_lingual",
    },
    "6": {
        "name":        "Experiment 6 — Cross-Lingual: AraHalluQA → TruthfulQA EN",
        "train_label": "AraHalluQA train",
        "test_label":  "TruthfulQA EN test (25%)",
        "type":        "cross_lingual",
    },
    "7": {
    "name":        "Experiment 7 — Cross-Domain: TruthfulQA AR → AraHalluQA",
    "train_label": "TruthfulQA AR train (75%)",
    "test_label":  "AraHalluQA test",
    "type":        "cross_domain",
},
"8": {
    "name":        "Experiment 8 — Cross-Domain: AraHalluQA → TruthfulQA AR",
    "train_label": "AraHalluQA train",
    "test_label":  "TruthfulQA AR test (25%)",
    "type":        "cross_domain",
},
"9": {
    "name":        "Experiment 9 — Cross-Lingual: TruthfulQA AR → TruthfulQA EN",
    "train_label": "TruthfulQA AR train (75%)",
    "test_label":  "TruthfulQA EN test (25%)",
    "type":        "cross_lingual",
},
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a HalluShift experiment (same-language or cross-lingual)."
    )
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment number: 1 | 2 | 3 | 4 | 5 | 6"
    )
    parser.add_argument(
        "--train_features", type=str, required=True,
        help="Path to the training features Parquet file."
    )
    parser.add_argument(
        "--test_features", type=str, required=True,
        help="Path to the test features Parquet file."
    )
    parser.add_argument(
        "--train_meta", type=str, required=True,
        help="Path to meta.json produced during training-set feature extraction. "
             "Provides num_layers so the model does not need to be reloaded."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save the trained model, scaler, and metrics CSV."
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs. Default: 50"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate. Default: 0.001"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed. Default: 42"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    info = EXPERIMENTS[args.experiment]

    print(f"\n{'='*65}")
    print(f"  {info['name']}")
    print(f"{'='*65}")
    print(f"  Train : {info['train_label']}")
    print(f"  Test  : {info['test_label']}")
    print(f"  Type  : {info['type']}")
    print(f"{'='*65}\n")

    # ── Load meta ─────────────────────────────────────────────────────────────
    with open(args.train_meta) as f:
        meta = json.load(f)
    num_layers = meta["num_layers"]

    # ── Load feature files ────────────────────────────────────────────────────
    print(f"Loading train features : {args.train_features}")
    train_df = pd.read_parquet(args.train_features)
    print(f"  {len(train_df)} examples, {train_df.shape[1]} columns")

    print(f"Loading test features  : {args.test_features}")
    test_df = pd.read_parquet(args.test_features)
    print(f"  {len(test_df)} examples, {test_df.shape[1]} columns\n")

    # ── Train and evaluate ────────────────────────────────────────────────────
    _, results = classifier.train_crosslingual_model(
        train_df=train_df,
        test_df=test_df,
        num_layers=num_layers,
        seed=args.seed,
        save_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
    )

    # ── Save enriched metrics CSV (one row per variant) ───────────────────────
    meta = {
        "experiment":          args.experiment,
        "experiment_name":     info["name"],
        "type":                info["type"],
        "train_set":           info["train_label"],
        "test_set":            info["test_label"],
        "train_n":             len(train_df),
        "test_n":              len(test_df),
        "train_features_file": args.train_features,
        "test_features_file":  args.test_features,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    pd.DataFrame([{"variant": k, **meta, **v} for k, v in results.items()]).to_csv(
        metrics_path, index=False
    )

    # ── Final results summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Results — {info['name']}")
    print(f"  Train : {info['train_label']}  (n={len(train_df)})")
    print(f"  Test  : {info['test_label']}  (n={len(test_df)})")
    print(f"{'='*65}")
    print(f"  Saved to: {args.output_dir}")
    print(f"    nn_no_imbalance.pth  — no imbalance handling")
    print(f"    nn_pos_weight.pth    — pos_weight (variants 2 & 3)")
    print(f"    thresholds.csv       — tuned threshold for variant 3")
    print(f"    metrics.csv          — all 3 variants\n")


if __name__ == "__main__":
    main()