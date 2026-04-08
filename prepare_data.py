"""
prepare_data.py – One-time data preparation before feature extraction.

What this script does
---------------------
1. Reads your single TruthfulQA CSV (contains both English and Arabic columns).
2. Produces four split CSVs from it — EN train, EN test, AR train, AR test —
   each containing the two columns that extract_features.py expects.
3. Validates your AraHalluQA train and test CSVs exist and have the required
   columns (you supply these — no splitting is performed on them).
4. Prints a summary of all files and row counts, plus the exact
   extract_features.py commands to run next.

Your TruthfulQA CSV must have these four columns
-------------------------------------------------
  Original Question       – English question
  Translated Question     – Arabic question
  Original Best Answer    – English reference answer
  Translated Best Answer  – Arabic reference answer

Output files produced
---------------------
  <output_dir>/
    truthfulqa_en_train.csv   – 75% of rows, columns: Original Question,
                                Original Best Answer
    truthfulqa_en_test.csv    – 25% of rows, same columns
    truthfulqa_ar_train.csv   – 75% of rows, columns: Translated Question,
                                Translated Best Answer
    truthfulqa_ar_test.csv    – 25% of rows, same columns

Files you supply (not created here)
-------------------------------------
  arahalluqa_train.csv    – your AraHalluQA train split  (columns: question, answer)
  arahalluqa_test.csv     – your AraHalluQA test split   (columns: question, answer)

Usage
-----
python prepare_data.py \\
    --truthfulqa_csv   ./data/truthfulqa.csv \\
    --arahalluqa_train ./data/arahalluqa_train.csv \\
    --arahalluqa_test  ./data/arahalluqa_test.csv \\
    --output_dir       ./data
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Column names in the TruthfulQA CSV
# ---------------------------------------------------------------------------

TQA_EN_QUESTION = "Original Question"
TQA_EN_ANSWER   = "Original Best Answer"
TQA_AR_QUESTION = "Translated Question"
TQA_AR_ANSWER   = "Translated Best Answer"
TQA_CATEGORY    = "category"   # optional — used for stratified splitting if present

# Required columns for AraHalluQA
ARAHALLUQA_REQUIRED = ["Question", "Answer"]


# ---------------------------------------------------------------------------
# TruthfulQA loader and splitter
# ---------------------------------------------------------------------------

def load_and_split_truthfulqa(csv_path: str, output_dir: str,
                               test_size: float = 0.25,
                               seed: int = 42) -> dict:
    """Load the single TruthfulQA CSV and produce four split files.

    Reads the shared CSV, validates all four required columns are present,
    then splits rows 75/25 (stratified on 'category' if available) and writes:
      - truthfulqa_en_train.csv  (Original Question + Original Best Answer)
      - truthfulqa_en_test.csv
      - truthfulqa_ar_train.csv  (Translated Question + Translated Best Answer)
      - truthfulqa_ar_test.csv

    The split is performed once on the row indices so EN and AR train/test sets
    are aligned — row N in the EN train file corresponds to row N in the AR
    train file.

    Args:
        csv_path:   Path to the TruthfulQA CSV file.
        output_dir: Directory to write the four output CSVs.
        test_size:  Fraction for the test split. Default 0.25.
        seed:       Random seed. Default 42.

    Returns:
        dict mapping descriptive labels to (path, row_count) tuples.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TruthfulQA CSV not found: {csv_path}")

    print(f"Loading TruthfulQA CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    required = [TQA_EN_QUESTION, TQA_EN_ANSWER, TQA_AR_QUESTION, TQA_AR_ANSWER]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"TruthfulQA CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[TQA_EN_QUESTION, TQA_AR_QUESTION]).reset_index(drop=True)
    print(f"  {len(df)} rows loaded.\n")

    # ── Determine stratification ──────────────────────────────────────────────
    stratify = None
    if TQA_CATEGORY in df.columns:
        min_class = df[TQA_CATEGORY].value_counts().min()
        if min_class >= 2:
            stratify = df[TQA_CATEGORY]
            print(f"  Using stratified split on '{TQA_CATEGORY}' column.")
        else:
            print(f"  '{TQA_CATEGORY}' has classes with < 2 samples — using plain random split.")
    else:
        print("  No 'category' column found — using plain random split.")

    pct = int((1 - test_size) * 100)
    print(f"  Splitting {pct}% train / {int(test_size * 100)}% test (seed={seed})...\n")

    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=stratify
    )

    # ── Write EN splits ───────────────────────────────────────────────────────
    en_train = df.loc[train_idx, [TQA_EN_QUESTION, TQA_EN_ANSWER]].reset_index(drop=True)
    en_test  = df.loc[test_idx,  [TQA_EN_QUESTION, TQA_EN_ANSWER]].reset_index(drop=True)

    en_train_path = os.path.join(output_dir, "truthfulqa_en_train.csv")
    en_test_path  = os.path.join(output_dir, "truthfulqa_en_test.csv")
    en_train.to_csv(en_train_path, index=False)
    en_test.to_csv(en_test_path,   index=False)
    print(f"  TruthfulQA EN train → {en_train_path}  ({len(en_train)} rows)")
    print(f"  TruthfulQA EN test  → {en_test_path}   ({len(en_test)} rows)")

    # ── Write AR splits ───────────────────────────────────────────────────────
    ar_train = df.loc[train_idx, [TQA_AR_QUESTION, TQA_AR_ANSWER]].reset_index(drop=True)
    ar_test  = df.loc[test_idx,  [TQA_AR_QUESTION, TQA_AR_ANSWER]].reset_index(drop=True)

    ar_train_path = os.path.join(output_dir, "truthfulqa_ar_train.csv")
    ar_test_path  = os.path.join(output_dir, "truthfulqa_ar_test.csv")
    ar_train.to_csv(ar_train_path, index=False)
    ar_test.to_csv(ar_test_path,   index=False)
    print(f"  TruthfulQA AR train → {ar_train_path}  ({len(ar_train)} rows)")
    print(f"  TruthfulQA AR test  → {ar_test_path}   ({len(ar_test)} rows)\n")

    return {
        "TruthfulQA EN — train": (en_train_path, len(en_train)),
        "TruthfulQA EN — test":  (en_test_path,  len(en_test)),
        "TruthfulQA AR — train": (ar_train_path, len(ar_train)),
        "TruthfulQA AR — test":  (ar_test_path,  len(ar_test)),
    }


# ---------------------------------------------------------------------------
# AraHalluQA validation
# ---------------------------------------------------------------------------

def validate_arahalluqa(train_path: str, test_path: str) -> dict:
    """Validate that the AraHalluQA CSVs exist and have the required columns.

    Args:
        train_path: Path to AraHalluQA train CSV.
        test_path:  Path to AraHalluQA test CSV.

    Returns:
        dict mapping descriptive labels to (path, row_count) tuples.
    """
    print("Validating AraHalluQA CSVs...")
    result = {}
    for label, path in [("AraHalluQA — train", train_path),
                         ("AraHalluQA — test",  test_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"AraHalluQA file not found: {path}")
        df = pd.read_csv(path)
        missing = [c for c in ARAHALLUQA_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"'{path}' is missing required columns: {missing}\n"
                f"Found: {list(df.columns)}"
            )
        print(f"  {label} → {path}  ({len(df)} rows)  ✓")
        result[label] = (path, len(df))
    print()
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(files: dict, model_name: str = "llama2_7B"):
    """Print a formatted file summary and the exact commands to run next."""

    print("=" * 70)
    print("  Data Preparation Complete — File Summary")
    print("=" * 70)
    print(f"  {'Role':<35} {'Rows':>5}  Path")
    print(f"  {'-'*35} {'-'*5}  {'-'*28}")
    for role, (path, nrows) in files.items():
        status = "✓" if os.path.exists(path) else "✗ MISSING"
        print(f"  {role:<35} {str(nrows):>5}  {path}  {status}")

    print("\n" + "=" * 70)
    print("  Next — Extract Features (run each on GPU)")
    print("=" * 70)

    m = model_name
    extraction_commands = [
        ("TruthfulQA EN train",  "truthfulqa",    files["TruthfulQA EN — train"][0], f"./results/tqa_en_train"),
        ("TruthfulQA EN test",   "truthfulqa",    files["TruthfulQA EN — test"][0],  f"./results/tqa_en_test"),
        ("TruthfulQA AR train",  "truthfulqa_ar", files["TruthfulQA AR — train"][0], f"./results/tqa_ar_train"),
        ("TruthfulQA AR test",   "truthfulqa_ar", files["TruthfulQA AR — test"][0],  f"./results/tqa_ar_test"),
    ]
    if "AraHalluQA — train" in files:
        extraction_commands += [
            ("AraHalluQA train", "arahalluqa", files["AraHalluQA — train"][0], f"./results/arahalluqa_train"),
            ("AraHalluQA test",  "arahalluqa", files["AraHalluQA — test"][0],  f"./results/arahalluqa_test"),
        ]

    for label, dataset_name, csv_path, out_dir in extraction_commands:
        print(f"\n  # {label}")
        print(f"  python extract_features.py \\")
        print(f"      --model_name   {m} \\")
        print(f"      --dataset_name {dataset_name} \\")
        print(f"      --csv_path     {csv_path} \\")
        print(f"      --output_dir   {out_dir}")
    print()
    print("  NOTE: Set OPENAI_KEY in extract_features.py before running.")

    print("\n" + "=" * 70)
    print("  Then — Run Experiments (CPU only, seconds each)")
    print("=" * 70)

    experiments = [
        ("1", "TruthfulQA EN baseline",
         f"./results/tqa_en_train/truthfulqa_{m}_features.parquet",
         f"./results/tqa_en_test/truthfulqa_{m}_features.parquet",
         "./results/tqa_en_train/meta.json",
         "./results/exp1_tqa_en"),
        ("2", "TruthfulQA AR baseline",
         f"./results/tqa_ar_train/truthfulqa_ar_{m}_features.parquet",
         f"./results/tqa_ar_test/truthfulqa_ar_{m}_features.parquet",
         "./results/tqa_ar_train/meta.json",
         "./results/exp2_tqa_ar"),
        ("4", "EN train → AR test",
         f"./results/tqa_en_train/truthfulqa_{m}_features.parquet",
         f"./results/tqa_ar_test/truthfulqa_ar_{m}_features.parquet",
         "./results/tqa_en_train/meta.json",
         "./results/exp4_en_to_ar"),
    ]
    if "AraHalluQA — train" in files:
        experiments += [
            ("3", "AraHalluQA baseline",
             f"./results/arahalluqa_train/arahalluqa_{m}_features.parquet",
             f"./results/arahalluqa_test/arahalluqa_{m}_features.parquet",
             "./results/arahalluqa_train/meta.json",
             "./results/exp3_arahalluqa"),
            ("5", "EN train → AraHalluQA test",
             f"./results/tqa_en_train/truthfulqa_{m}_features.parquet",
             f"./results/arahalluqa_test/arahalluqa_{m}_features.parquet",
             "./results/tqa_en_train/meta.json",
             "./results/exp5_en_to_arahalluqa"),
            ("6", "AraHalluQA train → EN test",
             f"./results/arahalluqa_train/arahalluqa_{m}_features.parquet",
             f"./results/tqa_en_test/truthfulqa_{m}_features.parquet",
             "./results/arahalluqa_train/meta.json",
             "./results/exp6_arahalluqa_to_en"),
        ]

    for exp_id, label, train_f, test_f, meta, out_dir in experiments:
        print(f"\n  # Experiment {exp_id} — {label}")
        print(f"  python crosslingual_eval.py \\")
        print(f"      --experiment     {exp_id} \\")
        print(f"      --train_features {train_f} \\")
        print(f"      --test_features  {test_f} \\")
        print(f"      --train_meta     {meta} \\")
        print(f"      --output_dir     {out_dir}")

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare all dataset CSVs before HalluShift feature extraction."
    )
    parser.add_argument(
        "--truthfulqa_csv", type=str, required=True,
        help=(
            "Path to your single TruthfulQA CSV containing both English and Arabic. "
            f"Must have columns: '{TQA_EN_QUESTION}', '{TQA_EN_ANSWER}', "
            f"'{TQA_AR_QUESTION}', '{TQA_AR_ANSWER}'."
        )
    )
    parser.add_argument(
        "--arahalluqa_train", type=str, default=None,
        help="Path to your AraHalluQA train CSV (columns: question, answer). "
             "Optional — skip if you do not have this file yet."
    )
    parser.add_argument(
        "--arahalluqa_test", type=str, default=None,
        help="Path to your AraHalluQA test CSV (columns: question, answer). "
             "Optional — skip if you do not have this file yet."
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Directory where the four split CSVs will be saved. Default: ./data"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.25,
        help="Test fraction for TruthfulQA splits. Default: 0.25"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the train/test split. Default: 42"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  HalluShift — Data Preparation")
    print("=" * 70 + "\n")

    # ── 1. TruthfulQA EN + AR splits from the single CSV ─────────────────────
    tqa_files = load_and_split_truthfulqa(
        args.truthfulqa_csv, args.output_dir, args.test_size, args.seed
    )

    # ── 2. AraHalluQA validation (skipped if files not provided) ────────────
    ara_files = {}
    if args.arahalluqa_train or args.arahalluqa_test:
        if not args.arahalluqa_train or not args.arahalluqa_test:
            raise ValueError(
                "Provide both --arahalluqa_train and --arahalluqa_test, or neither."
            )
        ara_files = validate_arahalluqa(args.arahalluqa_train, args.arahalluqa_test)
    else:
        print("AraHalluQA files not provided — skipping validation.")
        print("Run prepare_data.py again with --arahalluqa_train and --arahalluqa_test")
        print("once you have those files.")

    # ── 3. Summary + next-step commands ───────────────────────────────────────
    print_summary({**tqa_files, **ara_files})


if __name__ == "__main__":
    main()