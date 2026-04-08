"""
extract_features.py – HalluShift feature extraction with GPT-4o-as-a-Judge labelling.

Supported datasets (all loaded from local CSV files)
-----------------------------------------------------
  truthfulqa    – TruthfulQA English
  truthfulqa_ar – TruthfulQA Arabic translation
  arahalluqa    – AraHalluQA

Expected CSV columns
--------------------
  truthfulqa    : question, best_answer
  truthfulqa_ar : question, best_answer
  arahalluqa    : question, answer

Labelling method
----------------
GPT-4o compares the LLM-generated answer against the reference answer and
returns a binary label directly:
  0 = not hallucinated (generated answer is semantically correct)
  1 = hallucinated     (generated answer is wrong or fabricated)

English datasets receive an English judge prompt.
Arabic datasets (truthfulqa_ar, arahalluqa) receive an Arabic judge prompt.

Each example is one GPT-4o API call. For ~817 TruthfulQA questions expect
roughly 5–15 minutes of labelling time depending on API rate limits.

Usage examples
--------------
# TruthfulQA English
python extract_features.py \\
    --model_name   llama2_7B \\
    --dataset_name truthfulqa \\
    --csv_path     ./data/truthfulqa_en_train.csv \\
    --output_dir   ./results/tqa_en_train

# TruthfulQA Arabic
python extract_features.py \\
    --model_name   llama2_7B \\
    --dataset_name truthfulqa_ar \\
    --csv_path     ./data/truthfulqa_ar_train.csv \\
    --output_dir   ./results/tqa_ar_train

# AraHalluQA
python extract_features.py \\
    --model_name   llama2_7B \\
    --dataset_name arahalluqa \\
    --csv_path     ./data/arahalluqa_train.csv \\
    --output_dir   ./results/arahalluqa_train
"""

import os
import json
import time
import argparse
import warnings
import re as _re
import pandas as pd
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

import functions

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OPENAI_KEY  = "" #add your key
login(token="") #add your key
MODELS = {
    'allam_7B':    "ALLaM-AI/ALLaM-7B-Instruct-preview",
    'qwen2.5_14B': "Qwen/Qwen2.5-14B-Instruct",
    'silma_9B':    "silma-ai/SILMA-9B-Instruct-v1.0",
    'hala_9B':     "hammh0a/Hala-9B",
    'mistral_8B':  "mistralai/Ministral-8B-Instruct-2410",
    'phi4_mini':   "microsoft/Phi-4-mini-instruct",
}
 
# ALLaM is prompted directly without a chat template.
# All other models use their tokenizer's built-in chat template.
PLAIN_CAUSAL_MODELS = {'allam_7B'}
 
# Inference prompt templates — language must match the dataset.
PROMPT_TEMPLATES = {
    'truthfulqa':    "Answer the question concisely. Q: {question} A: Put your final answer within \\boxed{{}}.",
    'truthfulqa_ar': "أجب عن السؤال بإيجاز. س: {question} ج:",
    'arahalluqa':    "أجب عن السؤال بإيجاز. س: {question} ج:",
}
 
# Per-dataset column configuration.
# question_col – actual column name in the CSV for the question.
# answer_col   – actual column name in the CSV for the reference answer.
# language     – controls which judge prompt and inference template are used.
DATASET_CONFIG = {
    'truthfulqa': {
        'question_col': 'Original Question',
        'answer_col':   'Original Best Answer',
        'language':     'en',
    },
    'truthfulqa_ar': {
        'question_col': 'Translated Question',
        'answer_col':   'Translated Best Answer',
        'language':     'ar',
    },
    'arahalluqa': {
        'question_col': 'Question',   # capital Q
        'answer_col':   'Answer',     # capital A
        'language':     'ar',
    },
}
 
VALID_DATASETS = list(DATASET_CONFIG.keys())
 
# ---------------------------------------------------------------------------
# GPT-4o judge prompt templates
#
# Design principles:
#   - Chain-of-thought: one reasoning sentence then the label on a new line.
#   - Focus on factual equivalence, not phrasing.
#   - "I don't know" / "it depends" are always label 0 — not hallucination.
#   - Partial answers with a clear factual error are label 1.
#   - Trap question rule: trust the reference answer even against popular belief.
#   - Temperature 0, seed 42 for deterministic judgements.
#   - Arabic prompt mirrors English exactly in meaning.
# ---------------------------------------------------------------------------
 
JUDGE_SYSTEM_EN = (
    "You are an expert factual judge for a hallucination detection task. "
    "You will be given a question, a reference answer, and a generated answer. "
    "Your job is to decide whether the generated answer is factually correct "
    "relative to the reference answer.\n\n"
    "Rules:\n"
    "1. Focus on factual equivalence, not wording. A paraphrase of the reference "
    "answer is correct (label 0).\n"
    "2. A partially correct answer that also contains a clear factual error "
    "is hallucinated (label 1).\n"
    "3. An answer such as 'I don't know' or 'it depends' is always correct (label 0). "
    "A model expressing uncertainty or refusing to guess is not hallucinating.\n"
    "4. If the question is a common misconception trap and the reference answer "
    "contradicts popular belief, trust the reference answer — it is correct.\n\n"
    "Output format (strictly follow this):\n"
    "Line 1: One sentence explaining your reasoning.\n"
    "Line 2: 0 (correct, not hallucinated) or 1 (wrong, hallucinated). "
    "Nothing else on this line."
)
 
JUDGE_USER_EN = (
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Generated answer: {generated}\n\n"
    "Is the generated answer factually correct relative to the reference?\n"
    "Reason briefly, then output 0 or 1 on a new line."
)
 
JUDGE_SYSTEM_AR = (
    "أنت محكّم خبير في اكتشاف الهلوسة في إجابات النماذج اللغوية. "
    "ستُعطى سؤالاً وإجابةً مرجعيةً وإجابةً مُولَّدة. "
    "مهمتك تحديد ما إذا كانت الإجابة المُولَّدة صحيحةً واقعياً مقارنةً بالإجابة المرجعية.\n\n"
    "القواعد:\n"
    "1. ركّز على التكافؤ الواقعي لا على الصياغة. إعادة الصياغة بمعنى صحيح تُعدّ إجابة صحيحة (0).\n"
    "2. الإجابة الجزئية التي تحتوي على خطأ واقعي واضح تُعدّ مهلوسة (1).\n"
    "3. إجابة 'لا أعلم' أو 'يعتمد الأمر' تُعدّ دائماً صحيحة (0). "
    "النموذج الذي يُعبّر عن عدم اليقين أو يرفض التخمين لا يُعدّ مهلوساً.\n"
    "4. إذا كان السؤال يتضمّن مفهوماً خاطئاً شائعاً والإجابة المرجعية تعارضه، "
    "فثق بالإجابة المرجعية — فهي الصحيحة.\n\n"
    "تنسيق الإخراج (اتبعه بدقة):\n"
    "السطر الأول: جملة واحدة تشرح استنتاجك.\n"
    "السطر الثاني: 0 (صحيحة، غير مهلوسة) أو 1 (خاطئة، مهلوسة). "
    "لا شيء آخر في هذا السطر."
)
 
JUDGE_USER_AR = (
    "السؤال: {question}\n"
    "الإجابة المرجعية: {reference}\n"
    "الإجابة المُولَّدة: {generated}\n\n"
    "هل الإجابة المُولَّدة صحيحة واقعياً مقارنةً بالإجابة المرجعية؟\n"
    "اشرح باختصار، ثم اكتب 0 أو 1 في سطر جديد."
)
 
 
# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
 
def load_csv_dataset(csv_path: str, dataset_name: str) -> pd.DataFrame:
    """Load and validate a CSV dataset file.
 
    Renames the dataset-specific column names to internal standard names:
      question    → always 'question'
      answer col  → 'best_answer' for TruthfulQA variants, 'answer' for AraHalluQA
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
 
    df  = pd.read_csv(csv_path)
    cfg = DATASET_CONFIG[dataset_name]
 
    question_col = cfg['question_col']
    answer_col   = cfg['answer_col']
 
    missing = [c for c in [question_col, answer_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns for '{dataset_name}': {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
 
    # Internal answer column name — always lowercase
    internal_answer = 'answer' if dataset_name == 'arahalluqa' else 'best_answer'
    df = df.rename(columns={
        question_col: 'question',
        answer_col:   internal_answer,
    })
 
    return df.dropna(subset=['question']).reset_index(drop=True)
 
 
# ---------------------------------------------------------------------------
# Inference prompt builder
# ---------------------------------------------------------------------------
 
def build_prompt(row: pd.Series, dataset_name: str) -> str:
    """Fill the inference prompt template for a single row."""
    return PROMPT_TEMPLATES[dataset_name].format(question=row['question'])
 
 
# ---------------------------------------------------------------------------
# HalluShift feature extraction for one row
# ---------------------------------------------------------------------------
 
def process_row(row: pd.Series, model, tokenizer, num_layers: int,
                dataset_name: str, model_name: str) -> list:
    """Generate a model response and extract all HalluShift features.
 
    Handles two prompting styles:
      - Plain causal LM (allam_7B): tokenized directly, no chat template.
      - Chat-template models: wrapped via apply_chat_template.
 
    Returns a flat list:
        hidden_state_features + attention_features + [max_probs, min_probs] + [decoded_text]
    """
    prompt_text = build_prompt(row, dataset_name)
 
    # Determine device from model parameters
    device = next(model.parameters()).device
 
    if model_name in PLAIN_CAUSAL_MODELS:
        prompt = tokenizer(prompt_text, return_tensors='pt').to(device)
    else:
        messages  = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt = tokenizer(formatted, return_tensors='pt').to(device)
 
    generated = model.generate(
        **prompt,
        do_sample=False,          # greedy decoding — deterministic
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_attentions=True,
        output_logits=True,
    )
 
    decoded = tokenizer.decode(
        generated.sequences[0, prompt["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )
 
    # ── Answer cleaning ───────────────────────────────────────────────────────
 
    # Step 1: extract last \boxed{...} content if present
    boxed_matches = _re.findall(r'\\boxed\{([^}]*)\}', decoded)
    boxed_matches = [m.strip() for m in boxed_matches if m.strip()]
    if boxed_matches:
        decoded = boxed_matches[-1]
    else:
        # Step 2: extract after answer marker (A: or ج:)
        for marker in ["A:", "ج:"]:
            if marker in decoded:
                decoded = decoded.split(marker, 1)[1].strip()
                break
 
        # Step 3: fallback — take text after last question mark
        if not boxed_matches:
            for q_marker in ["؟", "?"]:
                if q_marker in decoded:
                    after_q = decoded.rsplit(q_marker, 1)[-1].strip()
                    if after_q:
                        decoded = after_q
                        break
 
        # Step 4: cut off repeated prompt noise
        for noise in ["Q:", "س:", " A:", " ج:"]:
            if noise in decoded:
                decoded = decoded.split(noise, 1)[0].strip()
 
    # Step 5: remove \text{...} LaTeX wrapper
    decoded = _re.sub(r'\\text\{?([^}\\]*)', r'\1', decoded)
 
    # Step 6: keep first two sentences (split on first two dots)
    parts   = decoded.split(".")
    decoded = ". ".join(parts[:2]).strip()
    if decoded and not decoded.endswith("."):
        decoded += "."
 
    return (
        functions.plot_internal_state_2(generated, num_layers)
        + functions.plot_internal_state_2(generated, num_layers, state="attention")
        + functions.probability_function(generated)
        + [decoded]
    )
 
 
# ---------------------------------------------------------------------------
# GPT-4o-as-a-Judge labelling
# ---------------------------------------------------------------------------
 
def judge_single(client: OpenAI, question: str, reference: str,
                 generated: str, language: str,
                 retries: int = 3, retry_delay: float = 5.0) -> int:
    """Call GPT-4o to judge one (question, reference, generated) triple.
 
    Returns:
        int: 0 (not hallucinated) or 1 (hallucinated).
             Conservative fallback of 1 if all retries fail.
    """
    if language == 'ar':
        system_prompt = JUDGE_SYSTEM_AR
        user_prompt   = JUDGE_USER_AR.format(
            question=question, reference=reference, generated=generated
        )
    else:
        system_prompt = JUDGE_SYSTEM_EN
        user_prompt   = JUDGE_USER_EN.format(
            question=question, reference=reference, generated=generated
        )
 
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0,    # deterministic
                max_tokens=80,    # enough for one reasoning sentence + label
                seed=42,
            )
            raw = response.choices[0].message.content.strip()
 
            # GPT-4o sometimes puts reasoning and label on the same line.
            # Try last line first, then fall back to last character of response.
            lines     = [l.strip() for l in raw.splitlines() if l.strip()]
            label_str = lines[-1] if lines else ""
 
            # If last line is not a bare digit, check if response ends with 0 or 1
            if label_str not in ("0", "1"):
                last_char = raw.strip()[-1] if raw.strip() else ""
                if last_char in ("0", "1"):
                    label_str = last_char
 
            if label_str in ("0", "1"):
                return int(label_str)
 
            print(f"  Warning: unexpected GPT-4o response '{raw}' — retrying "
                  f"({attempt + 1}/{retries})")
 
        except Exception as e:
            print(f"  Warning: API error on attempt {attempt + 1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(retry_delay)
 
    print("  Warning: all retries failed — defaulting to hallucinated (1)")
    return 1
 
 
def run_gpt4o_labelling(dataset: pd.DataFrame, dataset_name: str,
                        llm_answers: pd.Series,
                        openai_client: OpenAI) -> pd.DataFrame:
    """Label all examples using GPT-4o as a judge."""
    cfg        = DATASET_CONFIG[dataset_name]
    language   = cfg['language']
    # Use internal column name (post-rename) — always lowercase
    answer_col = 'answer' if dataset_name == 'arahalluqa' else 'best_answer'
 
    labels = []
    for i, (_, row) in enumerate(tqdm(dataset.iterrows(),
                                      total=len(dataset),
                                      desc="GPT-4o judging")):
        label = judge_single(
            client    = openai_client,
            question  = str(row['question']),
            reference = str(row[answer_col]),
            generated = str(llm_answers.iloc[i]),
            language  = language,
        )
        labels.append(label)
 
    return pd.DataFrame({
        'id':            [str(i) for i in range(len(dataset))],
        'hallucination': labels,
    })
 
 
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
 
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser(
        description="Extract HalluShift features and label with GPT-4o-as-a-Judge."
    )
    parser.add_argument(
        '--model_name', type=str, required=True, choices=list(MODELS.keys()),
        help='Model key, e.g. phi4_mini'
    )
    parser.add_argument(
        '--dataset_name', type=str, required=True, choices=VALID_DATASETS,
        help='Dataset key: truthfulqa | truthfulqa_ar | arahalluqa'
    )
    parser.add_argument(
        '--csv_path', type=str, required=True,
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory where all output files will be saved'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed. Default: 42'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit to N rows for trial runs. Remove for full run.'
    )
    args = parser.parse_args()
 
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
 
    lang = DATASET_CONFIG[args.dataset_name]['language']
 
    print(f"\n{'='*65}")
    print(f"  HalluShift Feature Extraction + GPT-4o Labelling")
    print(f"  Dataset  : {args.dataset_name}  (language: {lang})")
    print(f"  CSV      : {args.csv_path}")
    print(f"  Model    : {args.model_name}")
    print(f"  Output   : {args.output_dir}")
    print(f"{'='*65}\n")
 
    # Checkpoint paths
    ckpt_raw    = os.path.join(args.output_dir, "checkpoint_raw.parquet")
    ckpt_labels = os.path.join(args.output_dir, "checkpoint_labels.csv")
    meta_path   = os.path.join(args.output_dir, "meta.json")
    features_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_{args.model_name}_features.parquet"
    )
    responses_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_{args.model_name}_responses.csv"
    )
 
    # ── 1. Load dataset ───────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_csv_dataset(args.csv_path, args.dataset_name)
    #dataset = dataset.head(20).reset_index(drop=True)  # trial: remove this line for full run
    if args.max_samples is not None:
        dataset = dataset.head(args.max_samples).reset_index(drop=True)
        print(f"  {len(dataset)} examples loaded (limited to {args.max_samples}).")
    else:
        print(f"  {len(dataset)} examples loaded.")
    print()
 
    # ── 2 + 3. LLM loading + feature extraction (skip if checkpoint exists) ───
    if os.path.exists(ckpt_raw):
        print(f"Checkpoint found — skipping LLM inference.\n"
              f"  Loading raw features from {ckpt_raw}\n")
        raw_df     = pd.read_parquet(ckpt_raw)
        num_layers = json.load(open(meta_path))['num_layers']
    else:
        model_id  = MODELS[args.model_name]
        print(f"Loading model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,   # fixed: was dtype=
            low_cpu_mem_usage=True,
            device_map="auto",
            cache_dir="./models",
            attn_implementation="eager",
        )
        num_layers = len(lm.model.layers)
        print(f"  {num_layers} transformer layers detected.\n")
 
        # Save meta immediately so it survives a crash
        with open(meta_path, 'w') as f:
            json.dump({
                'num_layers':   num_layers,
                'model_name':   args.model_name,
                'dataset_name': args.dataset_name,
                'language':     lang,
                'judge':        'gpt-4o',
            }, f, indent=2)
 
        print("Extracting HalluShift features (this may take a while)...")
        results = []
        for _, row in tqdm(dataset.iterrows(), total=len(dataset),
                           desc="Generating responses"):
            results.append(
                process_row(row, lm, tokenizer, num_layers,
                            args.dataset_name, args.model_name)
            )
 
        raw_df = pd.DataFrame(results)
        raw_df.to_parquet(ckpt_raw, index=False)
        print(f"Feature extraction complete. Checkpoint saved → {ckpt_raw}\n")
 
    llm_answers = raw_df.iloc[:, -1]
    feature_df  = raw_df.iloc[:, :-1]
 
    # ── 4. GPT-4o labelling (skip if checkpoint exists) ───────────────────────
    if os.path.exists(ckpt_labels):
        print(f"Checkpoint found — skipping GPT-4o labelling.\n"
              f"  Loading labels from {ckpt_labels}\n")
        df_labels = pd.read_csv(ckpt_labels)
    else:
        print(f"Running GPT-4o-as-a-Judge labelling "
              f"({'Arabic' if lang == 'ar' else 'English'} prompts)...")
        openai_client = OpenAI(api_key=OPENAI_KEY)
        df_labels = run_gpt4o_labelling(
            dataset, args.dataset_name, llm_answers, openai_client
        )
        df_labels.to_csv(ckpt_labels, index=False)
        n_hal     = df_labels['hallucination'].sum()
        n_not_hal = len(df_labels) - n_hal
        print(f"  Labelling complete: {n_not_hal} not hallucinated (0), "
              f"{n_hal} hallucinated (1) out of {len(df_labels)} examples.")
        print(f"  Checkpoint saved → {ckpt_labels}\n")
 
    # ── 5. Save responses + labels ────────────────────────────────────────────
    pd.DataFrame({
        'question':      dataset['question'].values,
        'llm_answer':    llm_answers.values,
        'hallucination': df_labels['hallucination'].values,
    }).to_csv(responses_path, index=False)
    print(f"Responses saved  → {responses_path}")
 
    # ── 6. Feature engineering + merge labels ─────────────────────────────────
    print("Engineering features...")
    processed = functions.data_preparation(feature_df, df_labels, num_layers)
    processed.to_parquet(features_path, index=False)
    print(f"Features saved   → {features_path}\n")
 
    # ── 7. Finalise meta ──────────────────────────────────────────────────────
    with open(meta_path, 'w') as f:
        json.dump({
            'num_layers':   num_layers,
            'model_name':   args.model_name,
            'dataset_name': args.dataset_name,
            'language':     lang,
            'judge':        'gpt-4o',
        }, f, indent=2)
    print(f"Meta info saved  → {meta_path}")
    print("\nFeature extraction pipeline finished successfully.\n")
 
 
if __name__ == '__main__':
    main()