#!/usr/bin/env python3
"""
Train an Unsloth LoRA model on spatial sentences.
"""

import os
import re
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, Features, Sequence, Value, load_from_disk
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
from transformers import TrainerCallback, TrainerControl, TrainerState

# example
# python -m tissuenarrator.train \
#   --data ./data/merfish_spatial_df.parquet \
#   --dataset_path ./cache_merfish \
#   --output_dir ./output_merfish \
#   --model_name unsloth/Qwen3-4B-Base \
#   --max_seq_length 32000 \
#   --epochs 3


# ----------------------------
# Simple epoch-end save callback
# ----------------------------
class SaveAtEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"\n💾 Saving model at end of epoch {int(state.epoch)}")
        control.should_save = True
        return control


# ----------------------------
# Coordinate masking function
# ----------------------------
COORD_RE = re.compile(r'(X|Y):\s*-?\d+(?:\.\d+)?')

# mask XY numbers in labels (do not train to predict coords)
def split_and_mask(
    text,
    tokenizer,
    max_seq_length=32000,
    overlap=2,          # number of sentences to repeat between chunks
    min_length=100       # minimum token count to keep a chunk (0 = keep all)
):
    
    sentences = re.findall(r"<pos>.*?</cs>", text, flags=re.DOTALL)

    results = []
    i = 0
    n = len(sentences)

    while i < n:
        start = max(0, i - overlap)
        current_chunk = []
        token_count = 0
        j = start

        # pack sentences until adding the next would exceed max_seq_length
        while j < n:
            sent = sentences[j]
            tok_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
            new_count = token_count + len(tok_ids)
            if new_count > max_seq_length:
                break
            current_chunk.append(sent)
            token_count = new_count
            j += 1

        chunk_text = " ".join(current_chunk) if current_chunk else ""

        if chunk_text:
            enc_len = len(tokenizer(chunk_text, add_special_tokens=False)["input_ids"])
            if min_length == 0 or enc_len >= min_length:
                enc = tokenizer(
                    chunk_text,
                    return_offsets_mapping=True,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_seq_length,
                )
                input_ids = enc["input_ids"]
                offsets = enc["offset_mapping"]
                labels = input_ids.copy()

                mask_spans = []
                for m in COORD_RE.finditer(chunk_text):
                    full_start, full_end = m.span()
                    match_str = m.group()
                    num_start = match_str.find(":") + 2
                    span_start = full_start + num_start
                    span_end = full_end
                    mask_spans.append((span_start, span_end))
                    
                for k, (s_char, e_char) in enumerate(offsets):
                    for a, b in mask_spans:
                        if s_char >= a and e_char <= b:
                            labels[k] = -100
                            break

                results.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": [1] * len(input_ids),
                })

        if j >= n:
            i = n
        else:
            i = max(j - overlap + 1, 0)

    return results


# ----------------------------
# Main training logic
# ----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Unsloth model on spatial data.")
    parser.add_argument("--data", type=str, help="Path to input parquet file.")
    parser.add_argument("--dataset_path", type=str, default="./cache_merfish", help="Path to save/load HF dataset.")
    parser.add_argument("--output_dir", type=str, default="./output_merfish", help="Path to save model checkpoints.")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Base", help="Base model name (e.g. unsloth/Qwen3-4B-Base).")
    parser.add_argument("--max_seq_length", type=int, default=16000, help="Maximum sequence length.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n===== CONFIGURATION =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("==========================\n")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load and process data
    df = pd.read_parquet(args.data).head(100)
    print(f"Loaded dataframe: {df.shape}")

    if os.path.exists(args.dataset_path):
        hf_dataset = load_from_disk(args.dataset_path)
        print(f"Loaded cached dataset from: {args.dataset_path}")
    else:
        os.makedirs(args.dataset_path, exist_ok=True)
        tmp_file = os.path.join(args.dataset_path, "records.jsonl")
        with open(tmp_file, "w") as f:
            for sent, split in tqdm(zip(df["sentence"], df["split"]), total=len(df), desc="Splitting & Masking"):
                chunks = split_and_mask(sent, tokenizer, max_seq_length=args.max_seq_length)
                for c in chunks:
                    c["split"] = split
                    f.write(json.dumps(c) + "\n")

        features = Features({
            "input_ids": Sequence(Value("int32")),
            "labels": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "split": Value("string"),
        })

        hf_dataset = Dataset.from_json(tmp_file, features=features)
        hf_dataset.save_to_disk(args.dataset_path)
        print(f"Saved dataset to: {args.dataset_path}")

    train_dataset = hf_dataset.filter(lambda x: x["split"] == "train")
    
    if "val" in set(hf_dataset["split"]) and len(hf_dataset.filter(lambda x: x["split"] == "val")) > 0:
        val_dataset = hf_dataset.filter(lambda x: x["split"] == "val")
    else:
        val_dataset = None
        print("⚠️ No validation split found — training without evaluation.")
    
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset and len(val_dataset) > 0 else None,
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        callbacks=[SaveAtEpochEndCallback()],
        args=UnslothTrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=args.epochs,
            warmup_ratio=0.01,
            learning_rate=2e-4,
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=args.output_dir,
            report_to="none",
            save_strategy="steps",
            save_steps=500,
            save_total_limit=50,
            logging_strategy="steps",
            eval_strategy="epoch" if val_dataset is not None else "no",
        ),
    )
    trainer.train()
    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
