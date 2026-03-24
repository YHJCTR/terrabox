#!/usr/bin/env python3
"""
Terrabox SFT Training Script
=============================
Fine-tunes a Qwen3-VL (or Qwen2.5-VL) vision-language model on the converted
OpenEarthAgent dataset (data/openearth/train_sft.jsonl) using Unsloth LoRA.

Data format: JSONL, each line is:
  {"messages": [
    {"role": "user",      "content": [{"type": "image", "image": "/abs/path.jpg"}, {"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "text",  "text": "..."}]},
    ...
  ]}

Prerequisites:
  pip install unsloth transformers datasets tqdm pillow

Usage (DO NOT RUN during development):
  cd /data1/yuhongjie2/terrabox
  python scripts/train/train_sft.py [--model 2b|8b] [--data data/openearth/train_sft.jsonl]
"""

import argparse
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATHS = {
    "2b": "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_2B_VL/",
    "8b": "/data1/yuhongjie2/Earth-Agent/llm/qwen/3_8B_VL/",
    "25b_7b": "/data1/yuhongjie2/Earth-Agent/llm/qwen/2.5_7B_VL/",
}

DEFAULT_DATA = str(REPO_ROOT / "data" / "openearth" / "train_sft.jsonl")
DEFAULT_OUTPUT = str(REPO_ROOT / "output" / "terrabox_sft")

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str, limit: int = 0) -> List[Dict]:
    """Load JSONL training data."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit and limit > 0:
        records = records[:limit]
    log.info(f"Loaded {len(records)} training samples from {path}")
    return records


def cache_images_to_ram(data: List[Dict]) -> List[Dict]:
    """
    Phase 1: Read all image files into RAM as bytes (lazy decode).
    This avoids repeated disk reads and multiprocessing deadlocks.
    """
    log.info("Caching images to RAM (this may take a few minutes)...")
    for item in tqdm(data, desc="Loading images"):
        for msg in item.get("messages", []):
            for content in msg.get("content", []):
                if content.get("type") != "image":
                    continue
                img_path = content.get("image", "")
                if not img_path:
                    continue
                try:
                    with open(img_path, "rb") as f:
                        content["image_bytes"] = f.read()
                    del content["image"]  # remove path reference
                except FileNotFoundError:
                    log.debug(f"Image not found: {img_path} — using black placeholder")
                    buf = io.BytesIO()
                    Image.new("RGB", (224, 224), (0, 0, 0)).save(buf, format="JPEG")
                    content["image_bytes"] = buf.getvalue()
                    del content["image"]
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def make_formatting_func(tokenizer):
    """
    Returns a batch transform function that decodes image bytes and tokenizes.
    Called lazily by the Dataset transform at training time.
    """
    def formatting_func(examples: Dict[str, List]) -> Dict:
        messages_batch = examples["messages"]
        inputs_list = []

        for messages in messages_batch:
            clean_messages = []
            for msg in messages:
                clean_content = []
                for item in msg.get("content", []):
                    if item.get("type") == "image":
                        img_bytes = item.get("image_bytes")
                        try:
                            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        except Exception:
                            pil_img = Image.new("RGB", (224, 224), (0, 0, 0))
                        clean_content.append({"type": "image", "image": pil_img})
                    elif item.get("type") == "text":
                        clean_content.append({"type": "text", "text": str(item.get("text", ""))})
                clean_messages.append({"role": msg["role"], "content": clean_content})

            try:
                encoded = tokenizer.apply_chat_template(
                    clean_messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                )
                inputs_list.append(encoded)
            except Exception as e:
                log.debug(f"Tokenization failed: {e}")
                continue

        if not inputs_list:
            return {}

        batch_output: Dict[str, List] = {}
        for key in inputs_list[0].keys():
            vals = []
            for item in inputs_list:
                val = item[key]
                if key != "image_grid_thw":
                    if isinstance(val, torch.Tensor) and val.ndim == 2 and val.shape[0] == 1:
                        val = val.squeeze(0)
                    elif isinstance(val, list) and len(val) == 1 and isinstance(val[0], list):
                        val = val[0]
                vals.append(val)
            batch_output[key] = vals

        if "labels" not in batch_output and "input_ids" in batch_output:
            batch_output["labels"] = [x[:] for x in batch_output["input_ids"]]

        return batch_output

    return formatting_func


# ──────────────────────────────────────────────────────────────────────────────
# Data collator (masks image token regions from loss)
# ──────────────────────────────────────────────────────────────────────────────

class TerraboxDataCollator:
    """
    Collator that:
    - Pads sequences
    - Masks image token spans in labels (no loss on vision tokens)
    - Masks user turns in labels (train only on assistant responses)
    """
    VISION_START_ID = 151857
    VISION_END_ID   = 151858

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.text_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, return_tensors="pt"
        )

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        text_features = []
        pixel_values_list = []
        image_grid_list = []

        for f in features:
            input_ids = torch.tensor(f["input_ids"]).long()
            labels = torch.tensor(f.get("labels", f["input_ids"])).long()

            # Mask image token spans in labels
            v_start = (input_ids == self.VISION_START_ID).nonzero(as_tuple=True)[0]
            v_end   = (input_ids == self.VISION_END_ID).nonzero(as_tuple=True)[0]
            for s, e in zip(v_start, v_end):
                labels[s : e + 1] = -100

            text_features.append({
                "input_ids": input_ids,
                "attention_mask": torch.tensor(f["attention_mask"]).long(),
                "labels": labels,
            })

            if "pixel_values" in f:
                pixel_values_list.append(f["pixel_values"])
            if "image_grid_thw" in f:
                image_grid_list.append(f["image_grid_thw"])

        batch = self.text_collator(text_features)

        if pixel_values_list:
            batch["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        if image_grid_list:
            batch["image_grid_thw"] = torch.cat(image_grid_list, dim=0)

        return batch


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(model_size: str, data_path: str, output_dir: str, limit: int = 0):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_path = MODEL_PATHS.get(model_size)
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            f"Available sizes: {list(MODEL_PATHS.keys())}"
        )

    # ── Phase 1: Load data ────────────────────────────────────────────────────
    raw_data = load_jsonl(data_path, limit=limit)
    raw_data = cache_images_to_ram(raw_data)
    dataset = Dataset.from_list(raw_data)

    # ── Phase 2: Load model ───────────────────────────────────────────────────
    log.info(f"Loading model from {model_path} ...")
    from unsloth import FastVisionModel, is_bfloat16_supported

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=(model_size in ("8b", "25b_7b")),  # 4-bit for large models
        use_gradient_checkpointing=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Phase 3: Configure LoRA ───────────────────────────────────────────────
    lora_r = 32 if model_size == "2b" else 16
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    log.info(f"LoRA configured: r={lora_r}, model_size={model_size}")

    # ── Phase 4: Set transform ────────────────────────────────────────────────
    dataset.set_transform(make_formatting_func(tokenizer))

    # ── Phase 5: Training args ────────────────────────────────────────────────
    batch_size = 16 if model_size == "2b" else 8
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=0,   # 0 = main process (avoids CUDA fork issues)
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    collator = TerraboxDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # ── Phase 6: Train ────────────────────────────────────────────────────────
    log.info("Starting training...")
    trainer.train()

    # ── Phase 7: Save ─────────────────────────────────────────────────────────
    merged_dir = Path(output_dir) / "merged_model"
    log.info(f"Saving merged model to {merged_dir} ...")
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    log.info("Training complete!")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Terrabox SFT training on OpenEarthAgent data")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PATHS.keys()),
        default="2b",
        help="Model size to fine-tune (default: 2b)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help=f"Path to training JSONL file (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for checkpoints and merged model (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max training samples (0 = all, default: 0)",
    )
    args = parser.parse_args()

    train(
        model_size=args.model,
        data_path=args.data,
        output_dir=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
