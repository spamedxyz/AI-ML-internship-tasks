"""
Task 5: Optional fine-tuning script for EmpatheticDialogues.

Run when internet access and enough compute are available:
    python finetune_empathetic.py

The notebook for this task is designed to run offline. This script contains the
real Hugging Face training workflow and saves the model to empathetic_model/.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = Path(__file__).parent / "empathetic_model"
MAX_SAMPLES = 1200
MAX_LENGTH = 128


def format_dialogue(example: dict) -> dict:
    """Format one EmpatheticDialogues row for causal language modeling."""
    context = str(example.get("context", "")).strip()
    prompt = str(example.get("prompt", "")).strip()
    utterance = str(example.get("utterance", "")).strip()
    text = (
        f"User emotion: {context}\n"
        f"User message: {prompt}\n"
        f"Supportive response: {utterance}<|endoftext|>"
    )
    return {"text": text}


def main() -> None:
    print("Loading EmpatheticDialogues dataset...")
    dataset = load_dataset("empathetic_dialogues")
    train_split = dataset["train"].shuffle(seed=42)
    train_split = train_split.select(range(min(MAX_SAMPLES, len(train_split))))
    train_split = train_split.map(format_dialogue, remove_columns=train_split.column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized = train_split.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=25,
        logging_steps=25,
        save_steps=300,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print(f"Training on {len(tokenized)} examples...")
    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Saved fine-tuned model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
