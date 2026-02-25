"""
LoRA fine-tune Whisper for Hebrew speech recognition.

Usage:
    python train_lora.py \
        --model_name ivrit-ai/whisper-large-v3-turbo \
        --dataset_path ./data/prepared \
        --output_dir ./checkpoints/whisper-lora \
        --language he

    # Resume from checkpoint:
    python train_lora.py ... --resume_from_checkpoint

    # Force CPU if MPS has issues:
    python train_lora.py ... --force_cpu
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model


def get_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(example, processor, language):
    audio = example["audio"]
    input_features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="np"
    ).input_features[0]

    labels = processor.tokenizer(example["text"], return_tensors="np").input_ids[0]

    return {"input_features": input_features, "labels": labels}


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Whisper")
    parser.add_argument("--model_name", default="ivrit-ai/whisper-large-v3-turbo")
    parser.add_argument("--dataset_path", required=True, help="Path to prepared dataset")
    parser.add_argument("--output_dir", default="./checkpoints/whisper-lora")
    parser.add_argument("--language", default="he")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_all_linear", action="store_true")

    # Training config
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_checkpoints", type=int, default=3)
    args = parser.parse_args()

    device = get_device(args.force_cpu)
    print(f"Device: {device}")

    print(f"Loading model: {args.model_name}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    # Freeze encoder, configure for training
    model.freeze_encoder()
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language, task="transcribe"
    )
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    # LoRA setup
    target_modules = (
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        if args.lora_all_linear
        else ["q_proj", "v_proj"]
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Dataset
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_from_disk(args.dataset_path)

    print("Processing dataset...")
    prep_fn = partial(prepare_dataset, processor=processor, language=args.language)
    ds = ds.map(prep_fn, remove_columns=ds["train"].column_names, num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER metric
    compute_metrics = None
    try:
        from jiwer import wer

        def compute_metrics_fn(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
            if not pairs:
                return {"wer": 0.0}
            pred_str, label_str = zip(*pairs)
            return {"wer": wer(list(label_str), list(pred_str))}

        compute_metrics = compute_metrics_fn
        print("WER metric enabled")
    except ImportError:
        print("Warning: jiwer not installed, WER disabled")

    has_test = "test" in ds

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if has_test else "no",
        save_total_limit=args.max_checkpoints,
        logging_steps=10,
        report_to="none",
        predict_with_generate=True,
        generation_max_length=225,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=has_test,
        metric_for_best_model="wer" if has_test else None,
        greater_is_better=False if has_test else None,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("test"),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    print("\nStarting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    final_path = Path(args.output_dir) / "final-adapter"
    model.save_pretrained(str(final_path))
    processor.save_pretrained(str(final_path))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDone! Adapter saved to {final_path}")
    print(f"Trainable params: {trainable:,} (~{trainable * 4 / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
