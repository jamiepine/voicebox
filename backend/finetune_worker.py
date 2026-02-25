#!/usr/bin/env python3
"""
Standalone fine-tuning worker script for Qwen3-TTS.

Follows the official Qwen3-TTS fine-tuning pipeline (sft_12hz.py):
  1. Extract audio codes using Qwen3TTSTokenizer (12Hz, 16 codebook layers)
  2. Build dual-channel input (text + codec) with speaker embedding injection
  3. Train with LoRA on the talker sub-model
  4. Loss = talker_loss + 0.3 * sub_talker_loss

Runs as a subprocess, isolated from the API server.
Receives configuration via a JSON config file path argument.
Writes progress to training_log.jsonl.

Usage:
    python -m backend.finetune_worker /path/to/config.json
    # or
    python backend/finetune_worker.py /path/to/config.json
"""

import json
import sys
import os
import random
import time
import shutil
from pathlib import Path
from datetime import datetime, timezone


def write_log(log_path: str, entry: dict):
    """Append a log entry to the training log file."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def main():
    if len(sys.argv) < 2:
        print("Usage: python finetune_worker.py <config.json>", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = json.load(f)

    profile_id = config["profile_id"]
    job_id = config["job_id"]
    manifest_path = config["manifest_path"]
    ref_audio_path = config["ref_audio_path"]
    output_dir = config["output_dir"]
    log_path = config["log_path"]
    epochs = config.get("epochs", 3)
    learning_rate = config.get("learning_rate", 2e-5)
    batch_size = config.get("batch_size", 1)
    lora_rank = config.get("lora_rank", 32)
    lora_alpha = config.get("lora_alpha", 64)

    print(f"[Worker] Starting fine-tuning for profile {profile_id}")
    print(f"[Worker] Config: epochs={epochs}, lr={learning_rate}, batch={batch_size}, rank={lora_rank}")

    write_log(log_path, {
        "type": "status",
        "message": "Starting training worker",
        "profile_id": profile_id,
    })

    try:
        import torch

        # Load training data from manifest
        training_data = []
        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    training_data.append(json.loads(line))

        num_samples = len(training_data)
        print(f"[Worker] Loaded {num_samples} training samples")

        write_log(log_path, {
            "type": "status",
            "message": f"Loaded {num_samples} training samples",
        })

        # Try to use the real Qwen3-TTS fine-tuning pipeline
        try:
            from qwen_tts import Qwen3TTSTokenizer
            from peft import LoraConfig, get_peft_model, TaskType

            _run_qwen_finetune(
                training_data=training_data,
                ref_audio_path=ref_audio_path,
                output_dir=output_dir,
                log_path=log_path,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        except ImportError as e:
            if os.environ.get("VOICEBOX_DEV_MODE"):
                print(f"[Worker] Qwen3-TTS fine-tuning dependencies not available: {e}")
                print(f"[Worker] DEV MODE: Falling back to simulated training")

                gradient_accumulation_steps = 4
                steps_per_epoch = max(1, num_samples // (batch_size * gradient_accumulation_steps))
                total_steps = steps_per_epoch * epochs

                _run_simulated_training(
                    output_dir=output_dir,
                    log_path=log_path,
                    epochs=epochs,
                    total_steps=total_steps,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                )
            else:
                raise RuntimeError(
                    f"Fine-tuning dependencies not available: {e}. "
                    "Please install: pip install peft qwen3-tts"
                )

        write_log(log_path, {
            "type": "status",
            "message": "Training completed successfully",
        })

        print(f"[Worker] Training completed. Adapter saved to {output_dir}")

    except Exception as e:
        error_msg = str(e)
        print(f"[Worker] Training failed: {error_msg}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        write_log(log_path, {
            "type": "error",
            "message": error_msg,
        })
        sys.exit(1)


def _run_qwen_finetune(
    training_data: list,
    ref_audio_path: str,
    output_dir: str,
    log_path: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int,
    lora_alpha: int,
):
    """
    Run Qwen3-TTS fine-tuning following the official pipeline (sft_12hz.py).

    Stage 1: Extract audio codes using Qwen3TTSTokenizer (12Hz, 16 layers)
    Stage 2: Build dual-channel inputs and train with LoRA on talker
    """
    import torch
    import numpy as np
    import librosa
    from peft import LoraConfig, get_peft_model, TaskType
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # Apple Silicon MPS — use CPU for training (MPS doesn't support all ops)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ============================================
    # Stage 1: Extract audio codes
    # ============================================
    print("[Worker] Stage 1: Extracting audio codes with Qwen3TTSTokenizer...")
    write_log(log_path, {"type": "status", "message": "Extracting audio codes..."})

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    )

    # Extract codes — tokenizer.encode accepts file paths directly
    ENCODE_BATCH = 4
    for i in range(0, len(training_data), ENCODE_BATCH):
        batch = training_data[i:i + ENCODE_BATCH]
        audio_paths = [item["audio"] for item in batch]

        enc_result = tokenizer.encode(audio_paths)
        # enc_result.audio_codes is List[Tensor], each shape [T, 16]
        for j, item in enumerate(batch):
            item["audio_codes"] = enc_result.audio_codes[j].cpu()  # keep as tensor

        write_log(log_path, {
            "type": "progress",
            "epoch": 0,
            "step": 0,
            "total_steps": 0,
            "loss": None,
            "message": f"Extracting audio codes: {min(i + ENCODE_BATCH, len(training_data))}/{len(training_data)}",
        })

    print(f"[Worker] Audio codes extracted for {len(training_data)} samples")

    # Free tokenizer memory
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    # ============================================
    # Stage 2: Load model and apply LoRA
    # ============================================
    print("[Worker] Stage 2: Loading Qwen3-TTS base model...")
    write_log(log_path, {"type": "status", "message": "Loading base model..."})

    wrapper = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        dtype=dtype,
    )
    model = wrapper.model  # Qwen3TTSForConditionalGeneration
    processor = wrapper.processor
    config = model.config  # Qwen3TTSConfig

    # Extract config values for special token IDs
    talker_config = config.talker_config
    tts_pad_token_id = config.tts_pad_token_id      # 151671
    tts_bos_token_id = config.tts_bos_token_id      # 151672
    tts_eos_token_id = config.tts_eos_token_id      # 151673
    codec_eos_token_id = talker_config.codec_eos_token_id  # 4198
    codec_nothink_id = talker_config.codec_nothink_id      # 4203
    codec_think_bos_id = talker_config.codec_think_bos_id  # 4204
    codec_think_eos_id = talker_config.codec_think_eos_id  # 4205
    codec_pad_id = talker_config.codec_pad_id              # 4196
    codec_bos_id = talker_config.codec_bos_id              # 4197
    num_code_groups = talker_config.num_code_groups         # 16 for 12Hz variant
    hidden_size = talker_config.hidden_size

    print(f"[Worker] Model loaded. hidden_size={hidden_size}, num_code_groups={num_code_groups}")

    # Apply LoRA to the talker sub-model
    print(f"[Worker] Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) to talker...")
    write_log(log_path, {"type": "status", "message": f"Applying LoRA (rank={lora_rank})"})

    target_modules = set()
    for name, module in model.talker.named_modules():
        if isinstance(module, torch.nn.Linear):
            mod_name = name.split(".")[-1]
            if mod_name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                target_modules.add(mod_name)
    if not target_modules:
        target_modules = {"q_proj", "v_proj"}

    print(f"[Worker] LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,  # rank-stabilized scaling: alpha/sqrt(r) instead of alpha/r
    )

    model.talker = get_peft_model(model.talker, lora_config)
    model.talker.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency (~35% VRAM savings)
    model.talker.gradient_checkpointing_enable()
    model.talker.enable_input_require_grads()

    if device != "cpu":
        model = model.to(device)

    # After PEFT wrapping, the inner TalkerModel is at:
    # model.talker (PeftModelForCausalLM) -> base_model -> model (ForConditionalGen) -> model (TalkerModel)
    talker_inner = model.talker.base_model.model.model

    # ============================================
    # Helper: extract mel spectrogram for speaker encoder
    # ============================================
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    def extract_mels(audio_path):
        audio, sr = librosa.load(audio_path, sr=24000, mono=True)
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        mels = mel_spectrogram(
            audio_t, n_fft=1024, num_mels=128, sampling_rate=24000,
            hop_size=256, win_size=1024, fmin=0, fmax=12000,
        ).transpose(1, 2)
        return mels

    # Pre-compute reference speaker embedding
    print("[Worker] Computing speaker embedding from reference audio...")
    ref_mel = extract_mels(ref_audio_path).to(device).to(dtype)
    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(ref_mel).detach()  # [1, hidden_size]

    # ============================================
    # Helper: build a single training sample's tensors
    # following the official dataset.py collate_fn
    # ============================================
    def build_sample_tensors(item):
        """Build the dual-channel input_ids, masks, and labels for one sample."""
        # Tokenize text
        text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        text_inputs = processor(text=text, return_tensors="pt", padding=False)
        text_ids = text_inputs["input_ids"][0, :-5]  # trim trailing tokens, shape [t]

        # Audio codes: [T, num_code_groups]
        audio_codes = item["audio_codes"]
        if isinstance(audio_codes, list):
            audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        codec_len = audio_codes.shape[0]
        codec_0 = audio_codes[:, 0]  # first codebook layer
        text_len = text_ids.shape[0]

        # Build dual-channel input_ids: [seq_len, 2]
        # Channel 0 = text token IDs, Channel 1 = codec token IDs
        # Layout (following official collate_fn):
        #   pos 0..2:  text[0:3]         | pad
        #   pos 3:     pad               | codec_nothink
        #   pos 4:     pad               | codec_think_bos
        #   pos 5:     pad               | codec_think_eos
        #   pos 6:     pad               | 0 (speaker embedding slot)
        #   pos 7:     tts_bos           | codec_pad
        #   pos 8..8+text_len-4: text[3:]| codec_pad
        #   pos 8+text_len-3:   tts_eos  | codec_bos
        #   pos 8+text_len-2...: tts_pad | codec_0[:]
        #   pos end:   tts_pad           | codec_eos
        #
        total_len = 8 + (text_len - 3) + 1 + codec_len + 1  # +1 for codec_bos, +1 for codec_eos

        input_ids = torch.zeros(total_len, 2, dtype=torch.long)
        text_embedding_mask = torch.zeros(total_len, 1, dtype=dtype)
        codec_embedding_mask = torch.zeros(total_len, 1, dtype=dtype)
        codec_0_labels = torch.full((total_len,), -100, dtype=torch.long)
        codec_mask = torch.zeros(total_len, dtype=torch.bool)
        all_codec_ids = torch.zeros(total_len, num_code_groups, dtype=torch.long)

        # Channel 0: text
        input_ids[:3, 0] = text_ids[:3]
        input_ids[3:7, 0] = tts_pad_token_id
        input_ids[7, 0] = tts_bos_token_id
        text_body_end = 8 + text_len - 3
        input_ids[8:text_body_end, 0] = text_ids[3:]
        input_ids[text_body_end, 0] = tts_eos_token_id
        input_ids[text_body_end + 1:, 0] = tts_pad_token_id

        # Channel 1: codec
        input_ids[:3, 1] = 0  # pad
        input_ids[3, 1] = codec_nothink_id
        input_ids[4, 1] = codec_think_bos_id
        input_ids[5, 1] = codec_think_eos_id
        input_ids[6, 1] = 0  # speaker embedding slot
        input_ids[7, 1] = codec_pad_id
        input_ids[8:text_body_end, 1] = codec_pad_id
        codec_start = text_body_end  # position of codec_bos
        input_ids[codec_start, 1] = codec_bos_id
        input_ids[codec_start + 1:codec_start + 1 + codec_len, 1] = codec_0
        input_ids[codec_start + 1 + codec_len, 1] = codec_eos_token_id

        # Text embedding mask: 1 for all active positions
        text_embedding_mask[:total_len, 0] = 1.0

        # Codec embedding mask: 1 from pos 3 onward, but 0 at pos 6 (speaker slot)
        codec_embedding_mask[3:, 0] = 1.0
        codec_embedding_mask[6, 0] = 0.0  # speaker embedding injected here

        # Codec mask: True only for actual audio codec positions
        codec_mask[codec_start + 1:codec_start + 1 + codec_len] = True

        # Labels: codec_0 IDs at audio positions, -100 everywhere else
        codec_0_labels[codec_start + 1:codec_start + 1 + codec_len] = codec_0
        # Also set codec_eos as a label
        codec_0_labels[codec_start + 1 + codec_len] = codec_eos_token_id

        # Full codec IDs (all 16 layers) at audio positions
        all_codec_ids[codec_start + 1:codec_start + 1 + codec_len] = audio_codes

        return {
            "input_ids": input_ids,                # [T, 2]
            "text_embedding_mask": text_embedding_mask,  # [T, 1]
            "codec_embedding_mask": codec_embedding_mask, # [T, 1]
            "codec_0_labels": codec_0_labels,      # [T]
            "codec_mask": codec_mask,              # [T]
            "codec_ids": all_codec_ids,            # [T, 16]
            "total_len": total_len,
        }

    # ============================================
    # Training loop (following official sft_12hz.py)
    # ============================================
    import math

    gradient_accumulation_steps = 4
    batches_per_epoch = math.ceil(len(training_data) / batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / gradient_accumulation_steps)
    total_steps = steps_per_epoch * epochs

    write_log(log_path, {
        "type": "progress",
        "epoch": 0,
        "step": 0,
        "total_steps": total_steps,
        "loss": None,
    })

    model.talker.train()
    if model.speaker_encoder is not None:
        model.speaker_encoder.eval()

    trainable_params = [p for p in model.talker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    # LR warmup (5% of total steps) + cosine decay
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_steps = max(1, int(total_steps * 0.05))
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

    global_step = 0
    step_times = []  # for ETA estimation
    training_diverged = False
    CHECKPOINT_EVERY_STEPS = max(20, total_steps // 3)  # checkpoint ~3 times during training

    def _do_optimizer_step(batch_loss_sum, batch_count, epoch_idx):
        """Perform optimizer step, log progress, return step loss."""
        nonlocal global_step
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        step_loss = batch_loss_sum / max(batch_count, 1)

        # Track step time for ETA
        step_time = time.time() - step_start_holder[0]
        step_times.append(step_time)
        if len(step_times) > 20:
            step_times.pop(0)
        avg_step_time = sum(step_times) / len(step_times)
        eta_seconds = max(0, int((total_steps - global_step) * avg_step_time))

        current_lr = scheduler.get_last_lr()[0]

        write_log(log_path, {
            "type": "progress",
            "epoch": epoch_idx + 1,
            "step": global_step,
            "total_steps": total_steps,
            "loss": round(step_loss, 4),
            "grad_norm": round(grad_norm.item(), 4) if hasattr(grad_norm, 'item') else round(float(grad_norm), 4),
            "learning_rate": round(current_lr, 8),
            "eta_seconds": eta_seconds,
        })
        print(f"[Worker] Step {global_step}/{total_steps} - Loss: {step_loss:.4f} | GradNorm: {float(grad_norm):.3f} | LR: {current_lr:.2e} | ETA: {eta_seconds}s")
        return step_loss

    # Mutable holder so _do_optimizer_step can read step start time
    step_start_holder = [time.time()]

    print(f"[Worker] Starting training: {epochs} epochs, {total_steps} total steps, warmup={warmup_steps} steps")

    for epoch in range(epochs):
        if training_diverged:
            break

        epoch_loss = 0.0
        epoch_steps = 0
        accumulation_count = 0  # reset per epoch
        accum_loss_sum = 0.0
        accum_count = 0

        random.shuffle(training_data)

        for i in range(0, len(training_data), batch_size):
            if training_diverged:
                break

            batch_items = training_data[i:i + batch_size]
            if accumulation_count == 0:
                step_start_holder[0] = time.time()

            for item in batch_items:
                try:
                    sample = build_sample_tensors(item)
                    seq_len = sample["total_len"]

                    input_ids = sample["input_ids"].unsqueeze(0).to(device)         # [1, T, 2]
                    text_emb_mask = sample["text_embedding_mask"].unsqueeze(0).to(device)  # [1, T, 1]
                    codec_emb_mask = sample["codec_embedding_mask"].unsqueeze(0).to(device) # [1, T, 1]
                    codec_0_labels = sample["codec_0_labels"].unsqueeze(0).to(device)  # [1, T]
                    codec_mask = sample["codec_mask"].unsqueeze(0).to(device)          # [1, T]
                    codec_ids = sample["codec_ids"].unsqueeze(0).to(device)            # [1, T, 16]

                    # Attention mask: all ones
                    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=device)

                    # ---- Build embeddings (official approach) ----
                    text_embed = talker_inner.text_embedding(input_ids[:, :, 0])  # [1, T, hidden]
                    text_embed = text_embed * text_emb_mask

                    codec_embed = talker_inner.codec_embedding(input_ids[:, :, 1])  # [1, T, hidden]
                    codec_embed = codec_embed * codec_emb_mask

                    # Inject speaker embedding at position 6
                    codec_embed[:, 6, :] = speaker_embedding

                    # Sum text + codec embeddings (NOT concatenate!)
                    input_embeddings = text_embed + codec_embed

                    # Add sub-codebook embeddings (layers 1-15) at codec positions
                    for layer_idx in range(1, num_code_groups):
                        sub_embed = model.talker.code_predictor.get_input_embeddings()[layer_idx - 1](
                            codec_ids[:, :, layer_idx]
                        )
                        sub_embed = sub_embed * codec_mask.unsqueeze(-1)
                        input_embeddings = input_embeddings + sub_embed

                    # ---- Forward pass ----
                    outputs = model.talker(
                        inputs_embeds=input_embeddings[:, :-1, :],
                        attention_mask=attention_mask[:, :-1],
                        labels=codec_0_labels[:, 1:],
                        output_hidden_states=True,
                    )

                    loss = outputs.loss

                    # ---- Sub-talker loss (codec layers 1-15) ----
                    try:
                        if isinstance(outputs.hidden_states, tuple):
                            if isinstance(outputs.hidden_states[0], tuple):
                                hidden = outputs.hidden_states[0][-1]
                            else:
                                hidden = outputs.hidden_states[-1]
                        else:
                            hidden = outputs.hidden_states[-1]

                        shifted_codec_mask = codec_mask[:, 1:]
                        codec_hidden = hidden[shifted_codec_mask]
                        shifted_codec_ids = codec_ids[:, 1:]
                        codec_at_mask = shifted_codec_ids[shifted_codec_mask]

                        if codec_hidden.shape[0] > 0 and codec_at_mask.shape[0] > 0:
                            _, sub_loss = model.talker.forward_sub_talker_finetune(
                                codec_at_mask, codec_hidden
                            )
                            loss = loss + 0.3 * sub_loss
                    except Exception as e:
                        print(f"[Worker] Sub-talker loss skipped: {e}")

                    # ---- Backward ----
                    # Scale by actual batches in this accumulation group
                    # (last group per epoch may have fewer than gradient_accumulation_steps)
                    remaining_batches = batches_per_epoch - (i // batch_size)
                    actual_accum = min(gradient_accumulation_steps, remaining_batches)
                    scaled_loss = loss / actual_accum
                    scaled_loss.backward()
                    accum_loss_sum += loss.item()
                    accum_count += 1

                except Exception as e:
                    print(f"[Worker] Warning: Error processing sample: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            accumulation_count += 1

            # Gradient step every gradient_accumulation_steps
            if accumulation_count % gradient_accumulation_steps == 0:
                step_loss = _do_optimizer_step(accum_loss_sum, accum_count, epoch)
                epoch_loss += step_loss
                epoch_steps += 1
                accum_loss_sum = 0.0
                accum_count = 0
                step_start_holder[0] = time.time()  # start timing next step

                # Early stopping: detect divergence
                if step_loss > 50.0 or (accum_count > 0 and torch.isnan(torch.tensor(step_loss))):
                    print(f"[Worker] Training diverged at step {global_step} (loss={step_loss:.4f}). Stopping early.")
                    write_log(log_path, {
                        "type": "error",
                        "message": f"Training diverged at step {global_step}, loss={step_loss:.4f}",
                    })
                    training_diverged = True
                    break

                # Periodic checkpointing
                if global_step % CHECKPOINT_EVERY_STEPS == 0 and global_step < total_steps:
                    _save_checkpoint(model, speaker_embedding, output_dir, global_step)
                    write_log(log_path, {
                        "type": "checkpoint",
                        "step": global_step,
                        "message": f"Checkpoint saved at step {global_step}",
                    })

        # Flush remaining gradients at end of epoch (partial accumulation group)
        if accumulation_count % gradient_accumulation_steps != 0 and accum_count > 0:
            step_loss = _do_optimizer_step(accum_loss_sum, accum_count, epoch)
            epoch_loss += step_loss
            epoch_steps += 1
            accum_loss_sum = 0.0
            accum_count = 0

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"[Worker] Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f}")
        write_log(log_path, {
            "type": "progress",
            "epoch": epoch + 1,
            "step": global_step,
            "total_steps": total_steps,
            "loss": round(avg_epoch_loss, 4),
        })

    if training_diverged:
        raise RuntimeError("Training diverged (loss exploded). Try lowering the learning rate.")

    # ============================================
    # Save final LoRA adapter + speaker embedding
    # ============================================
    print(f"[Worker] Saving final LoRA adapter to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.talker.save_pretrained(output_dir, safe_serialization=True)

    # Save speaker embedding as safetensors (secure, fast loading)
    try:
        import safetensors.torch as st
        st.save_file(
            {"speaker_embedding": speaker_embedding.cpu()},
            str(Path(output_dir) / "speaker_embedding.safetensors"),
        )
        print("[Worker] Speaker embedding saved (safetensors)")
    except ImportError:
        # Fallback to torch.save if safetensors not available
        torch.save(speaker_embedding.cpu(), str(Path(output_dir) / "speaker_embedding.pt"))
        print("[Worker] Speaker embedding saved (torch)")
    except Exception as e:
        print(f"[Worker] Warning: Could not save speaker embedding: {e}")

    # Clean up intermediate checkpoints (keep only the final adapter)
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if checkpoints_dir.exists():
        shutil.rmtree(str(checkpoints_dir))

    print(f"[Worker] Adapter saved successfully")


def _save_checkpoint(model, speaker_embedding, output_dir, global_step):
    """Save a periodic checkpoint during training."""
    import torch
    checkpoint_dir = Path(output_dir) / "checkpoints" / f"step-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.talker.save_pretrained(str(checkpoint_dir), safe_serialization=True)

    try:
        import safetensors.torch as st
        st.save_file(
            {"speaker_embedding": speaker_embedding.cpu()},
            str(checkpoint_dir / "speaker_embedding.safetensors"),
        )
    except Exception:
        torch.save(speaker_embedding.cpu(), str(checkpoint_dir / "speaker_embedding.pt"))

    # Keep only the last 2 checkpoints
    MAX_CHECKPOINTS = 2
    checkpoints_parent = Path(output_dir) / "checkpoints"
    existing = sorted(checkpoints_parent.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    for old in existing[:-MAX_CHECKPOINTS]:
        shutil.rmtree(str(old))

    print(f"[Worker] Checkpoint saved at step {global_step}")


def _run_simulated_training(
    output_dir: str,
    log_path: str,
    epochs: int,
    total_steps: int,
    lora_rank: int,
    lora_alpha: int,
):
    """
    Simulated training for development/testing when Qwen3-TTS
    fine-tuning dependencies aren't available.

    Creates a minimal adapter config so the pipeline can be tested end-to-end.
    """
    import time

    print("[Worker] Running simulated training (no GPU fine-tuning)")
    write_log(log_path, {"type": "status", "message": "Simulated training mode"})

    loss = 2.5
    global_step = 0
    steps_per_epoch = max(1, total_steps // epochs)

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            global_step += 1
            loss = max(0.05, loss * random.uniform(0.95, 0.99))

            write_log(log_path, {
                "type": "progress",
                "epoch": epoch + 1,
                "step": global_step,
                "total_steps": total_steps,
                "loss": round(loss, 4),
            })

            time.sleep(0.5)

        print(f"[Worker] Simulated epoch {epoch + 1}/{epochs}, loss: {loss:.4f}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "peft_type": "LORA",
        "r": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "simulated": True,
    }
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    try:
        import torch
        dummy_state = {"base_model.model.dummy.lora_A.weight": torch.zeros(lora_rank, 1)}
        torch.save(dummy_state, str(output_path / "adapter_model.bin"))
    except ImportError:
        (output_path / "adapter_model.bin").touch()

    print(f"[Worker] Simulated adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
