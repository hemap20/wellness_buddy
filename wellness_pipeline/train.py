"""
Training module — standalone runnable stage.

Fine-tunes a base model with LoRA on either the phase-2 (pairs-only, no
system role) or phase-3 (pairs + fixed system prompt) dataset, using the one
shared TrainingConfig for both phases and all models. Loss is computed only
on assistant tokens (masked). Checkpoints are saved at configurable fractions
of total training steps for later sample-efficiency scoring.

Run standalone:
    python train.py --phase 2
    python train.py --phase 3 --model dialogpt-small

Inspect output:
    cat logs/dialogpt-small_phase2_loss.jsonl
    ls checkpoints/dialogpt-small_phase2/
"""
import argparse
import json
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DataConfig,
    LOG_DIR,
    MODELS_UNDER_TEST,
    ModelUnderTest,
    SHARED_TRAINING_CONFIG,
    TrainingConfig,
)
from model_utils import build_training_text, resolve_device


def load_phase_data(phase: int, cfg: DataConfig) -> list[dict]:
    path = cfg.phase2_path if phase == 2 else cfg.phase3_path
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run data_prep.py first.")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"{path} is empty.")
    return records


def build_example_tensors(tokenizer, messages: list[dict], max_length: int):
    import torch

    prompt_text, full_text = build_training_text(tokenizer, messages)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        full_ids = full_ids + [eos]

    full_ids = full_ids[:max_length]
    input_ids = torch.tensor(full_ids, dtype=torch.long)
    labels = input_ids.clone()

    n_prompt_tokens = min(len(prompt_ids), len(full_ids))
    labels[:n_prompt_tokens] = -100  # mask everything except assistant tokens
    return input_ids, labels


def run(phase: int, model_cfg: ModelUnderTest = None, train_cfg: TrainingConfig = None,
        data_cfg: DataConfig = None) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = model_cfg or MODELS_UNDER_TEST[0]
    train_cfg = train_cfg or SHARED_TRAINING_CONFIG
    data_cfg = data_cfg or DataConfig()

    assert phase in (2, 3), "phase must be 2 or 3"
    records = load_phase_data(phase, data_cfg)

    device = resolve_device(model_cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code
    ).to(device)

    lora_config = LoraConfig(
        r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        lora_dropout=train_cfg.lora_dropout,
        target_modules=list(train_cfg.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()

    examples = [build_example_tensors(tokenizer, r["messages"], train_cfg.max_seq_length) for r in records]

    total_steps = len(examples) * train_cfg.num_train_epochs
    checkpoint_steps = sorted({max(1, round(total_steps * f)) for f in train_cfg.checkpoint_fractions})

    run_name = f"{model_cfg.name}_phase{phase}"
    ckpt_root = CHECKPOINT_DIR / run_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    loss_log_path = LOG_DIR / f"{run_name}_loss.jsonl"

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    step = 0
    loss_history = []
    with open(loss_log_path, "w", encoding="utf-8") as loss_log:
        for epoch in range(train_cfg.num_train_epochs):
            for example_idx, (input_ids, labels) in enumerate(examples):
                step += 1
                input_ids_b = input_ids.unsqueeze(0).to(device)
                labels_b = labels.unsqueeze(0).to(device)

                outputs = model(input_ids=input_ids_b, labels=labels_b)
                loss = outputs.loss
                loss.backward()

                if step % train_cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                loss_value = float(loss.item())
                loss_history.append(loss_value)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "example_idx": example_idx,
                    "loss": loss_value,
                    "examples_seen": step,  # batch size 1
                }
                loss_log.write(json.dumps(record) + "\n")
                loss_log.flush()

                if step in checkpoint_steps:
                    pct = round(100 * step / total_steps)
                    ckpt_path = ckpt_root / f"step_{step}_pct_{pct}"
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)

    final_path = ckpt_root / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    flat = _flat_or_nondecreasing(loss_history)

    summary = {
        "run_name": run_name,
        "phase": phase,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "final_checkpoint": str(final_path),
        "loss_log": str(loss_log_path),
        "first_loss": loss_history[0] if loss_history else None,
        "last_loss": loss_history[-1] if loss_history else None,
        "loss_curve_flat_or_nondecreasing": flat,
    }
    summary_path = ckpt_root / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _flat_or_nondecreasing(loss_history: list[float], tolerance: float = 1e-3) -> bool:
    if len(loss_history) < 2:
        return False
    return loss_history[-1] >= loss_history[0] - tolerance


def main():
    parser = argparse.ArgumentParser(description="Training stage: LoRA fine-tune on phase 2 or 3 data.")
    parser.add_argument("--phase", type=int, required=True, choices=[2, 3])
    parser.add_argument("--model", type=str, default=None, help="Name of a model in MODELS_UNDER_TEST")
    args = parser.parse_args()

    model_cfg = MODELS_UNDER_TEST[0]
    if args.model:
        matches = [m for m in MODELS_UNDER_TEST if m.name == args.model]
        if not matches:
            raise SystemExit(f"Unknown model name: {args.model}")
        model_cfg = matches[0]

    summary = run(args.phase, model_cfg=model_cfg)
    print(json.dumps(summary, indent=2))
    if summary["loss_curve_flat_or_nondecreasing"]:
        print("WARNING: loss curve is flat or non-decreasing — possible training issue.")


if __name__ == "__main__":
    main()
