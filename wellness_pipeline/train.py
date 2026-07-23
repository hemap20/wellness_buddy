"""
Training module — standalone runnable stage.

Fine-tunes a base model with LoRA on either the phase-2 (pairs-only, no
system role) or phase-3 (pairs + fixed system prompt) dataset, using the one
shared TrainingConfig for both phases and all models. Loss is computed only
on assistant tokens (masked). Checkpoints are saved at configurable fractions
of total training steps for later sample-efficiency scoring.

If data_prep.py produced a val.jsonl for this phase/dataset-size (see
DataConfig.val_split_ratio/val_samples), validation loss is computed at the
same checkpoint steps as train loss and both are recorded in
train_summary.json's "checkpoint_curve" — a simple diagnostic
"possible_overfitting" flag is set if val_loss rises while train_loss keeps
falling. No val.jsonl -> both fields are None/False, no behavior change from
before.

Run standalone:
    python train.py --phase 2
    python train.py --phase 3 --model dialogpt-small

Inspect output:
    cat logs/dialogpt-small_phase2_loss.jsonl
    ls checkpoints/dialogpt-small_phase2/
"""
import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Optional

from config import (
    CHECKPOINT_DIR,
    DataConfig,
    LOG_DIR,
    MODELS_UNDER_TEST,
    ModelUnderTest,
    PIPELINE_VERSION,
    SHARED_TRAINING_CONFIG,
    TrainingConfig,
    batch_settings_for,
    rank_alpha_for,
)
from model_utils import build_training_text, load_tokenizer_and_model, resolve_device
from results_manager import format_dataset_size


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phase_data(phase: int, cfg: DataConfig) -> tuple[list[dict], list[dict]]:
    """Returns (train_records, val_records). val_records is [] if data_prep.py
    skipped the validation split for this dataset size (val.jsonl absent)."""
    train_path = cfg.train_path(phase)
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run data_prep.py first.")
    train_records = _load_jsonl(train_path)
    if not train_records:
        raise ValueError(f"{train_path} is empty.")

    val_path = cfg.val_path(phase)
    val_records = _load_jsonl(val_path) if val_path.exists() else []
    return train_records, val_records


def build_example_tensors(tokenizer, messages: list[dict], max_length: int, expects_chat_template: bool = True):
    import torch

    prompt_text, full_text = build_training_text(tokenizer, messages, expects_chat_template=expects_chat_template)
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


def collate_batch(batch: list, pad_token_id: int):
    """Pads a list of (input_ids, labels) tensors to the batch's max length.
    input_ids padded with pad_token_id, labels padded with -100 (HF's default
    CrossEntropyLoss ignore_index — so padded positions contribute exactly
    zero to the loss, identical in effect to how prompt tokens are already
    masked). attention_mask is 0 over padding, 1 over real tokens, so the
    model's attention itself also never attends into padding.

    Returns (input_ids_batch, labels_batch, attention_mask_batch), each
    shape (batch_size, max_len_in_batch).
    """
    import torch

    max_len = max(ids.shape[0] for ids, _ in batch)
    input_ids_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask_batch = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, labels) in enumerate(batch):
        n = ids.shape[0]
        input_ids_batch[i, :n] = ids
        labels_batch[i, :n] = labels
        attention_mask_batch[i, :n] = 1

    return input_ids_batch, labels_batch, attention_mask_batch


def evaluate_avg_loss(model, examples: list, device: str) -> float:
    """Average loss over a held-out set, same assistant-only masking as
    training. Caller is responsible for model.train()/model.eval() around
    this — kept side-effect-free here beyond the temporary no_grad context."""
    import torch

    was_training = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_ids, labels in examples:
            input_ids_b = input_ids.unsqueeze(0).to(device)
            labels_b = labels.unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids_b, labels=labels_b)
            total_loss += float(outputs.loss.item())
    if was_training:
        model.train()
    return total_loss / len(examples)


def check_overfitting(checkpoint_train_loss: list[float], checkpoint_val_loss: list[Optional[float]]) -> bool:
    """True if val_loss at the final checkpoint is higher than at some
    earlier checkpoint, while train_loss is (non-strictly) monotonically
    decreasing over that same span. A diagnostic signal only — no early
    stopping is implemented."""
    if not checkpoint_val_loss or any(v is None for v in checkpoint_val_loss) or len(checkpoint_val_loss) < 2:
        return False
    tolerance = 1e-6
    train_nonincreasing = all(
        checkpoint_train_loss[i] >= checkpoint_train_loss[i + 1] - tolerance
        for i in range(len(checkpoint_train_loss) - 1)
    )
    val_rose_from_some_earlier_point = checkpoint_val_loss[-1] > min(checkpoint_val_loss[:-1])
    return bool(train_nonincreasing and val_rose_from_some_earlier_point)


def _save_training_state(ckpt_path: Path, optimizer, state: dict) -> None:
    import torch

    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    with open(ckpt_path / "training_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _load_training_state(ckpt_path: Path) -> dict:
    with open(ckpt_path / "training_state.json", encoding="utf-8") as f:
        return json.load(f)


def run(phase: int, model_cfg: ModelUnderTest = None, train_cfg: TrainingConfig = None,
        data_cfg: DataConfig = None, force: bool = False, resume_from_checkpoint: Optional[str] = None,
        version: str = None, per_device_batch_size_override: int = None) -> dict:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    model_cfg = model_cfg or MODELS_UNDER_TEST[0]
    train_cfg = train_cfg or SHARED_TRAINING_CONFIG
    data_cfg = data_cfg or DataConfig()
    version = version or PIPELINE_VERSION

    assert phase in (2, 3), "phase must be 2 or 3"

    # Resume support: if this exact (model, dataset_size, phase) run already
    # finished (a train_summary.json exists at its final checkpoint), reuse
    # it instead of re-downloading/re-training from scratch — this is what
    # lets `orchestrator.py run-full` be stopped and safely re-run later,
    # picking up at the next incomplete step rather than starting over.
    # `version` is namespaced into both the run_name and the directory itself
    # so different pipeline generations (v1/v2/v3) never collide on disk even
    # when re-run against the same model/dataset_size/phase.
    run_name = f"{model_cfg.name}_{version}_{format_dataset_size(data_cfg.num_samples)}_phase{phase}"
    existing_summary_path = CHECKPOINT_DIR / version / run_name / "train_summary.json"
    if existing_summary_path.exists() and not force and not resume_from_checkpoint:
        with open(existing_summary_path, encoding="utf-8") as f:
            print(f"[train] {run_name} already trained — reusing existing checkpoint (pass force=True to redo).")
            return json.load(f)

    records, val_records = load_phase_data(phase, data_cfg)

    device = resolve_device(model_cfg.device)
    tokenizer, base_model = load_tokenizer_and_model(model_cfg)
    base_model = base_model.to(device)

    # Per-model-tier LoRA rank/alpha (v3+) — replaces the single shared
    # train_cfg.lora_r/lora_alpha. See config.RANK_TIERS/rank_alpha_for.
    total_params = sum(p.numel() for p in base_model.parameters())
    rank_tier, lora_r, lora_alpha = rank_alpha_for(total_params)

    # Per-model batch size / gradient accumulation (v3+), targeting an
    # identical effective batch size across every model — replaces the
    # single shared train_cfg.per_device_train_batch_size/
    # gradient_accumulation_steps. See config.BATCH_CONFIG/batch_settings_for.
    # per_device_batch_size_override exists only for the isolated
    # batching-correctness verification (comparing batch_size=1 against
    # batch_size=N on the same model/data) — real runs never pass it.
    if per_device_batch_size_override is not None:
        per_device_batch_size = per_device_batch_size_override
        gradient_accumulation_steps = 1
    else:
        per_device_batch_size, gradient_accumulation_steps, _ = batch_settings_for(model_cfg.name)

    # A tuple/list of exact names is PEFT's plain suffix-match mode; a single
    # string is instead treated as a full regex, needed when bare names alone
    # would collide with same-named submodules elsewhere in the model (e.g.
    # a multimodal model's vision/audio towers) — see ModelUnderTest.lora_target_modules.
    target_modules = (
        model_cfg.lora_target_modules if isinstance(model_cfg.lora_target_modules, str)
        else list(model_cfg.lora_target_modules)
    )

    if resume_from_checkpoint:
        # Real resume: reload the adapter weights actually trained so far
        # (not a fresh LoRA init) and pick the optimizer/step state back up —
        # skip-if-done only ever restarted from scratch or reused a finished
        # run; this instead continues an interrupted run from where it left off.
        model = PeftModel.from_pretrained(base_model, resume_from_checkpoint, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=train_cfg.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    model.train()

    examples = [
        build_example_tensors(tokenizer, r["messages"], train_cfg.max_seq_length,
                               expects_chat_template=model_cfg.expects_chat_template)
        for r in records
    ]
    val_examples = [
        build_example_tensors(tokenizer, r["messages"], train_cfg.max_seq_length,
                               expects_chat_template=model_cfg.expects_chat_template)
        for r in val_records
    ]

    num_batches_per_epoch = -(-len(examples) // per_device_batch_size)  # ceil div — last batch may be smaller
    total_steps = num_batches_per_epoch * train_cfg.num_train_epochs
    checkpoint_steps = sorted({max(1, round(total_steps * f)) for f in train_cfg.checkpoint_fractions})

    # run_name already computed above (namespaced by version + dataset size —
    # same "n050"/"n500" convention as data/ and results/ — so re-training
    # the same model/phase at a different sample count or pipeline generation
    # never overwrites a previous run's checkpoints or loss log).
    ckpt_root = CHECKPOINT_DIR / version / run_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    log_dir = LOG_DIR / version
    log_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = log_dir / f"{run_name}_loss.jsonl"

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay
    )

    # Early stopping only engages when a val split exists — it stops training
    # at the first checkpoint where val_loss regresses above the best value
    # seen so far, then uses that best checkpoint (not the current, worse
    # weights) as "final". No patience window: one regression is enough.
    has_val = bool(val_examples)
    best_val_loss = float("inf")
    best_ckpt_path = None
    early_stopped = False
    early_stopped_at_step = None
    step = 0
    loss_history = []
    checkpoint_train_loss = []
    checkpoint_val_loss = []  # entries stay None per-checkpoint if val_examples is empty
    loss_log_mode = "w"

    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", map_location=device))
        state = _load_training_state(resume_path)
        step = state["step"]
        loss_history = state["loss_history"]
        checkpoint_train_loss = state["checkpoint_train_loss"]
        checkpoint_val_loss = state["checkpoint_val_loss"]
        best_val_loss = state["best_val_loss"] if state["best_val_loss"] is not None else float("inf")
        best_ckpt_path = Path(state["best_ckpt_path"]) if state["best_ckpt_path"] else None
        loss_log_mode = "a"  # continue the existing log rather than truncating it
        print(f"[train] resuming {run_name} from step {step} ({resume_from_checkpoint})")

    # Flattened (epoch, batch_idx, batch) sequence lets resume skip already-
    # completed steps by slicing, instead of re-deriving epoch/batch indices
    # from a raw step count. Each batch is a list of (input_ids, labels)
    # tuples of up to per_device_batch_size examples — the last batch in an
    # epoch may be smaller if len(examples) doesn't divide evenly.
    def _batches(example_list):
        return [example_list[i:i + per_device_batch_size] for i in range(0, len(example_list), per_device_batch_size)]

    flat_batches = [
        (epoch, batch_idx, batch)
        for epoch in range(train_cfg.num_train_epochs)
        for batch_idx, batch in enumerate(_batches(examples))
    ]

    with open(loss_log_path, loss_log_mode, encoding="utf-8") as loss_log:
        for epoch, batch_idx, batch in flat_batches[step:]:
            if early_stopped:
                break
            step += 1
            input_ids_b, labels_b, attention_mask_b = collate_batch(batch, tokenizer.pad_token_id)
            input_ids_b = input_ids_b.to(device)
            labels_b = labels_b.to(device)
            attention_mask_b = attention_mask_b.to(device)

            outputs = model(input_ids=input_ids_b, attention_mask=attention_mask_b, labels=labels_b)
            loss = outputs.loss
            loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_value = float(loss.item())
            loss_history.append(loss_value)
            record = {
                "step": step,
                "epoch": epoch,
                "batch_idx": batch_idx,
                "batch_size": len(batch),
                "loss": loss_value,
                "examples_seen": step * per_device_batch_size,
            }
            loss_log.write(json.dumps(record) + "\n")
            loss_log.flush()

            if step in checkpoint_steps:
                pct = round(100 * step / total_steps)
                ckpt_path = ckpt_root / f"step_{step}_pct_{pct}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

                checkpoint_train_loss.append(loss_value)
                val_loss_here = evaluate_avg_loss(model, val_examples, device) if val_examples else None
                checkpoint_val_loss.append(val_loss_here)

                if has_val:
                    if val_loss_here < best_val_loss:
                        best_val_loss = val_loss_here
                        best_ckpt_path = ckpt_path
                    else:
                        early_stopped = True
                        early_stopped_at_step = step

                # Saved at every checkpoint (not just on early stop) so a
                # --resume-from-checkpoint pointed at ANY intermediate
                # checkpoint dir has what it needs to continue.
                _save_training_state(ckpt_path, optimizer, {
                    "step": step, "loss_history": loss_history,
                    "checkpoint_train_loss": checkpoint_train_loss,
                    "checkpoint_val_loss": checkpoint_val_loss,
                    "best_val_loss": best_val_loss if best_ckpt_path is not None else None,
                    "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
                })

                if early_stopped:
                    break

    final_path = ckpt_root / "final"
    if early_stopped and best_ckpt_path is not None:
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.copytree(best_ckpt_path, final_path)
    else:
        model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)

    flat = _flat_or_nondecreasing(loss_history)
    overfitting = check_overfitting(checkpoint_train_loss, checkpoint_val_loss) if has_val else False

    summary = {
        "run_name": run_name,
        "version": version,
        "rank_tier": rank_tier,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_batch_size * gradient_accumulation_steps,
        "phase": phase,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "final_checkpoint": str(final_path),
        "loss_log": str(loss_log_path),
        "first_loss": loss_history[0] if loss_history else None,
        "last_loss": loss_history[-1] if loss_history else None,
        "loss_curve_flat_or_nondecreasing": flat,
        "val_examples_used": len(val_examples),
        # Explicit None (not omitted) when val was skipped, so downstream
        # consumers (orchestrator's loss_curve.json, report.py) can check for
        # presence cleanly rather than guessing from a missing key.
        "checkpoint_curve": {
            "steps": checkpoint_steps,
            "train_loss": checkpoint_train_loss,
            "val_loss": checkpoint_val_loss if has_val else None,
        },
        "possible_overfitting": overfitting,
        "early_stopped": early_stopped,
        "early_stopped_at_step": early_stopped_at_step,
        "best_val_loss": best_val_loss if has_val and best_ckpt_path is not None else None,
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
    parser.add_argument("--num-samples", type=int, default=None,
                         help="Must match the --num-samples you passed to data_prep.py — "
                              "it's what selects data/n{size}/phase{N}/train.jsonl. "
                              "Defaults to DataConfig.num_samples in config.py if omitted.")
    parser.add_argument("--tag", type=str, default=None,
                         help="Appends '-{tag}' to the model name for checkpoint/results naming "
                              "only (e.g. --tag v2 -> checkpoints/{model}-v2_n{size}_phase{N}/), "
                              "so a re-tuned retrain never overwrites the original run.")
    parser.add_argument("--force", action="store_true",
                         help="Redo training even if a completed checkpoint for this exact "
                              "(model, dataset_size, phase) already exists. Omit this to make "
                              "re-running the same command resume-safe (already-done runs are skipped).")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                         help="Path to an intermediate checkpoint dir (e.g. "
                              "checkpoints/{model}_n{size}_phase{N}/step_31_pct_25) — loads its "
                              "adapter weights, optimizer state, and step counter, and continues "
                              "training from that step rather than restarting from scratch.")
    parser.add_argument("--version", type=str, default=None,
                         help="Pipeline generation to write under (checkpoints/{version}/, "
                              "logs/{version}/) — defaults to config.PIPELINE_VERSION.")
    args = parser.parse_args()

    model_cfg = MODELS_UNDER_TEST[0]
    if args.model:
        matches = [m for m in MODELS_UNDER_TEST if m.name == args.model]
        if not matches:
            raise SystemExit(f"Unknown model name: {args.model}")
        model_cfg = matches[0]
    if args.tag:
        model_cfg = dataclasses.replace(model_cfg, name=f"{model_cfg.name}-{args.tag}")

    data_cfg = DataConfig(num_samples=args.num_samples) if args.num_samples is not None else None

    summary = run(args.phase, model_cfg=model_cfg, data_cfg=data_cfg, force=args.force,
                  resume_from_checkpoint=args.resume_from_checkpoint, version=args.version)
    print(json.dumps(summary, indent=2))
    if summary["loss_curve_flat_or_nondecreasing"]:
        print("WARNING: loss curve is flat or non-decreasing — possible training issue.")


if __name__ == "__main__":
    main()
