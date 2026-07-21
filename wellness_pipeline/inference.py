"""
Inference module — standalone runnable stage.

Wraps a base (optionally LoRA-adapted) HF causal LM behind a simple
.generate(messages) -> (text, latency_seconds) interface, so the simulator
and scoring stages don't need to know anything about tokenization.

Run standalone (single-turn smoke test):
    python inference.py --model dialogpt-small
    python inference.py --model dialogpt-small --adapter checkpoints/dialogpt-small_phase2/final \
        --system-prompt "You are a warm, supportive wellness companion. Keep responses casual and conversational."
"""
import argparse
import json
import time
from pathlib import Path
from typing import Optional

from config import MODELS_UNDER_TEST, ModelUnderTest
from model_utils import format_prompt, load_tokenizer_and_model, resolve_device


class ModelRunner:
    """Loads a model once; call .generate() many times (e.g. across a
    multi-turn simulator conversation) without reloading weights."""

    def __init__(self, model_cfg: ModelUnderTest, adapter_path: Optional[str] = None):
        self.model_cfg = model_cfg
        self.adapter_path = adapter_path
        self.device = resolve_device(model_cfg.device)

        self.tokenizer, base_model = load_tokenizer_and_model(model_cfg)

        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.to(self.device)
        self.model.eval()

        # Context window in tokens. GPT-2-family configs expose this under
        # several aliases; fall back to 1024 (GPT-2's own default) if none apply.
        config = self.model.config
        self.context_window = (
            getattr(config, "max_position_embeddings", None)
            or getattr(config, "n_positions", None)
            or getattr(config, "n_ctx", None)
            or 1024
        )

    def generate(self, messages: list[dict]) -> tuple[str, float]:
        """messages: prior conversation, ending in a user turn (no trailing
        assistant turn). Returns (assistant_text, latency_seconds)."""
        import torch

        prompt = format_prompt(self.tokenizer, messages, add_generation_prompt=True)
        max_prompt_tokens = max(1, self.context_window - self.model_cfg.max_new_tokens)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
        ).to(self.device)

        start = time.monotonic()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.model_cfg.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.monotonic() - start

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            text = "(no response generated)"

        # Free cached (but no-longer-referenced) MPS memory back to the OS.
        # Without this, PyTorch's MPS allocator pools freed tensors rather
        # than releasing them, so peak memory can climb turn-over-turn across
        # a long multi-turn conversation (each turn re-feeds a growing
        # history) until the OS's low-memory killer silently SIGKILLs the
        # process.
        del output_ids, new_tokens, inputs
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        return text, latency


def main():
    parser = argparse.ArgumentParser(description="Inference stage: run a single generation for inspection.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--user-message", type=str, default="I've had a really rough day.")
    args = parser.parse_args()

    model_cfg = MODELS_UNDER_TEST[0]
    if args.model:
        matches = [m for m in MODELS_UNDER_TEST if m.name == args.model]
        if not matches:
            raise SystemExit(f"Unknown model name: {args.model}")
        model_cfg = matches[0]

    runner = ModelRunner(model_cfg, adapter_path=args.adapter)

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.user_message})

    text, latency = runner.generate(messages)
    print(json.dumps({"response": text, "latency_seconds": latency}, indent=2))


if __name__ == "__main__":
    main()
