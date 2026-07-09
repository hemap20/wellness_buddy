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
from model_utils import format_prompt, resolve_device


class ModelRunner:
    """Loads a model once; call .generate() many times (e.g. across a
    multi-turn simulator conversation) without reloading weights."""

    def __init__(self, model_cfg: ModelUnderTest, adapter_path: Optional[str] = None):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_cfg = model_cfg
        self.adapter_path = adapter_path
        self.device = resolve_device(model_cfg.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.model = base_model
        self.model.to(self.device)
        self.model.eval()

    def generate(self, messages: list[dict]) -> tuple[str, float]:
        """messages: prior conversation, ending in a user turn (no trailing
        assistant turn). Returns (assistant_text, latency_seconds)."""
        import torch

        prompt = format_prompt(self.tokenizer, messages, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

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
