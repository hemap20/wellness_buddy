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


class ThinkingTruncatedError(RuntimeError):
    """Raised when generation ran out of max_new_tokens while still inside
    the thinking segment (no closing marker) — there's no valid final answer
    to return, so callers should treat this as a generation failure, not a
    real response, rather than silently leaking the raw reasoning trace."""


def _strip_thinking_channel(text: str, tokenizer, open_marker: str, close_marker: str) -> str:
    """Generic thinking-segment stripper, parameterized by each model's own
    open/close markers (see ModelUnderTest.thinking_markers) — different
    architectures delimit their reasoning trace differently:
      - gemma-4-e2b-it: "<|channel>thought\\n...\\n<channel|>answer"
      - Qwen3-family (e.g. empathetic-qwen3-8b-jan): "<think>...</think>answer"
    Both follow the same open-marker...close-marker...visible-answer shape,
    so one generic split-based implementation covers both — this removes
    the segment, keeping only the final visible answer, same as what
    thinking=off output would show.

    Raises ThinkingTruncatedError if a thinking segment was opened but never
    closed (generation ran out of budget mid-thought) — there is no final
    answer in that case, so returning the raw text as if it were one would
    silently feed the judge/report a chain-of-thought dump instead of a
    real response."""
    if open_marker in text and close_marker not in text:
        raise ThinkingTruncatedError(
            f"generation ran out of max_new_tokens while still inside the thinking segment "
            f"(opened with {open_marker!r} but never closed) — no final answer was produced. "
            f"Raise ModelUnderTest.thinking_max_new_tokens for this model."
        )
    if close_marker in text:
        parts = text.split(close_marker)
        kept = []
        for part in parts:
            if open_marker in part:
                kept.append(part.split(open_marker, 1)[0])
            else:
                kept.append(part)
        text = "".join(kept)
    for special in tokenizer.all_special_tokens:
        text = text.replace(special, "")
    return text.strip()


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

    def generate(self, messages: list[dict], thinking: Optional[bool] = None) -> tuple[str, float]:
        """messages: prior conversation, ending in a user turn (no trailing
        assistant turn). Returns (assistant_text, latency_seconds).

        `thinking` only applies to models with model_cfg.supports_thinking —
        toggles the chat template's enable_thinking variable, and (when the
        model actually emits a thinking segment) strips it from the returned
        text so transcripts are comparable across thinking on/off."""
        import torch

        enable_thinking = thinking if self.model_cfg.supports_thinking else None
        prompt = format_prompt(self.tokenizer, messages, add_generation_prompt=True,
                                enable_thinking=enable_thinking)
        # Thinking needs real headroom for a full reasoning trace plus a
        # final answer — using the plain max_new_tokens budget for it (as an
        # earlier version of this code did) truncates generation mid-thought
        # before any answer is produced. Reserve prompt-truncation room based
        # on whichever budget this call actually uses.
        effective_max_new_tokens = (
            self.model_cfg.thinking_max_new_tokens
            if enable_thinking and self.model_cfg.thinking_max_new_tokens
            else self.model_cfg.max_new_tokens
        )
        max_prompt_tokens = max(1, self.context_window - effective_max_new_tokens)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens
        ).to(self.device)

        start = time.monotonic()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=effective_max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.monotonic() - start

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        if enable_thinking:
            # The thinking-channel delimiters are themselves single special
            # tokens, so decoding with skip_special_tokens=True would erase
            # the boundary markers while keeping the thinking prose merged
            # into the visible answer. Decode raw first, strip by marker,
            # then clean up any other special-token strings that survived.
            raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            open_marker, close_marker = self.model_cfg.thinking_markers
            try:
                text = _strip_thinking_channel(raw_text, self.tokenizer, open_marker, close_marker)
            except ThinkingTruncatedError:
                # Don't crash the whole simulate/score batch over one turn
                # that ran out of budget mid-thought — surface it as a
                # visibly-flagged non-answer instead of a silent CoT leak.
                text = "(thinking truncated before a final answer — increase thinking_max_new_tokens)"
        else:
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            text = "(no response generated)"

        # Free cached (but no-longer-referenced) MPS memory back to the OS.
        # Without this, PyTorch's MPS allocator pools freed tensors rather
        # than releasing them, so peak memory can climb turn-over-turn across
        # a long multi-turn conversation (each turn re-feeds a growing
        # history, and thinking-mode turns generate large intermediate
        # tensors) until the OS's low-memory killer silently SIGKILLs the
        # process — discovered the hard way on empathetic-qwen3-8b-jan, which
        # died at the same point (early into a second thinking-mode
        # conversation) on two separate runs.
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
    parser.add_argument("--thinking", type=str, default=None, choices=["on", "off"],
                         help="Only applies to models with supports_thinking=True in config.py.")
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

    thinking = {"on": True, "off": False, None: None}[args.thinking]
    text, latency = runner.generate(messages, thinking=thinking)
    print(json.dumps({"response": text, "latency_seconds": latency, "thinking": thinking}, indent=2))


if __name__ == "__main__":
    main()
