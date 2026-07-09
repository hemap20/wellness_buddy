"""
Shared, model-agnostic helpers for turning {system?, user, assistant} messages
into a single prompt string and back. Used by train.py and inference.py so
formatting is identical between training and inference.

If a model's tokenizer ships a chat_template (most instruction-tuned models),
that's used. DialoGPT-small has no chat template, so we fall back to a plain
"System: ...\nUser: ...\nBot: " format — this keeps the pipeline compatible
with any future model, chat-template or not.
"""
from typing import Optional


def format_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}"""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return _fallback_format(messages, add_generation_prompt)


def _fallback_format(messages: list[dict], add_generation_prompt: bool) -> str:
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Bot: {content}")
    text = "\n".join(lines)
    if add_generation_prompt:
        text += "\nBot:"
    return text


def build_training_text(tokenizer, messages: list[dict]) -> tuple[str, str]:
    """Returns (prompt_text, full_text) where prompt_text is everything up to
    but not including the assistant's reply, and full_text includes it. Used
    to compute the loss mask (only assistant tokens are trained on)."""
    assert messages[-1]["role"] == "assistant"
    prompt_messages = messages[:-1]
    prompt_text = format_prompt(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = prompt_text + " " + messages[-1]["content"]
    return prompt_text, full_text


def resolve_device(device_pref: str) -> str:
    import torch

    if device_pref != "auto":
        return device_pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
