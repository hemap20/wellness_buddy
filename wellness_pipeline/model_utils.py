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


def format_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True,
                   enable_thinking: Optional[bool] = None) -> str:
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}.
    enable_thinking is only meaningful for models whose chat template checks
    that Jinja variable (Qwen3/Gemma-4-style) — pass None for every other
    model and it's simply not sent."""
    template_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
            )
        except Exception as exc:
            # Some chat templates (older Gemma releases, some Llama variants)
            # reject a "system" role turn outright. Fold it into the first
            # user turn and retry once rather than failing the whole run —
            # this is generic so it silently no-ops for models that DO
            # support system role (their apply_chat_template call above
            # would have already succeeded).
            if any(m["role"] == "system" for m in messages):
                merged = _merge_system_into_first_user(messages)
                return tokenizer.apply_chat_template(
                    merged, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
                )
            raise RuntimeError(f"tokenizer.apply_chat_template failed: {exc}") from exc
    return _fallback_format(messages, add_generation_prompt)


def _merge_system_into_first_user(messages: list[dict]) -> list[dict]:
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    if not system_parts or not rest or rest[0]["role"] != "user":
        # Nothing sensible to merge into — return as-is and let the template
        # raise its own (now unmodified) error.
        return messages
    merged_content = "\n\n".join(system_parts) + "\n\n" + rest[0]["content"]
    return [{"role": "user", "content": merged_content}] + rest[1:]


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


def load_tokenizer_and_model(model_cfg, for_training: bool = False):
    """Shared load path for train.py/inference.py so gated-repo errors get one
    clear message instead of a raw 401 traceback in three different places."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = None
    if getattr(model_cfg, "torch_dtype", None):
        import torch

        dtype = getattr(torch, model_cfg.torch_dtype)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code, dtype=dtype
        )
    except OSError as exc:
        if getattr(model_cfg, "gated", False) or "gated repo" in str(exc).lower() or "401" in str(exc):
            raise RuntimeError(
                f"Could not load {model_cfg.hf_model_id!r} — this repo is gated. "
                f"1) Accept the license at https://huggingface.co/{model_cfg.hf_model_id}, "
                f"2) authenticate via `huggingface-cli login` or `export HF_TOKEN=...`, then retry."
            ) from exc
        raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def resolve_device(device_pref: str) -> str:
    import torch

    if device_pref != "auto":
        return device_pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
