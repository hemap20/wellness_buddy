"""
Thin, provider-agnostic wrapper around the LLM used for the simulator and
judge roles. Supports Anthropic (Claude) and Google (Gemini) — pick per
role via SimulatorConfig.provider / JudgeConfig.provider in config.py.

This is the only place that talks to either vendor SDK directly; simulator.py
and judge.py just call .generate_json() / .continue_json().

Env vars:
  ANTHROPIC_API_KEY            (or `ant auth login`) for provider="anthropic"
  GEMINI_API_KEY / GOOGLE_API_KEY                     for provider="gemini"
"""
import json
import os
from typing import Optional


def _strip_for_gemini_schema(schema: dict) -> dict:
    """Gemini's response_schema is an OpenAPI-3 subset — drop keys the
    Anthropic-style JSON Schema uses that Gemini doesn't accept."""
    schema = dict(schema)
    schema.pop("additionalProperties", None)
    if "properties" in schema:
        schema["properties"] = {k: _strip_for_gemini_schema(v) if isinstance(v, dict) else v
                                 for k, v in schema["properties"].items()}
    return schema


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic()
        elif provider == "gemini":
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY (or GOOGLE_API_KEY) must be set in the environment to use provider='gemini'."
                )
            self._client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'anthropic' or 'gemini'.")

    # -- single-turn, plain text -------------------------------------------------
    def generate_text(self, system: str, user_content: str, max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            return next(b.text for b in resp.content if b.type == "text")
        else:
            from google.genai import types

            resp = self._client.models.generate_content(
                model=self.model, contents=user_content,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
            )
            return resp.text or ""

    # -- single-turn, JSON output (optionally schema-constrained) ---------------
    def generate_json(self, system: str, user_content: str, schema: Optional[dict] = None,
                       max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            kwargs = {}
            if schema is not None:
                kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user_content}], **kwargs,
            )
            return next(b.text for b in resp.content if b.type == "text")
        else:
            from google.genai import types

            config_kwargs = dict(
                system_instruction=system, max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
            if schema is not None:
                config_kwargs["response_schema"] = _strip_for_gemini_schema(schema)
            resp = self._client.models.generate_content(
                model=self.model, contents=user_content, config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp.text or ""

    # -- multi-turn JSON (used by judge.py's malformed-JSON retry loop) ---------
    def continue_json(self, system: str, history: list[dict], max_tokens: int = 1024) -> str:
        """history: list of {"role": "user"|"assistant", "content": str}."""
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system, messages=history,
            )
            return next(b.text for b in resp.content if b.type == "text")
        else:
            from google.genai import types

            contents = [
                types.Content(role=("model" if turn["role"] == "assistant" else "user"),
                               parts=[types.Part(text=turn["content"])])
                for turn in history
            ]
            resp = self._client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens),
            )
            return resp.text or ""
