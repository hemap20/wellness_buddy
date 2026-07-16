"""
Thinking-Mode Compatibility Probe — standalone, additive experiment.

WHY THIS FILE IS SEPARATE FROM THE MAIN PIPELINE
-------------------------------------------------
There can be an important `orchestrator.py run-full` job already running
(train/simulate/score across every model in MODELS_UNDER_TEST). This script
must never touch that job or the code it depends on, so it:
  - only IMPORTS config.py / data_prep.py / model_utils.py / llm_client.py /
    results_manager.py as read-only library code (same as orchestrator.py
    itself does) — it never edits them, never calls train.run()/data_prep.run()
    (which write into the SHARED checkpoints/, data/, results/ trees).
  - writes every artifact of its own under a private `thinking_probe/`
    directory (data/checkpoints/results/logs), so it cannot overwrite or
    race with anything the shared pipeline is producing.

WHAT THIS TESTS
---------------
Applies only to candidate models with a native thinking/non-thinking toggle
(ModelUnderTest.supports_thinking=True in config.py — currently
gemma-4-e2b-it, empathetic-qwen3-8b-jan, qwen3-8b; THINKING_MODELS below is
computed at import time, so adding a 4th such model to config.py and
re-running this script picks it up automatically, no code change needed).

Hypothesis: after a tiny 50-example exposure, a model that (a) keeps its
<think> format well-formed, (b) doesn't regress non-thinking quality, and
(c) produces measurably better responses in thinking mode specifically on
ambiguous emotional disclosures is a stronger final-model candidate than one
that only clears (a)+(b), or shows no real (c).

ONE COMMAND DOES EVERYTHING (meant for an unattended run on another machine):
    python thinking_mode_probe.py run-all

That single command: builds the 50-example mixed training set (reusing the
same HF dataset id + seed as config.DataConfig, adding synthetic <think>
traces itself since the raw dataset has none) -> builds the ~15-20 item
held-out probe set -> LoRA fine-tunes each thinking-capable model on the
mixed set -> runs the probe-set evaluation (format fidelity / mode
obedience / non-thinking regression / reasoning-trace substance /
thinking-mode value-add / efficiency cost) -> writes a report -> `git add`
(only the files this script created) + commit + push.

Requires the same env this pipeline already needs for its judge/simulator
LLM calls: GEMINI_API_KEY (or ANTHROPIC_API_KEY, matching JUDGE_CONFIG.provider
in config.py) in .env, since <think> trace synthesis and scoring both go
through the same LLMClient judge.py already uses.

Dry run (tiny: 1 model unless overridden, 6 train examples, 4 probe items,
1 epoch, short generations — verifies the whole plumbing end-to-end fast,
without a real training run or git push):
    python thinking_mode_probe.py run-all --dry-run
"""
import argparse
import dataclasses
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from config import (
    FIXED_SYSTEM_PROMPT,
    MODELS_UNDER_TEST,
    ModelUnderTest,
    DataConfig,
    JUDGE_CONFIG,
    ROOT_DIR,
)
from llm_client import LLMClient
from model_utils import build_training_text, format_prompt, load_tokenizer_and_model, resolve_device
import data_prep

# ---------------------------------------------------------------------------
# Private paths — never overlaps with checkpoints/, data/, results/, logs/
# ---------------------------------------------------------------------------
PROBE_ROOT = ROOT_DIR / "thinking_probe"
PROBE_DATA_DIR = PROBE_ROOT / "data"
PROBE_CKPT_DIR = PROBE_ROOT / "checkpoints"
PROBE_RESULTS_DIR = PROBE_ROOT / "results"
PROBE_LOG_DIR = PROBE_ROOT / "logs"
PROBE_TRANSCRIPT_DIR = PROBE_ROOT / "transcripts"
PROBE_SCORE_DIR = PROBE_ROOT / "scores"
for _d in (PROBE_DATA_DIR, PROBE_CKPT_DIR, PROBE_RESULTS_DIR, PROBE_LOG_DIR, PROBE_TRANSCRIPT_DIR, PROBE_SCORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_SET_PATH = PROBE_DATA_DIR / "mixed_train.jsonl"
PROBE_SET_PATH = PROBE_DATA_DIR / "probe_set.jsonl"
REPORT_JSON_PATH = PROBE_ROOT / "report.json"
REPORT_MD_PATH = PROBE_ROOT / "REPORT.md"

# Reuses config.py's own DataConfig seed/dataset id (only num_samples is
# bumped up from the pipeline's default so there are enough (user, assistant)
# exchanges to fill 50 train + ~20 held-out probe items — same dataset,
# same seed, just a larger deterministic slice of it).
PROBE_DATA_CFG = DataConfig(num_samples=120)

N_THINK = 20
N_NO_THINK = 20
N_SWITCH = 10
N_PROBE_EASY = 4
N_PROBE_HARD = 6
N_PROBE_FLAG = 4
N_PROBE_SWITCH = 3


def thinking_models() -> list[ModelUnderTest]:
    """Computed at call time, not hardcoded — add a new supports_thinking=True
    model to config.py and it shows up here with zero changes to this file."""
    return [m for m in MODELS_UNDER_TEST if m.supports_thinking]


def _judge_client() -> LLMClient:
    return LLMClient(JUDGE_CONFIG.provider, JUDGE_CONFIG.model)


# ---------------------------------------------------------------------------
# Step 1 — build the 50-example mixed training set
# ---------------------------------------------------------------------------

def _synthesize_think_trace(client: LLMClient, user_text: str, assistant_text: str) -> str:
    """Generates a genuine reasoning trace (EPITOME-style: emotional reaction,
    interpretation of the underlying situation, exploration/strategy fit)
    that would plausibly precede the given (already-written) response —
    the raw dataset has no traces, so this fills exactly that gap."""
    system = (
        "You write the hidden reasoning trace of an empathetic wellness-support "
        "assistant, EPITOME-style: (1) what emotional reaction is the user "
        "showing, (2) what's the likely underlying situation/need they haven't "
        "stated outright, (3) what support strategy fits. 3-5 sentences, first "
        "person as the assistant's internal reasoning, no meta-commentary, "
        "never repeat the final answer verbatim."
    )
    prompt = (
        f"User said: {user_text!r}\n"
        f"The assistant's eventual visible reply is: {assistant_text!r}\n\n"
        "Write only the internal reasoning trace that would lead to that reply."
    )
    return client.generate_text(system, prompt, max_tokens=300).strip()


def build_mixed_training_set(dry_run: bool = False) -> list[dict]:
    """Returns a list of {"user", "kind", "turns"} records, where "turns" is
    a list of (user_text_with_flag, think_trace_or_None, assistant_text)
    tuples — one tuple per exchange, in order. kind is "think" | "no_think" |
    "switch". Model-specific <think>/</think>-style markers are substituted
    in later, at train-record materialization time (see materialize_records),
    so this set is shared, marker-agnostic, and built once regardless of how
    many thinking-capable models are being probed."""
    if TRAIN_SET_PATH.exists() and not dry_run:
        with open(TRAIN_SET_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    n_think, n_no_think, n_switch = (2, 2, 2) if dry_run else (N_THINK, N_NO_THINK, N_SWITCH)

    raw_samples = data_prep.load_raw_samples(PROBE_DATA_CFG)
    pairs = data_prep.build_pairs(raw_samples)
    needed = n_think + n_no_think + n_switch * 2
    if len(pairs) < needed:
        raise RuntimeError(
            f"Only {len(pairs)} (user, assistant) exchanges available from "
            f"num_samples={PROBE_DATA_CFG.num_samples}; need >= {needed}. "
            "Raise PROBE_DATA_CFG.num_samples in thinking_mode_probe.py."
        )
    indices = list(range(len(pairs)))
    random.Random(PROBE_DATA_CFG.seed).shuffle(indices)

    client = _judge_client()
    records: list[dict] = []
    cursor = 0

    for _ in range(n_think):
        user, assistant = pairs[indices[cursor]]; cursor += 1
        trace = "(dry-run trace)" if dry_run else _synthesize_think_trace(client, user, assistant)
        records.append({"kind": "think", "turns": [(f"{user} /think", trace, assistant)]})

    for _ in range(n_no_think):
        user, assistant = pairs[indices[cursor]]; cursor += 1
        records.append({"kind": "no_think", "turns": [(f"{user} /no_think", "", assistant)]})

    for _ in range(n_switch):
        user1, assistant1 = pairs[indices[cursor]]; cursor += 1
        user2, assistant2 = pairs[indices[cursor]]; cursor += 1
        flip = cursor % 2 == 0  # alternate which flag comes first across examples
        flag1, flag2 = ("/no_think", "/think") if flip else ("/think", "/no_think")
        trace1 = "" if flag1 == "/no_think" else ("(dry-run trace)" if dry_run else _synthesize_think_trace(client, user1, assistant1))
        trace2 = "" if flag2 == "/no_think" else ("(dry-run trace)" if dry_run else _synthesize_think_trace(client, user2, assistant2))
        records.append({
            "kind": "switch",
            "turns": [(f"{user1} {flag1}", trace1, assistant1), (f"{user2} {flag2}", trace2, assistant2)],
        })

    with open(TRAIN_SET_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


# ---------------------------------------------------------------------------
# Step 2 — build the held-out probe set (never trained on — disjoint indices)
# ---------------------------------------------------------------------------

def _synthesize_ambiguous_prompt(client: LLMClient, base_user_text: str) -> str:
    """Theory-of-mind-style reframe: turns a plain disclosure into one where
    the real need is implied rather than stated, so thinking mode has
    something genuine to infer."""
    system = (
        "Rewrite the given message from a wellness-support user so the "
        "underlying feeling/need is implied, not stated outright — an "
        "indirect disclosure a careful listener has to infer, not a plain "
        "complaint. Keep it under 3 sentences, first person, casual tone."
    )
    return client.generate_text(system, base_user_text, max_tokens=150).strip()


def build_probe_set(dry_run: bool = False) -> list[dict]:
    """Returns a list of probe items: {"id", "category", "turns": [(user_text,
    thinking_flag_or_None), ...]}. thinking_flag: True forces /think, False
    forces /no_think, None means the item isn't testing flag mechanics —
    caller runs it under whatever mode the current experiment step needs."""
    if PROBE_SET_PATH.exists() and not dry_run:
        with open(PROBE_SET_PATH, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    n_easy, n_hard, n_flag, n_switch = (
        (1, 1, 1, 1) if dry_run else (N_PROBE_EASY, N_PROBE_HARD, N_PROBE_FLAG, N_PROBE_SWITCH)
    )

    raw_samples = data_prep.load_raw_samples(PROBE_DATA_CFG)
    pairs = data_prep.build_pairs(raw_samples)
    train_needed = N_THINK + N_NO_THINK + N_SWITCH * 2
    probe_needed = n_easy + n_hard + n_flag + n_switch * 2
    indices = list(range(len(pairs)))
    random.Random(PROBE_DATA_CFG.seed).shuffle(indices)
    # Disjoint from the training indices — training used indices[:train_needed]
    # (in the un-dry-run-sized case) off the very same shuffle; slicing the
    # tail guarantees zero overlap regardless of dry_run's smaller counts.
    probe_pool_start = train_needed if not dry_run else len(indices) // 2
    probe_indices = indices[probe_pool_start:probe_pool_start + probe_needed]
    if len(probe_indices) < probe_needed:
        raise RuntimeError("Not enough held-out pairs left for the probe set — raise PROBE_DATA_CFG.num_samples.")

    client = _judge_client()
    items = []
    cursor = 0

    for i in range(n_easy):
        user, _ = pairs[probe_indices[cursor]]; cursor += 1
        items.append({"id": f"easy{i}", "category": "easy_baseline", "turns": [(user, None)]})

    for i in range(n_hard):
        user, _ = pairs[probe_indices[cursor]]; cursor += 1
        ambiguous = user if dry_run else _synthesize_ambiguous_prompt(client, user)
        items.append({"id": f"hard{i}", "category": "ambiguous_hard", "turns": [(ambiguous, None)]})

    for i in range(n_flag):
        user, _ = pairs[probe_indices[cursor]]; cursor += 1
        flag = i % 2 == 0
        items.append({"id": f"flag{i}", "category": "explicit_flag", "turns": [(user, flag)]})

    for i in range(n_switch):
        user1, _ = pairs[probe_indices[cursor]]; cursor += 1
        user2, _ = pairs[probe_indices[cursor]]; cursor += 1
        flip = i % 2 == 0
        items.append({
            "id": f"switch{i}", "category": "multi_turn_switch",
            "turns": [(user1, not flip), (user2, flip)],
        })

    with open(PROBE_SET_PATH, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


# ---------------------------------------------------------------------------
# Step 3 — LoRA fine-tune each thinking-capable model on the mixed set
# ---------------------------------------------------------------------------

def materialize_records(mixed: list[dict], model_cfg: ModelUnderTest) -> list[dict]:
    """Turns the shared, marker-agnostic mixed set into this model's own
    {"messages": [...]} chat records, using its native thinking_markers."""
    open_marker, close_marker = model_cfg.thinking_markers
    records = []
    for r in mixed:
        messages = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}]
        for user_text, trace, assistant_text in r["turns"]:
            messages.append({"role": "user", "content": user_text})
            think_block = f"{open_marker}{trace}{close_marker}"
            messages.append({"role": "assistant", "content": f"{think_block}\n\n{assistant_text}"})
        records.append({"messages": messages, "kind": r["kind"]})
    return records


def finetune_thinking_probe(model_cfg: ModelUnderTest, records: list[dict], dry_run: bool = False) -> dict:
    """Minimal, self-contained LoRA fine-tune loop (deliberately NOT calling
    train.run(), which is hardcoded to load data/n{size}/phase{2,3}/train.jsonl
    from the SHARED data/ tree — this experiment's records live only in
    memory + thinking_probe/data/, never in that shared tree)."""
    import torch
    from peft import LoraConfig, get_peft_model

    run_name = f"{model_cfg.name}-thinkprobe"
    ckpt_root = PROBE_CKPT_DIR / run_name
    final_path = ckpt_root / "final"
    if final_path.exists() and not dry_run:
        print(f"[thinkprobe] {run_name} already fine-tuned — reusing existing adapter.")
        return {"run_name": run_name, "final_checkpoint": str(final_path), "reused": True}

    device = resolve_device(model_cfg.device)
    tokenizer, base_model = load_tokenizer_and_model(model_cfg)
    base_model = base_model.to(device)

    target_modules = (
        model_cfg.lora_target_modules if isinstance(model_cfg.lora_target_modules, str)
        else list(model_cfg.lora_target_modules)
    )
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=target_modules, task_type="CAUSAL_LM")
    model = get_peft_model(base_model, lora_config)
    model.train()

    max_seq_length = 512 if dry_run else 768  # thinking traces run longer than plain phase2/3 examples
    examples = []
    for r in records:
        prompt_text, full_text = build_training_text(tokenizer, r["messages"])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        eos = tokenizer.eos_token_id
        if eos is not None:
            full_ids = full_ids + [eos]
        full_ids = full_ids[:max_seq_length]
        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels[:n_prompt] = -100
        examples.append((input_ids, labels))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    num_epochs = 1 if dry_run else 3
    loss_log_path = PROBE_LOG_DIR / f"{run_name}_loss.jsonl"
    start = time.monotonic()
    with open(loss_log_path, "w", encoding="utf-8") as loss_log:
        for epoch in range(num_epochs):
            for idx, (input_ids, labels) in enumerate(examples):
                input_ids_b = input_ids.unsqueeze(0).to(device)
                labels_b = labels.unsqueeze(0).to(device)
                outputs = model(input_ids=input_ids_b, labels=labels_b)
                outputs.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_log.write(json.dumps({"epoch": epoch, "example_idx": idx, "loss": float(outputs.loss.item())}) + "\n")
    train_seconds = time.monotonic() - start

    ckpt_root.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    peak_mem_mb = None
    if device == "cuda" and torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    summary = {
        "run_name": run_name,
        "final_checkpoint": str(final_path),
        "num_examples": len(examples),
        "num_epochs": num_epochs,
        "train_seconds": train_seconds,
        "peak_vram_mb": peak_mem_mb,
        "loss_log": str(loss_log_path),
        "reused": False,
    }
    with open(ckpt_root / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Step 4 — evaluate the fine-tuned model on the probe set against every metric
# ---------------------------------------------------------------------------

def _format_fidelity_ok(raw_text: str, open_marker: str, close_marker: str) -> tuple[bool, str]:
    n_open, n_close = raw_text.count(open_marker), raw_text.count(close_marker)
    if n_open == 0 and n_close == 0:
        return False, "no thinking markers present at all"
    if n_open != 1 or n_close != 1:
        return False, f"expected exactly one open/close pair, got open={n_open} close={n_close}"
    if raw_text.find(open_marker) > raw_text.find(close_marker):
        return False, "close marker appears before open marker"
    return True, "well-formed"


def _extract_trace_and_answer(raw_text: str, open_marker: str, close_marker: str) -> tuple[str, str]:
    if open_marker not in raw_text or close_marker not in raw_text:
        return "", raw_text.strip()
    before, _, rest = raw_text.partition(open_marker)
    trace, _, answer = rest.partition(close_marker)
    return trace.strip(), (before + answer).strip()


def _mode_obedience_ok(trace: str, expected_thinking: bool) -> bool:
    has_substantive_trace = len(trace.strip()) >= 15
    return has_substantive_trace if expected_thinking else not has_substantive_trace


def _judge_score(client: LLMClient, system: str, user_content: str) -> dict:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}, "justification": {"type": "string"}},
        "required": ["score", "justification"],
    }
    text = client.generate_json(system, user_content, schema=schema, max_tokens=400)
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


_EPITOME_TRACE_RUBRIC_SYSTEM = (
    "You score a hidden reasoning trace from a wellness-support assistant on a "
    "1-5 scale using the EPITOME framework: does it identify a real Emotional "
    "Reaction, a plausible Interpretation of the underlying situation/need, and "
    "does it Explore a strategy that actually fits — versus degenerate filler "
    "that restates the message without inference. Output STRICT JSON: "
    '{"score": <1-5>, "justification": "..."}.'
)

_NON_THINKING_QUALITY_SYSTEM = (
    "You score a wellness-support assistant's response on a 1-5 scale for "
    "persona warmth and empathetic specificity (not generic/parroting). "
    'Output STRICT JSON: {"score": <1-5>, "justification": "..."}.'
)

_VALUE_ADD_SYSTEM = (
    "You compare two candidate replies (A and B) to the same ambiguous "
    "emotional disclosure, where the real feeling/need is implied rather than "
    "stated. Judge which reply better infers and addresses the UNSTATED need. "
    'Output STRICT JSON: {"winner": "A"|"B"|"tie", "justification": "..."}.'
)


def evaluate_model(model_cfg: ModelUnderTest, probe_items: list[dict], dry_run: bool = False) -> dict:
    from inference import ModelRunner, ThinkingTruncatedError

    run_name = f"{model_cfg.name}-thinkprobe"
    adapter_path = PROBE_CKPT_DIR / run_name / "final"
    open_marker, close_marker = model_cfg.thinking_markers
    client = _judge_client()

    base_runner = ModelRunner(model_cfg, adapter_path=None)
    tuned_runner = ModelRunner(model_cfg, adapter_path=str(adapter_path))

    def _raw_generate(runner: ModelRunner, messages: list[dict], thinking: Optional[bool]) -> tuple[str, str, float]:
        """Bypasses ModelRunner.generate()'s own marker-stripping so format
        fidelity/mode obedience can be checked on the UNSTRIPPED text — the
        same generation is then reused (already-stripped) as the visible
        answer everywhere else."""
        import torch

        enable_thinking = thinking if model_cfg.supports_thinking else None
        prompt = format_prompt(runner.tokenizer, messages, add_generation_prompt=True, enable_thinking=enable_thinking)
        effective_max_new_tokens = (
            model_cfg.thinking_max_new_tokens if enable_thinking and model_cfg.thinking_max_new_tokens
            else model_cfg.max_new_tokens
        )
        if dry_run:
            effective_max_new_tokens = min(effective_max_new_tokens, 64)
        max_prompt_tokens = max(1, runner.context_window - effective_max_new_tokens)
        inputs = runner.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens).to(runner.device)
        start = time.monotonic()
        with torch.no_grad():
            output_ids = runner.model.generate(
                **inputs, max_new_tokens=effective_max_new_tokens, do_sample=True,
                temperature=0.8, top_p=0.95, pad_token_id=runner.tokenizer.pad_token_id,
            )
        latency = time.monotonic() - start
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_text = runner.tokenizer.decode(new_tokens, skip_special_tokens=False)
        return raw_text, prompt, latency

    format_fidelity_results = []
    mode_obedience_results = []
    trace_substance_scores = []
    latencies = {"thinking": [], "non_thinking": []}
    hard_item_pairs = []  # for thinking-mode value-add

    for item in probe_items:
        messages = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}]
        for turn_idx, (user_text, forced_flag) in enumerate(item["turns"]):
            messages.append({"role": "user", "content": user_text})
            is_last_turn = turn_idx == len(item["turns"]) - 1

            if item["category"] == "easy_baseline" and is_last_turn:
                raw, _, lat = _raw_generate(tuned_runner, messages, thinking=False)
                latencies["non_thinking"].append(lat)
                _, answer = _extract_trace_and_answer(raw, open_marker, close_marker)
                messages.append({"role": "assistant", "content": answer})
                continue

            if item["category"] == "ambiguous_hard" and is_last_turn:
                raw_think, _, lat_think = _raw_generate(tuned_runner, messages, thinking=True)
                raw_nothink, _, lat_nothink = _raw_generate(tuned_runner, messages, thinking=False)
                latencies["thinking"].append(lat_think)
                latencies["non_thinking"].append(lat_nothink)
                ok, reason = _format_fidelity_ok(raw_think, open_marker, close_marker)
                format_fidelity_results.append({"item": item["id"], "ok": ok, "reason": reason})
                trace, think_answer = _extract_trace_and_answer(raw_think, open_marker, close_marker)
                _, nothink_answer = _extract_trace_and_answer(raw_nothink, open_marker, close_marker)
                if trace:
                    trace_substance_scores.append(_judge_score(client, _EPITOME_TRACE_RUBRIC_SYSTEM, trace))
                hard_item_pairs.append({"item": item["id"], "prompt": user_text, "thinking": think_answer, "non_thinking": nothink_answer})
                messages.append({"role": "assistant", "content": think_answer})
                continue

            # explicit_flag / multi_turn_switch: forced_flag drives obedience checks
            expected_thinking = bool(forced_flag)
            raw, _, lat = _raw_generate(tuned_runner, messages, thinking=expected_thinking)
            latencies["thinking" if expected_thinking else "non_thinking"].append(lat)
            ok, reason = _format_fidelity_ok(raw, open_marker, close_marker)
            format_fidelity_results.append({"item": f"{item['id']}_turn{turn_idx}", "ok": ok, "reason": reason})
            trace, answer = _extract_trace_and_answer(raw, open_marker, close_marker)
            obeyed = _mode_obedience_ok(trace, expected_thinking)
            mode_obedience_results.append({
                "item": f"{item['id']}_turn{turn_idx}", "expected_thinking": expected_thinking, "obeyed": obeyed,
            })
            messages.append({"role": "assistant", "content": answer})

    # Non-thinking quality regression: base (pre-finetune) vs tuned, same easy items, non-thinking mode
    regression_scores = []
    for item in [it for it in probe_items if it["category"] in ("easy_baseline", "ambiguous_hard")]:
        user_text = item["turns"][0][0]
        base_messages = [{"role": "system", "content": FIXED_SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
        raw_base, _, _ = _raw_generate(base_runner, base_messages, thinking=False if model_cfg.supports_thinking else None)
        _, base_answer = _extract_trace_and_answer(raw_base, open_marker, close_marker)
        raw_tuned, _, _ = _raw_generate(tuned_runner, base_messages, thinking=False)
        _, tuned_answer = _extract_trace_and_answer(raw_tuned, open_marker, close_marker)
        before = _judge_score(client, _NON_THINKING_QUALITY_SYSTEM, base_answer)
        after = _judge_score(client, _NON_THINKING_QUALITY_SYSTEM, tuned_answer)
        regression_scores.append({"item": item["id"], "before": before["score"], "after": after["score"]})

    # Thinking-mode value-add: pairwise LLM judge, hard/ambiguous items only
    value_add_results = []
    for pair in hard_item_pairs:
        schema_prompt = (
            f"Ambiguous disclosure: {pair['prompt']!r}\n\n"
            f"Reply A (thinking mode): {pair['thinking']!r}\n\n"
            f"Reply B (non-thinking mode): {pair['non_thinking']!r}"
        )
        text = client.generate_json(_VALUE_ADD_SYSTEM, schema_prompt, max_tokens=300)
        start, end = text.find("{"), text.rfind("}")
        verdict = json.loads(text[start:end + 1])
        value_add_results.append({"item": pair["item"], **verdict})

    gate_format_ok = all(r["ok"] for r in format_fidelity_results) if format_fidelity_results else False
    gate_obedience_ok = all(r["obeyed"] for r in mode_obedience_results) if mode_obedience_results else False

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None

    thinking_wins = sum(1 for v in value_add_results if v.get("winner") == "A")
    result = {
        "model": model_cfg.name,
        "gates": {
            "format_fidelity_pass": gate_format_ok,
            "format_fidelity_detail": format_fidelity_results,
            "mode_obedience_pass": gate_obedience_ok,
            "mode_obedience_detail": mode_obedience_results,
        },
        "scored": {
            "non_thinking_quality_regression": regression_scores,
            "non_thinking_quality_regression_avg_delta": _avg(
                [r["after"] - r["before"] for r in regression_scores]
            ),
            "reasoning_trace_substance_avg": _avg([s["score"] for s in trace_substance_scores]),
            "reasoning_trace_substance_detail": trace_substance_scores,
            "thinking_value_add_wins_out_of": f"{thinking_wins}/{len(value_add_results)}",
            "thinking_value_add_detail": value_add_results,
            "efficiency": {
                "avg_latency_thinking_seconds": _avg(latencies["thinking"]),
                "avg_latency_non_thinking_seconds": _avg(latencies["non_thinking"]),
            },
        },
        "eliminated": not (gate_format_ok and gate_obedience_ok),
    }
    return result


# ---------------------------------------------------------------------------
# Step 4b — T1-T5 test-case suite, scored on the same 6-dimension judge
# rubric judge.py already uses for the main pipeline (reused verbatim via
# judge.score_transcript, NOT reimplemented) — run against the fine-tuned
# thinkprobe adapter, in both thinking modes where the model supports it.
# Kept fully separate from results_manager's shared manifest/scores.json so
# this never mixes into (or corrupts) report.py's aggregation of the main
# run's results.
# ---------------------------------------------------------------------------

# Same six rubric dimensions report.py aggregates for the main pipeline —
# imported, not copied, so the two never drift out of sync.
from report import DIMENSIONS as JUDGE_RUBRIC_DIMENSIONS


def _save_probe_transcript(result: dict) -> Path:
    meta = result["metadata"]
    fname = "_".join(str(meta.get(k, "na")) for k in
                      ("model_name", "test_case", "thinking_mode"))
    path = PROBE_TRANSCRIPT_DIR / f"{fname}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def _save_probe_scores(record: dict, runs: list[dict]) -> Path:
    meta = record["metadata"]
    fname = "_".join(str(meta.get(k, "na")) for k in
                      ("model_name", "test_case", "thinking_mode"))
    path = PROBE_SCORE_DIR / f"{fname}.scores.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "judge_runs": runs}, f, indent=2)
    return path


def _rubric_means(runs: list[dict]) -> dict:
    means = {}
    for dim in JUDGE_RUBRIC_DIMENSIONS:
        scores = [r[dim]["score"] for r in runs if isinstance(r.get(dim, {}).get("score"), int)]
        means[dim] = sum(scores) / len(scores) if scores else None
    return means


def run_test_case_suite(model_cfg: ModelUnderTest, dry_run: bool = False) -> dict:
    """Runs every T1-T5 test case (simulator.run_conversation, same module
    the main pipeline uses) against this model's thinkprobe LoRA adapter,
    once per thinking mode, scores each transcript with judge.py's existing
    6-dimension rubric (score_transcript), and returns per-test-case /
    per-thinking-mode rubric means plus any human-review flags raised."""
    import simulator
    import judge
    from inference import ModelRunner

    run_name = f"{model_cfg.name}-thinkprobe"
    adapter_path = str(PROBE_CKPT_DIR / run_name / "final")
    runner = ModelRunner(model_cfg, adapter_path=adapter_path)

    test_case_ids = simulator.list_test_cases()
    if dry_run:
        test_case_ids = test_case_ids[:1]
    thinking_modes = [True, False] if model_cfg.supports_thinking else [None]

    per_test_case = {}
    any_flagged = []
    for tc_id in test_case_ids:
        test_case = simulator.load_test_case(tc_id)
        if dry_run:
            test_case = {**test_case, "num_turns": 2, "turn_beats": test_case["turn_beats"][:2]}

        per_test_case[tc_id] = {}
        for thinking in thinking_modes:
            generate_fn = (lambda messages, _t=thinking: runner.generate(messages, thinking=_t))
            result = simulator.run_conversation(
                generate_fn, test_case, FIXED_SYSTEM_PROMPT,
                metadata={
                    "model_name": run_name, "phase": "thinkprobe", "condition": "matched",
                    "thinking_mode": thinking, "adapter": adapter_path,
                },
            )
            _save_probe_transcript(result)

            runs = judge.score_transcript(result, related_records=None)
            _save_probe_scores(result, runs)

            flagged = [r for r in runs if r.get("flag_for_human_review")]
            if flagged:
                any_flagged.append({"test_case": tc_id, "thinking_mode": thinking})

            per_test_case[tc_id][str(thinking)] = {
                "rubric_means": _rubric_means(runs),
                "num_judge_runs": len(runs),
                "flagged_for_human_review": bool(flagged),
            }

    return {"per_test_case": per_test_case, "flagged_for_human_review": any_flagged}


# ---------------------------------------------------------------------------
# Step 5 — decision rule + report
# ---------------------------------------------------------------------------

def apply_decision_rule(per_model_results: list[dict], train_summaries: dict) -> dict:
    survivors = [r for r in per_model_results if not r["eliminated"]]
    for r in survivors:
        eff = r["scored"]["efficiency"]
        lat_thinking = eff.get("avg_latency_thinking_seconds") or 0.0
        lat_nothink = eff.get("avg_latency_non_thinking_seconds") or 0.001
        latency_overhead = lat_thinking / lat_nothink if lat_nothink else float("inf")
        wins_str = r["scored"]["thinking_value_add_wins_out_of"]
        wins, total = (int(x) for x in wins_str.split("/")) if "/" in wins_str and wins_str.split("/")[1] != "0" else (0, 1)
        value_add_rate = wins / total if total else 0.0
        r["_ranking_score"] = value_add_rate / latency_overhead if latency_overhead else 0.0

    survivors.sort(key=lambda r: r["_ranking_score"], reverse=True)
    any_value_add = any(r["_ranking_score"] > 0.34 for r in survivors)  # thinking wins >1/3 of pairwise comparisons, adjusted for latency

    return {
        "eliminated_models": [r["model"] for r in per_model_results if r["eliminated"]],
        "surviving_models_ranked": [r["model"] for r in survivors],
        "any_model_shows_thinking_value_add": any_value_add,
        "recommendation": (
            "At least one surviving model shows real thinking-mode value-add on "
            "ambiguous cases relative to its latency cost — worth pursuing full "
            "fine-tuning with mode fusion for that model."
            if any_value_add else
            "No candidate showed meaningful thinking-mode value-add after this "
            "50-example probe. Per the decision rule, fine-tune for non-thinking "
            "behavior only — simpler, avoids the mode-fusion quality tax — and "
            "treat thinking mode as a discarded stretch feature, not a core requirement."
        ),
    }


def write_report(per_model_results: list[dict], train_summaries: dict, decision: dict) -> None:
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models_probed": [r["model"] for r in per_model_results],
        "train_summaries": train_summaries,
        "per_model_results": per_model_results,
        "decision": decision,
    }
    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = ["# Thinking-Mode Compatibility Probe — Report", ""]
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Eliminated (failed a gate): {', '.join(decision['eliminated_models']) or 'none'}")
    lines.append(f"- Surviving, ranked best-first: {', '.join(decision['surviving_models_ranked']) or 'none'}")
    lines.append(f"- Verdict: {decision['recommendation']}")
    lines.append("")
    lines.append("## Per-model detail")
    for r in per_model_results:
        lines.append(f"### {r['model']}")
        lines.append(f"- format_fidelity_pass: {r['gates']['format_fidelity_pass']}")
        lines.append(f"- mode_obedience_pass: {r['gates']['mode_obedience_pass']}")
        lines.append(f"- eliminated: {r['eliminated']}")
        if not r["eliminated"]:
            s = r["scored"]
            lines.append(f"- non_thinking_quality_regression_avg_delta: {s['non_thinking_quality_regression_avg_delta']}")
            lines.append(f"- reasoning_trace_substance_avg: {s['reasoning_trace_substance_avg']}")
            lines.append(f"- thinking_value_add_wins_out_of: {s['thinking_value_add_wins_out_of']}")
            lines.append(f"- efficiency: {s['efficiency']}")
        tc = r.get("test_case_rubric")
        if tc:
            lines.append(f"- T1-T5 human-review flags: {tc['flagged_for_human_review'] or 'none'}")
            lines.append("- T1-T5 rubric means (6-dimension judge rubric, per thinking mode):")
            for tc_id, by_mode in tc["per_test_case"].items():
                for mode_key, entry in by_mode.items():
                    means = ", ".join(f"{k}={v:.2f}" if v is not None else f"{k}=N/A"
                                       for k, v in entry["rubric_means"].items())
                    lines.append(f"  - {tc_id} (thinking={mode_key}): {means}")
        lines.append("")
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Step 6 — git add/commit/push (only this experiment's own new files)
# ---------------------------------------------------------------------------

def git_commit_and_push(dry_run: bool = False) -> None:
    if dry_run:
        print("[thinkprobe] --dry-run: skipping git add/commit/push.")
        return
    repo_root = ROOT_DIR.parent
    subprocess.run(["git", "add", "wellness_pipeline/thinking_probe"], cwd=repo_root, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "wellness_pipeline/thinking_probe"],
        cwd=repo_root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not status:
        print("[thinkprobe] nothing new to commit.")
        return
    subprocess.run(
        ["git", "commit", "-m", "Thinking-mode compatibility probe: results + report"],
        cwd=repo_root, check=True,
    )
    subprocess.run(["git", "push"], cwd=repo_root, check=True)
    print("[thinkprobe] committed and pushed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_all(models: Optional[list[str]] = None, dry_run: bool = False, push: bool = True) -> dict:
    candidates = thinking_models()
    if models:
        candidates = [m for m in candidates if m.name in models]
    if dry_run:
        candidates = candidates[:1]
    if not candidates:
        raise SystemExit("No thinking-capable models found in config.MODELS_UNDER_TEST to probe.")

    print(f"[thinkprobe] models under test: {[m.name for m in candidates]}")

    mixed = build_mixed_training_set(dry_run=dry_run)
    probe_items = build_probe_set(dry_run=dry_run)
    print(f"[thinkprobe] mixed train set: {len(mixed)} examples, probe set: {len(probe_items)} items")

    train_summaries = {}
    per_model_results = []
    for model_cfg in candidates:
        print(f"[thinkprobe] fine-tuning {model_cfg.name} ...")
        records = materialize_records(mixed, model_cfg)
        train_summaries[model_cfg.name] = finetune_thinking_probe(model_cfg, records, dry_run=dry_run)
        print(f"[thinkprobe] evaluating {model_cfg.name} ...")
        result = evaluate_model(model_cfg, probe_items, dry_run=dry_run)
        print(f"[thinkprobe] running T1-T5 test-case suite for {model_cfg.name} ...")
        result["test_case_rubric"] = run_test_case_suite(model_cfg, dry_run=dry_run)
        per_model_results.append(result)

    decision = apply_decision_rule(per_model_results, train_summaries)
    write_report(per_model_results, train_summaries, decision)
    print(f"[thinkprobe] report written to {REPORT_MD_PATH}")
    print(json.dumps(decision, indent=2))

    if push:
        git_commit_and_push(dry_run=dry_run)

    return {"train_summaries": train_summaries, "per_model_results": per_model_results, "decision": decision}


def main():
    parser = argparse.ArgumentParser(description="Thinking-mode compatibility probe — run-all does everything.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run-all", help="prep data -> fine-tune -> evaluate -> report -> git push")
    p_all.add_argument("--models", type=str, default=None, help="comma-separated subset of thinking-capable model names")
    p_all.add_argument("--dry-run", action="store_true", help="tiny fast smoke test, no git push")
    p_all.add_argument("--no-push", action="store_true", help="do everything except the final git commit/push")

    sub.add_parser("prep-data", help="build the mixed training set + probe set only")
    sub.add_parser("list-models", help="print which config.py models are thinking-capable")

    args = parser.parse_args()

    if args.cmd == "list-models":
        print(json.dumps([m.name for m in thinking_models()], indent=2))
        return

    if args.cmd == "prep-data":
        mixed = build_mixed_training_set()
        probe = build_probe_set()
        print(f"Wrote {len(mixed)} train examples to {TRAIN_SET_PATH}")
        print(f"Wrote {len(probe)} probe items to {PROBE_SET_PATH}")
        return

    if args.cmd == "run-all":
        models = args.models.split(",") if args.models else None
        run_all(models=models, dry_run=args.dry_run, push=not args.no_push)


if __name__ == "__main__":
    main()
