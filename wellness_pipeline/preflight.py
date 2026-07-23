"""
v3 preflight gates — called by orchestrator.py's `run-v3` command before any
model enters training, and runnable standalone for inspection.

Four checks, all written into ONE comprehensive, timestamped report per run
— reports/{version}/preflight_{timestamp}.md — built up incrementally as
each stage completes (see run_preflight):

1. Chat template validation (validate_chat_templates) — formats one real
   training example per model via the pipeline's own
   model_utils.build_training_text, prints it framed for manual review, and
   runs an automated (necessary-but-not-sufficient) special-token presence/
   duplication check against config.EXPECTED_SPECIAL_TOKENS. This check
   alone cannot pass a model into training — see --reviewed-templates below.

2. Target-modules / load / trainable-pct gate (validate_target_modules) —
   thin wrapper around lora_target_module_diagnostic.py's proven diagnostic
   (reused, not duplicated), annotating any model in
   config.KNOWN_ARCHITECTURAL_ASYMMETRIES as "expected-asymmetric" instead
   of BLOCKED.

3. Batch-size table assertion (assert_batch_table) — confirms every model
   hits the same effective batch size (config.TARGET_EFFECTIVE_BATCH_SIZE).

4. Sequence-length audit (audit_sequence_lengths) — tokenizes every phase2/3
   training example per model, reports the length distribution and
   truncation-% at both the historical 256 and the configured
   TrainingConfig.max_seq_length (2048), as a soft warning only (data that
   exceeds max_seq_length truncates, it doesn't crash — this is a heads-up,
   not a gate).

Hard gates that BLOCK a model from `run-v3` training: failing (2), or
failing (1)'s automated check. Passing (1)'s automated check is NOT
sufficient on its own — a human must additionally confirm they read the
printed template output, via `--reviewed-templates` (or an interactive y/n
prompt when attached to a TTY and that flag is omitted).

Run standalone:
    python preflight.py --dataset-size 50
    python preflight.py --dataset-size 50 --reviewed-templates
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import config
import model_utils
import lora_target_module_diagnostic as diag
import results_manager as rm


class PreflightError(RuntimeError):
    pass


@dataclass
class ModelPreflightResult:
    name: str
    chat_template_text: str = ""
    chat_template_ok: bool = False
    chat_template_issues: list = field(default_factory=list)
    target_modules_ok: bool = False
    target_modules_note: str = ""
    seq_length_stats: dict = field(default_factory=dict)
    batch_settings: tuple = ()
    blocked: bool = False
    block_reasons: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Chat template validation
# ---------------------------------------------------------------------------

def _sample_messages(model_cfg, data_cfg) -> list[dict]:
    import json

    path = data_cfg.train_path(2)
    if not path.exists():
        raise PreflightError(f"No training data at {path} — run data-prep for dataset_size={data_cfg.num_samples} first.")
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()
    return json.loads(first_line)["messages"]


def validate_chat_template(model_cfg, data_cfg) -> tuple[str, bool, list[str]]:
    """Returns (formatted_text, automated_check_passed, issues)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = _sample_messages(model_cfg, data_cfg)
    prompt_text, full_text = model_utils.build_training_text(tokenizer, messages)

    expected_tokens = config.EXPECTED_SPECIAL_TOKENS.get(model_cfg.name, [])
    issues = []
    if not full_text.strip():
        issues.append("formatted output is empty")
    for tok in expected_tokens:
        count = full_text.count(tok)
        if count == 0:
            issues.append(f"expected special token {tok!r} not found")
        elif count > 2:  # a couple of legitimate repeats per multi-turn example is normal; many is not
            issues.append(f"special token {tok!r} appears suspiciously often ({count}x) — possible duplication")

    return full_text, (len(issues) == 0), issues


def validate_chat_templates(model_cfgs, data_cfg) -> dict[str, ModelPreflightResult]:
    results = {}
    for model_cfg in model_cfgs:
        result = ModelPreflightResult(name=model_cfg.name)
        try:
            text, ok, issues = validate_chat_template(model_cfg, data_cfg)
        except Exception as exc:
            text, ok, issues = "", False, [f"load/format error: {exc}"]
        result.chat_template_text = text
        result.chat_template_ok = ok
        result.chat_template_issues = issues
        results[model_cfg.name] = result

        print(f"\n{'=' * 70}\n=== {model_cfg.name} — chat template output (manual review required) ===\n{'=' * 70}")
        print(text if text else "<no output — see issues below>")
        print(f"{'-' * 70}")
        if issues:
            print(f"[automated check] FAILED: {issues}")
        else:
            print("[automated check] passed (expected special tokens present, no duplication)")
    return results


def render_chat_template_section(results: dict[str, ModelPreflightResult]) -> str:
    lines = ["## 1. Chat template validation\n"]
    for name, r in results.items():
        lines.append(f"### {name}\n")
        if r.chat_template_issues:
            lines.append(f"**[automated check] FAILED**: {r.chat_template_issues}\n")
        else:
            lines.append("**[automated check] passed** (expected special tokens present, no duplication)\n")
        lines.append(f"```\n{r.chat_template_text or '<no output — see issues above>'}\n```\n")
    return "\n".join(lines)


def confirm_templates_reviewed(reviewed_flag: bool) -> bool:
    if reviewed_flag:
        return True
    if not sys.stdin.isatty():
        return False
    answer = input(
        "\nHave you read every chat-template output printed above and confirmed it looks "
        "correct for that model's architecture? [y/N]: "
    ).strip().lower()
    return answer == "y"


# ---------------------------------------------------------------------------
# 2. Target-modules / rank / trainable-pct gate — reuses lora_target_module_diagnostic.py
# ---------------------------------------------------------------------------

def validate_target_modules(model_cfgs) -> dict[str, diag.DiagnosticResult]:
    results = {}
    for model_cfg in model_cfgs:
        print(f"\n=== {model_cfg.name} — target_modules / rank gate ===")
        try:
            result = diag.run_one(model_cfg)
        except Exception as exc:
            print(f"[error] {model_cfg.name}: {exc}", file=sys.stderr)
            result = diag.DiagnosticResult(
                name=model_cfg.name, rank_tier="?", r=0, alpha=0,
                trainable_params=0, total_params=0, trainable_pct=0.0,
                expected_range=(0, 0), status="BLOCKED", reason=f"load/apply error: {exc}",
            )
        asymmetry_note = config.KNOWN_ARCHITECTURAL_ASYMMETRIES.get(model_cfg.name)
        if asymmetry_note and result.status == "BLOCKED" and "load/apply error" not in result.reason:
            result.status = "READY"
            result.reason = f"expected-asymmetric: {asymmetry_note}"
            print(f"[expected-asymmetric] {model_cfg.name}: {asymmetry_note}")
        results[model_cfg.name] = result
    return results


def render_target_modules_section(results: dict[str, diag.DiagnosticResult]) -> str:
    lines = ["## 2. Target-modules / rank / trainable-pct gate\n"]
    lines.append("| model | rank_tier | r | alpha | trainable_params | total_params | trainable_pct | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, r in results.items():
        lines.append(
            f"| {name} | {r.rank_tier} | {r.r} | {r.alpha} | {r.trainable_params:,} | "
            f"{r.total_params:,} | {r.trainable_pct:.4f}% | {r.status} |"
        )
    lines.append("")
    for name, r in results.items():
        if r.reason:
            lines.append(f"- **{name}**: {r.reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Sequence-length audit — soft warning, not a gate
# ---------------------------------------------------------------------------

def audit_sequence_lengths(model_cfg, data_cfg, phases=(2, 3)) -> dict:
    import json

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_model_id, trust_remote_code=model_cfg.trust_remote_code)
    lengths = []
    for phase in phases:
        path = data_cfg.train_path(phase)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)["messages"]
                _, full_text = model_utils.build_training_text(tokenizer, messages)
                lengths.append(len(tokenizer(full_text)["input_ids"]))

    if not lengths:
        return {"count": 0}

    lengths.sort()
    n = len(lengths)

    def pct(p):
        return lengths[min(n - 1, int(n * p))]

    def truncation_rate(limit):
        return sum(1 for l in lengths if l > limit) / n * 100

    stats = {
        "count": n,
        "min": lengths[0],
        "max": lengths[-1],
        "mean": sum(lengths) / n,
        "median": pct(0.5),
        "p99": pct(0.99),
        "truncated_pct_at_256": truncation_rate(256),
        "truncated_pct_at_configured": truncation_rate(config.SHARED_TRAINING_CONFIG.max_seq_length),
        "configured_max_seq_length": config.SHARED_TRAINING_CONFIG.max_seq_length,
    }
    print(
        f"  {model_cfg.name}: n={stats['count']} min={stats['min']} max={stats['max']} "
        f"mean={stats['mean']:.0f} median={stats['median']} p99={stats['p99']} | "
        f"truncated@256={stats['truncated_pct_at_256']:.1f}% "
        f"truncated@{stats['configured_max_seq_length']}={stats['truncated_pct_at_configured']:.1f}%"
    )
    if stats["p99"] > stats["configured_max_seq_length"]:
        print(
            f"  [warning] {model_cfg.name}: p99 length ({stats['p99']}) exceeds configured "
            f"max_seq_length ({stats['configured_max_seq_length']}) — some long examples will truncate."
        )
    return stats


def render_seq_length_section(seq_stats_by_name: dict[str, dict]) -> str:
    lines = ["## 4. Sequence-length audit\n"]
    lines.append(f"Configured `max_seq_length` = {config.SHARED_TRAINING_CONFIG.max_seq_length}\n")
    lines.append("| model | n | min | max | mean | median | p99 | truncated@256 | truncated@configured |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, s in seq_stats_by_name.items():
        if not s.get("count"):
            lines.append(f"| {name} | 0 | - | - | - | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {s['count']} | {s['min']} | {s['max']} | {s['mean']:.0f} | {s['median']} | "
            f"{s['p99']} | {s['truncated_pct_at_256']:.1f}% | {s['truncated_pct_at_configured']:.1f}% |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch-size table assertion
# ---------------------------------------------------------------------------

def assert_batch_table(model_cfgs) -> list[tuple]:
    rows = []
    print(f"\n{'model':<28} {'per_device':>10} {'grad_accum':>11} {'effective':>10}")
    for model_cfg in model_cfgs:
        per_device, grad_accum, effective = config.batch_settings_for(model_cfg.name)
        rows.append((model_cfg.name, per_device, grad_accum, effective))
        print(f"{model_cfg.name:<28} {per_device:>10} {grad_accum:>11} {effective:>10}")
    effectives = {r[3] for r in rows}
    assert effectives == {config.TARGET_EFFECTIVE_BATCH_SIZE}, (
        f"Effective batch sizes are not identical across models: {rows}"
    )
    print(f"[ok] all {len(rows)} models hit effective batch size {config.TARGET_EFFECTIVE_BATCH_SIZE}")
    return rows


def render_batch_table_section(rows: list[tuple]) -> str:
    lines = ["## 3. Batch-size table\n"]
    lines.append(f"Target effective batch size: {config.TARGET_EFFECTIVE_BATCH_SIZE}\n")
    lines.append("| model | per_device_batch_size | gradient_accumulation_steps | effective_batch_size |")
    lines.append("|---|---|---|---|")
    for name, per_device, grad_accum, effective in rows:
        lines.append(f"| {name} | {per_device} | {grad_accum} | {effective} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def run_preflight(model_cfgs, data_cfg, version: str, reviewed_templates: bool = False) -> dict[str, ModelPreflightResult]:
    """Returns {model_name: ModelPreflightResult}. Models with `.blocked=True`
    must NOT proceed to training. Writes one comprehensive, timestamped
    report per run — reports/{version}/preflight_{timestamp}.md — covering
    every stage (chat templates, target_modules/rank gate, batch table,
    seq-length audit, final BLOCKED/READY status). The file is written
    incrementally as each stage completes (not just once at the end), so it
    exists to read/tail well before the manual-review confirmation prompt
    needs an answer, and a DIFFERENT timestamped file is created per run —
    nothing is ever overwritten or appended into a previous run's file, so
    you can always tell which run produced which output."""
    import time

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_path = config.REPORT_DIR / version / f"preflight_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections = [f"# Preflight report — {version} (run {timestamp})\n"]

    def _flush():
        report_path.write_text("\n\n".join(sections), encoding="utf-8")

    template_results = validate_chat_templates(model_cfgs, data_cfg)
    sections.append(render_chat_template_section(template_results))
    _flush()
    print(f"\n[report] preflight output being written to {report_path} as each stage completes "
          f"— read it there if terminal scrollback is inconvenient.")

    target_results = validate_target_modules(model_cfgs)
    sections.append(render_target_modules_section(target_results))
    _flush()

    batch_rows = assert_batch_table(model_cfgs)
    batch_by_name = {r[0]: r for r in batch_rows}
    sections.append(render_batch_table_section(batch_rows))
    _flush()

    print("\n=== sequence-length audit ===")
    seq_stats_by_name = {}
    for model_cfg in model_cfgs:
        seq_stats_by_name[model_cfg.name] = audit_sequence_lengths(model_cfg, data_cfg)
    sections.append(render_seq_length_section(seq_stats_by_name))
    _flush()

    templates_reviewed = confirm_templates_reviewed(reviewed_templates)
    if not templates_reviewed:
        print(
            "\n[gate] Chat templates have not been confirmed as manually reviewed "
            "(pass --reviewed-templates after reading the printed output above). "
            "ALL models are blocked from training until this is confirmed."
        )

    final = {}
    for model_cfg in model_cfgs:
        name = model_cfg.name
        result = template_results[name]
        target_result = target_results[name]
        result.target_modules_ok = target_result.status == "READY"
        result.target_modules_note = target_result.reason
        result.seq_length_stats = seq_stats_by_name[name]
        result.batch_settings = batch_by_name[name][1:]

        reasons = []
        if not result.chat_template_ok:
            reasons.append(f"chat template automated check failed: {result.chat_template_issues}")
        if not templates_reviewed:
            reasons.append("chat templates not manually reviewed/confirmed")
        if not result.target_modules_ok:
            reasons.append(f"target_modules/rank gate failed: {target_result.reason}")

        result.blocked = len(reasons) > 0
        result.block_reasons = reasons
        final[name] = result

    sections.append(_render_final_status_section(final))
    _flush()
    print(f"\n[report] final preflight report written to {report_path}")
    return final


def _render_final_status_section(results: dict[str, ModelPreflightResult]) -> str:
    lines = ["## Final status\n"]
    for name, r in results.items():
        status = "BLOCKED" if r.blocked else "READY"
        lines.append(f"### {name} — {status}\n")
        if r.blocked:
            for reason in r.block_reasons:
                lines.append(f"- {reason}")
    return "\n".join(lines)
    print(f"\n[report] wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-size", type=int, default=50)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--reviewed-templates", action="store_true")
    parser.add_argument("--version", type=str, default=config.PIPELINE_VERSION)
    args = parser.parse_args()

    model_cfgs = config.MODELS_UNDER_TEST
    if args.models:
        wanted = set(n.strip() for n in args.models.split(","))
        model_cfgs = [m for m in model_cfgs if m.name in wanted]

    data_cfg = config.DataConfig(num_samples=args.dataset_size)
    results = run_preflight(model_cfgs, data_cfg, args.version, args.reviewed_templates)

    print("\n=== preflight summary ===")
    blocked = [n for n, r in results.items() if r.blocked]
    ready = [n for n, r in results.items() if not r.blocked]
    print(f"READY ({len(ready)}): {', '.join(ready) or '(none)'}")
    print(f"BLOCKED ({len(blocked)}): {', '.join(blocked) or '(none)'}")
    for n in blocked:
        print(f"  - {n}: {results[n].block_reasons}")

    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
