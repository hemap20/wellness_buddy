"""
Standalone diagnostic: validates that lora_target_modules in config.py is
correctly and comparably applied across all MODELS_UNDER_TEST before any
real training run.

For each model it:
  1. Loads the base model via the pipeline's own load_tokenizer_and_model()
     (so this checks the exact same load path train.py uses).
  2. Auto-scales (r, alpha) by parameter-count tier (config.py currently
     shares one r/alpha across all models — this script applies its own
     tiering on top, purely for this diagnostic; it does not change
     config.py or train.py).
  3. Applies LoRA with the model's configured target_modules and calls
     print_trainable_parameters().
  4. Flags any model whose trainable_pct falls outside ~2x of its tier's
     expected range as "BLOCKED", everything else as "READY".

Run:
    python lora_target_module_diagnostic.py
    python lora_target_module_diagnostic.py --models dialogpt-small,qwen3-8b
    python lora_target_module_diagnostic.py --skip-gated
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CONFIG SECTION — edit here. Everything below pulls straight from your real
# pipeline config (wellness_pipeline/config.py) so this diagnostic exercises
# the exact same model list, hf_model_id, target_modules, dtype, and
# gated/trust_remote_code flags you'll actually train with.
# ---------------------------------------------------------------------------

from config import MODELS_UNDER_TEST, rank_alpha_for  # noqa: E402
from model_utils import load_tokenizer_and_model, resolve_device  # noqa: E402

# Rank tier -> r/alpha comes from config.rank_alpha_for() (the real, confirmed
# v3 tiering) rather than a copy kept here — this diagnostic and preflight.py's
# gate must never be able to drift apart on what "correct" looks like.
# expected_pct_range is diagnostic-only: the sane trainable_pct band for that
# tier when target_modules is correctly scoped to attention projections only.
EXPECTED_PCT_RANGES = {
    "small": (0.3, 1.0),
    "mid": (0.3, 1.5),
    "large": (0.15, 0.8),
}

# Outlier threshold: flag if actual_pct is more than this multiple above the
# tier's expected high, or this fraction of the expected low.
OUTLIER_HIGH_MULTIPLE = 2.0
OUTLIER_LOW_FRACTION = 0.5  # i.e. "more than 2x below" == less than half of the low bound

TASK_TYPE = "CAUSAL_LM"
LORA_DROPOUT = 0.05  # matches SHARED_TRAINING_CONFIG.lora_dropout; not part of what's being diagnosed


def rank_tier_for(total_params: int):
    tier_name, r, alpha = rank_alpha_for(total_params)
    expected_range = EXPECTED_PCT_RANGES[tier_name]
    return tier_name, r, alpha, expected_range


def find_all_linear_names(model) -> list[str]:
    """Enumerate actual torch.nn.Linear leaf-module names (last path segment,
    deduped) present in the model. Used only for the diagnostic's own
    reporting — e.g. to tell you what a silent-failure model's real linear
    layer names are, so you know what to put in lora_target_modules."""
    import torch

    names = set()
    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names.add(full_name.rsplit(".", 1)[-1])
    return sorted(names)


@dataclass
class DiagnosticResult:
    name: str
    rank_tier: str
    r: int
    alpha: int
    trainable_params: int
    total_params: int
    trainable_pct: float
    expected_range: tuple
    status: str  # "READY" or "BLOCKED"
    reason: str = ""


def run_one(model_cfg) -> DiagnosticResult:
    import torch
    from peft import LoraConfig, get_peft_model

    tokenizer, base_model = load_tokenizer_and_model(model_cfg)
    device = resolve_device(model_cfg.device)
    base_model = base_model.to(device)

    total_params_pre = sum(p.numel() for p in base_model.parameters())
    rank_tier, r, alpha, expected_range = rank_tier_for(total_params_pre)

    target_modules = (
        model_cfg.lora_target_modules if isinstance(model_cfg.lora_target_modules, str)
        else list(model_cfg.lora_target_modules)
    )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=target_modules,
        task_type=TASK_TYPE,
    )
    peft_model = get_peft_model(base_model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    trainable_pct = 100.0 * trainable_params / total_params

    peft_model.print_trainable_parameters()

    low, high = expected_range
    status, reason = "READY", ""
    if trainable_pct > high * OUTLIER_HIGH_MULTIPLE:
        status = "BLOCKED"
        reason = (
            f"trainable_pct ({trainable_pct:.3f}%) is >{OUTLIER_HIGH_MULTIPLE}x above the "
            f"expected high ({high}%) for tier '{rank_tier}' — target_modules likely matched "
            f"too many layers, matched non-Linear modules, or task_type is misconfigured."
        )
    elif trainable_pct < low * OUTLIER_LOW_FRACTION:
        status = "BLOCKED"
        reason = (
            f"trainable_pct ({trainable_pct:.3f}%) is far below the expected low ({low}%) for "
            f"tier '{rank_tier}' — target_modules likely didn't match any layers for this "
            f"architecture (silent failure: the adapter isn't actually adapting anything)."
        )
        if trainable_pct == 0.0 or trainable_params == 0:
            linear_names = find_all_linear_names(base_model)
            reason += f" Actual nn.Linear module names found in this model: {linear_names}"

    del peft_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DiagnosticResult(
        name=model_cfg.name,
        rank_tier=rank_tier,
        r=r,
        alpha=alpha,
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=trainable_pct,
        expected_range=expected_range,
        status=status,
        reason=reason,
    )


def print_summary_table(results: list[DiagnosticResult]) -> None:
    headers = ["model", "rank_tier", "r", "trainable_params", "total_params", "trainable_pct", "status"]
    rows = [
        [
            r.name,
            r.rank_tier,
            str(r.r),
            f"{r.trainable_params:,}",
            f"{r.total_params:,}",
            f"{r.trainable_pct:.4f}%",
            r.status,
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]

    def fmt_row(cells):
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))

    print()
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to check (default: all in MODELS_UNDER_TEST)")
    parser.add_argument("--skip-gated", action="store_true", help="Skip models flagged gated=True in config.py (avoids needing HF auth)")
    args = parser.parse_args()

    model_cfgs = MODELS_UNDER_TEST
    if args.models:
        wanted = set(n.strip() for n in args.models.split(","))
        model_cfgs = [m for m in model_cfgs if m.name in wanted]
        missing = wanted - {m.name for m in model_cfgs}
        if missing:
            print(f"[warn] unknown model name(s) skipped: {sorted(missing)}", file=sys.stderr)
    if args.skip_gated:
        model_cfgs = [m for m in model_cfgs if not m.gated]

    if not model_cfgs:
        print("No models selected — check --models/--skip-gated filters.", file=sys.stderr)
        sys.exit(1)

    results: list[DiagnosticResult] = []
    for model_cfg in model_cfgs:
        print(f"\n=== {model_cfg.name} ({model_cfg.hf_model_id}) ===")
        try:
            result = run_one(model_cfg)
        except Exception as exc:
            print(f"[error] {model_cfg.name}: failed to load/apply LoRA — {exc}", file=sys.stderr)
            results.append(
                DiagnosticResult(
                    name=model_cfg.name, rank_tier="?", r=0, alpha=0,
                    trainable_params=0, total_params=0, trainable_pct=0.0,
                    expected_range=(0, 0), status="BLOCKED",
                    reason=f"load/apply error: {exc}",
                )
            )
            continue
        results.append(result)

    print_summary_table(results)

    blocked = [r for r in results if r.status == "BLOCKED"]
    ready = [r for r in results if r.status == "READY"]

    if blocked:
        print(f"BLOCKED ({len(blocked)}) — target_modules needs manual fixing before training:")
        for r in blocked:
            print(f"  - {r.name}: {r.reason}")
        print()
    if ready:
        print(f"READY ({len(ready)}) — within expected trainable_pct range: {', '.join(r.name for r in ready)}")

    sys.exit(1 if blocked else 0)


if __name__ == "__main__":
    main()
