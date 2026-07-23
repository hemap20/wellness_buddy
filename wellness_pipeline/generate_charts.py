"""
Standalone chart generator — reads experiment results and produces
high-resolution PNGs for pasting into documentation.

Data sources (all read via results_manager.py's path-building helpers, never
hardcoded): results/manifest.jsonl, results/{model}/n{size}/phase{N}/scores.json,
results/{model}/n{size}/phase{N}/loss_curve.json, results/{model}/phase1_baseline.json.

Phase semantics (important for how charts compare things): phase 1 is the
untrained baseline. Phase 2 and phase 3 are two INDEPENDENT fine-tuning
branches off that same baseline (phase 2 = trained without the system prompt;
phase 3 = trained with it baked in) — not sequential steps. Every chart here
treats phase 2 and phase 3 as separate comparisons against the shared phase-1
baseline, never as one continuous progression.

Run standalone:
    python generate_charts.py --out-dir charts
    python generate_charts.py --phases 2,3 --robustness-metric aggregate
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import results_manager as rm
from config import CHECKPOINT_DIR, INFERENCE_CONDITIONS, PIPELINE_VERSION, REPORT_DIR, rank_alpha_for
from report import DIMENSIONS
from metrics import sample_efficiency

DPI = 200  # ~2x a typical 100dpi screen render, for crisp shrink-to-fit in docs

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.autolayout": True,
})

_PALETTE = plt.get_cmap("tab20").colors
_model_colors: dict[str, tuple] = {}


def color_for(model: str) -> tuple:
    if model not in _model_colors:
        _model_colors[model] = _PALETTE[len(_model_colors) % len(_PALETTE)]
    return _model_colors[model]


# Real total-parameter counts, measured live via lora_target_module_diagnostic.py
# during this session (not guessed from the name) — used as a fallback when a
# model has no recorded rank_tier yet (below). A name-substring heuristic here
# previously misclassified gemma-4-e2b-it (5.1B params) as "small" just
# because its name contains "gemma-...-e2b" alongside gemma-3-1b-it's — fixed
# by using measured params + config.rank_alpha_for's real tier boundaries
# instead of string-matching the model name.
_MEASURED_TOTAL_PARAMS = {
    "dialogpt-small": 124_734_720,
    "gemma-3-1b-it": 1_002_867_840,
    "gemma-4-e2b-it": 5_115_012_640,
    "starling-lm-7b-alpha": 7_269_011_456,
    "empathetic-qwen3-8b-jan": 8_221_406_208,
    "qwen3-8b": 8_221_406_208,
    "mistral-7b-instruct-v0.3": 7_275_286_528,
    "llama-3-8b": 8_057_524_224,
}


# Chart-grouping tier deliberately DIFFERS from the LoRA rank tier for this
# one model: gemma-4-e2b-it's raw measured params (5.1B) put it in the
# "large" rank tier for training (r=32/alpha=64, same as the 7-8B models —
# see config.RANK_TIERS), but "E2B" is Google's own "effective 2B params"
# designation for this Gemma 3n-family model (per-layer embeddings/shared
# params inflate the raw count well above its actual active-compute
# footprint) — for chart *grouping* (which is about presenting results the
# way you think about model size, not about training compute cost) it
# belongs with the small models. This mismatch vs. the rank-tier table is
# intentional, not a bug — don't "fix" it by changing RANK_TIERS/rank_alpha_for.
_EFFECTIVE_TIER_OVERRIDES = {
    "gemma-4-e2b-it": "small",
}


def size_tier(model: str, version: str = None) -> str:
    """Chart-grouping tier for this model. Checks, in order: (1) an explicit
    _EFFECTIVE_TIER_OVERRIDES entry, for models whose marketed/effective size
    class differs from their raw param count; (2) the rank_tier already
    recorded in a real train_summary.json for this model (train.py records
    it per run — see config.RANK_TIERS/rank_alpha_for), since that's the
    tier actually used for that run's LoRA config; (3) _MEASURED_TOTAL_PARAMS
    (still real measured numbers, just not from this specific run) for
    models without a completed training run yet. Never falls back to
    name-substring guessing."""
    if model in _EFFECTIVE_TIER_OVERRIDES:
        return _EFFECTIVE_TIER_OVERRIDES[model]

    version = version or PIPELINE_VERSION
    version_root = CHECKPOINT_DIR / version
    if version_root.is_dir():
        for run_dir in version_root.glob(f"{model}_{version}_*"):
            summary_path = run_dir / "train_summary.json"
            if summary_path.exists():
                try:
                    rank_tier = json.loads(summary_path.read_text()).get("rank_tier")
                except (json.JSONDecodeError, OSError):
                    rank_tier = None
                if rank_tier:
                    return rank_tier
    if model in _MEASURED_TOTAL_PARAMS:
        tier_name, _, _ = rank_alpha_for(_MEASURED_TOTAL_PARAMS[model])
        return tier_name
    return "unknown"


def discover_model_sizes(version: str = None) -> list[tuple[str, int]]:
    """(model, dataset_size) pairs with real phase 2/3 results for this
    pipeline generation, discovered by scanning results/{version}/ directly
    rather than trusting manifest.jsonl's "model" field — folders get
    renamed without rewriting the manifest, so a stale manifest name would
    point at an empty/moved folder. Each generation (v1/v2/v3) now lives
    under its own results/{version}/ directory (see results_manager.py) so
    there's no need to name-guess which models belong to which generation —
    just scan the version's own subtree. Model dirnames under a version are
    suffixed "_{version}" (e.g. "dialogpt-small_v2") — that suffix is
    stripped back off before returning, since every other rm.* lookup
    (load_scores/load_baseline/get_results_path) takes the base model name
    and re-appends "_{version}" itself; returning the raw dirname here would
    double-suffix those calls."""
    version = version or PIPELINE_VERSION
    version_root = rm.get_version_root(version)
    suffix = f"_{version}"
    pairs = set()
    if not version_root.is_dir():
        return []
    for model_dir in sorted(version_root.iterdir()):
        if not model_dir.is_dir():
            continue
        base_name = model_dir.name[: -len(suffix)] if model_dir.name.endswith(suffix) else model_dir.name
        for size_dir in model_dir.glob("n[0-9]*"):
            # Require an actual scores.json (not just an empty phase{N}/
            # directory scaffold — leftover renamed-away folders keep their
            # empty subdirs behind and would otherwise show up as spurious
            # "models" with no real data).
            if size_dir.is_dir() and any(size_dir.glob("phase*/scores.json")):
                pairs.add((base_name, int(size_dir.name.lstrip("n"))))
    return sorted(pairs)


def load_baseline(model: str, version: str = None) -> list[dict]:
    b = rm.load_baseline(model, version=version)
    return b["results"] if b else []


def load_scores(model: str, size: int, phase: int, version: str = None) -> list[dict]:
    try:
        return rm.load_scores(model, size, phase, version=version)
    except Exception:
        return []


def dim_avg(entries: list[dict], dim: str, condition: str = None) -> float:
    vals = []
    for e in entries:
        meta = e["metadata"]
        if condition is not None and meta.get("condition") != condition:
            continue
        for run in e["judge_runs"]:
            score = run.get(dim, {}).get("score")
            if isinstance(score, (int, float)):
                vals.append(score)
    return sum(vals) / len(vals) if vals else None


def aggregate_score(entries: list[dict], condition: str = None) -> float:
    """Mean across all 6 rubric dimensions (each dimension's own mean first),
    skipping dimensions with no numeric scores (e.g. system_prompt_dependency
    on phase 1, which has no comparative transcripts)."""
    vals = [dim_avg(entries, d, condition) for d in DIMENSIONS]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Train vs validation loss, per (model, phase) actually trained
# ---------------------------------------------------------------------------

def chart_loss_curves(model: str, size: int, phase: int, out_dir: Path, log: list, version: str = None):
    try:
        curve_path = rm.get_results_path(model, size, phase, version=version) / "loss_curve.json"
        if not curve_path.exists():
            log.append(f"SKIP loss_curve {model} n{size} phase{phase}: no loss_curve.json")
            return
        curve = json.loads(curve_path.read_text())
    except Exception as exc:
        log.append(f"SKIP loss_curve {model} n{size} phase{phase}: {exc}")
        return

    steps = curve.get("steps") or []
    train_loss = curve.get("train_loss") or []
    val_loss = curve.get("val_loss")
    if not steps or not train_loss:
        log.append(f"SKIP loss_curve {model} n{size} phase{phase}: empty curve")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(steps[:len(train_loss)], train_loss, marker="o", label="train loss", color="tab:blue")
    if val_loss and any(v is not None for v in val_loss):
        ax.plot(steps[:len(val_loss)], val_loss, marker="o", label="val loss", color="tab:orange")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_title(f"Train vs. validation loss — {model}, phase {phase}, n{size:03d}")
    ax.legend()
    savefig(fig, out_dir / f"n{size:03d}" / f"loss_curve_{model}_phase{phase}.png")
    log.append(f"OK   loss_curve {model} n{size} phase{phase}")


# ---------------------------------------------------------------------------
# 1b. Checkpoint-eval trajectory (v3+) — all 6 rubric dimensions across the
# checkpoint fractions checkpoint_eval.py evaluated, with the selected
# checkpoint marked. Reads reports/{version}/{model}_n{size}_phase{N}_checkpoint_trajectory.json
# (written by checkpoint_eval.py::run_checkpoint_eval_for_run), NOT
# loss_curve.json — this is judge-score trajectory, not training loss.
# ---------------------------------------------------------------------------

_DIM_MARKERS = ["o", "s", "^", "D", "v", "P"]
_DIM_LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def chart_checkpoint_trajectory(model: str, phase: int, size: int, out_dir: Path, log: list, version: str = None):
    version = version or PIPELINE_VERSION
    trajectory_path = REPORT_DIR / version / f"{model}_{rm.format_dataset_size(size)}_phase{phase}_checkpoint_trajectory.json"
    try:
        if not trajectory_path.exists():
            log.append(f"SKIP checkpoint_trajectory {model} phase{phase}: no {trajectory_path.name}")
            return
        report_data = json.loads(trajectory_path.read_text())
    except Exception as exc:
        log.append(f"SKIP checkpoint_trajectory {model} phase{phase}: {exc}")
        return

    if report_data.get("phase") != phase or report_data.get("dataset_size") != size:
        log.append(f"SKIP checkpoint_trajectory {model} phase{phase} n{size}: "
                    f"report is for phase{report_data.get('phase')} n{report_data.get('dataset_size')}")
        return

    trajectory = report_data.get("trajectory") or []
    if not trajectory:
        log.append(f"SKIP checkpoint_trajectory {model} phase{phase}: empty trajectory")
        return

    checkpoint_names = [entry["checkpoint"] for entry in trajectory]
    selected_name = report_data.get("selected_checkpoint_name")

    fig, ax = plt.subplots(figsize=(8, 5))
    per_checkpoint_avgs = []
    for i, dim in enumerate(DIMENSIONS):
        vals = [entry["scores"].get(dim) for entry in trajectory]
        if all(v is None for v in vals):
            continue  # e.g. system_prompt_dependency, always N/A for checkpoint-eval's single-condition passes
        ax.plot(checkpoint_names, vals, marker=_DIM_MARKERS[i % len(_DIM_MARKERS)],
                 linestyle=_DIM_LINESTYLES[i % len(_DIM_LINESTYLES)],
                 markersize=8, linewidth=1.5, alpha=0.75, label=dim.replace("_", " "))

    # Bold "average (all dims)" line — same convention as chart_baseline_vs_phases,
    # skipping dims with no numeric score at each checkpoint (e.g. system_prompt_dependency).
    for entry in trajectory:
        scores = [v for v in entry["scores"].values() if isinstance(v, (int, float))]
        per_checkpoint_avgs.append(sum(scores) / len(scores) if scores else None)
    ax.plot(checkpoint_names, per_checkpoint_avgs, marker="*", markersize=14, linewidth=3,
             color="black", label="average (all dims)")

    if selected_name in checkpoint_names:
        ax.axvline(checkpoint_names.index(selected_name), color="black", linewidth=1.5, linestyle=":",
                    label=f"selected: {selected_name}")

    ax.set_ylim(0.5, 5.5)
    ax.set_ylabel("score (1-5)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title(f"Checkpoint-eval trajectory — {model}, phase {phase}, n{size:03d} ({version})")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    savefig(fig, out_dir / f"n{size:03d}" / f"checkpoint_trajectory_{model}_phase{phase}.png")
    log.append(f"OK   checkpoint_trajectory {model} phase{phase} n{size} (selected: {selected_name})")


# ---------------------------------------------------------------------------
# 2. Baseline vs. phase2 vs. phase3, all in one graph per model
# ---------------------------------------------------------------------------


def chart_baseline_vs_phases(models: list[str], size: int, out_dir: Path, log: list, version: str = None):
    xs = ["phase1\nbaseline", "phase2\nno_prompt", "phase3\nmatched"]
    for model in models:
        baseline = load_baseline(model, version=version)
        phase2 = load_scores(model, size, 2, version=version)
        phase3 = load_scores(model, size, 3, version=version)
        if not baseline or (not phase2 and not phase3):
            log.append(f"SKIP baseline_vs_phases {model} n{size}: missing baseline/phase2/phase3 data")
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        # Small dims that land on identical values (common for a weak model
        # scoring the floor on several dimensions, e.g. flat 1.0 everywhere)
        # would otherwise draw perfectly on top of each other and look like
        # one line — a distinct marker shape + linestyle per dimension keeps
        # them visually separable even when the y-values coincide exactly.
        for i, dim in enumerate(DIMENSIONS):
            # phase1 baseline point uses no_prompt (matches phase2's training
            # exactly; phase3's "matched" condition is its own separate
            # comparison point, not phase1's system_prompt condition, so all
            # three points read consistently left-to-right as one trajectory).
            b = dim_avg(baseline, dim, "no_prompt")
            p2 = dim_avg(phase2, dim, "no_prompt") if phase2 else None
            p3 = dim_avg(phase3, dim, "matched") if phase3 else None
            if b is None and p2 is None and p3 is None:
                continue
            ax.plot(xs, [b, p2, p3], marker=_DIM_MARKERS[i % len(_DIM_MARKERS)],
                     linestyle=_DIM_LINESTYLES[i % len(_DIM_LINESTYLES)],
                     markersize=8, linewidth=1.5, alpha=0.75, label=dim.replace("_", " "))

        avg_b = aggregate_score(baseline, "no_prompt")
        avg_p2 = aggregate_score(phase2, "no_prompt") if phase2 else None
        avg_p3 = aggregate_score(phase3, "matched") if phase3 else None
        ax.plot(xs, [avg_b, avg_p2, avg_p3], marker="*", markersize=14, linewidth=3,
                 color="black", label="average (all dims)")

        ax.set_ylim(0.5, 5.5)
        ax.set_ylabel("score (1-5)")
        ax.set_title(f"Baseline vs. phase 2 vs. phase 3 — {model}, n{size:03d}")
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
        savefig(fig, out_dir / f"n{size:03d}" / f"baseline_vs_phase2_vs_phase3_{model}.png")
        log.append(f"OK   baseline_vs_phases {model} n{size}")


# ---------------------------------------------------------------------------
# 5 & 6. Cross-model ranking within size tier
# ---------------------------------------------------------------------------

def chart_cross_model_ranking(models: list[str], size: int, phase: int, condition: str,
                               out_dir: Path, log: list, version: str = None):
    tiers: dict[str, list[str]] = {}
    for m in models:
        tiers.setdefault(size_tier(m, version=version), []).append(m)

    any_chart = False
    for tier, tier_models in tiers.items():
        rows = {}
        for model in tier_models:
            entries = load_scores(model, size, phase, version=version)
            if entries:
                rows[model] = {d: dim_avg(entries, d, condition) for d in DIMENSIONS}
                rows[model]["average"] = aggregate_score(entries, condition)
        if not rows:
            continue
        categories = DIMENSIONS + ["average"]
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(rows) * len(categories)), 5))
        x = range(len(categories))
        width = 0.8 / max(len(rows), 1)
        for i, (model, dims) in enumerate(rows.items()):
            vals = [dims[d] if dims[d] is not None else 0 for d in categories]
            offset = (i - (len(rows) - 1) / 2) * width
            xs = [xi + offset for xi in x]
            bars = ax.bar(xs, vals, width, label=model, color=color_for(model))
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
        ax.axvline(len(DIMENSIONS) - 0.5, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.replace("_", "\n") for c in categories], fontsize=9)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("avg score (1-5)")
        safe_tier = tier.replace(" ", "_").replace("(", "").replace(")", "")
        safe_condition = condition.replace(" ", "_")
        ax.set_title(f"Cross-model ranking — {tier}, phase{phase}/{condition}, n{size:03d}")
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1))
        savefig(fig, out_dir / f"n{size:03d}" / f"ranking_phase{phase}_{safe_condition}_{safe_tier}.png")
        log.append(f"OK   ranking_phase{phase}/{condition} tier={tier} n{size}")
        any_chart = True
    if not any_chart:
        log.append(f"SKIP ranking phase{phase}/{condition} n{size}: no data for any tier")


# ---------------------------------------------------------------------------
# 7. Sample efficiency
# ---------------------------------------------------------------------------

def chart_sample_efficiency(models: list[str], size: int, phases: list[int], out_dir: Path, log: list,
                             version: str = None):
    results = {}  # model -> phase -> efficiency
    any_data = False
    for model in models:
        baseline = load_baseline(model, version=version)
        results[model] = {}
        for phase in phases:
            condition = "no_prompt" if phase == 2 else "matched"
            entries = load_scores(model, size, phase, version=version)
            summary_path = rm.get_results_path(model, size, phase, version=version) / "checkpoints_meta.json"
            if not entries or not summary_path.exists():
                continue
            meta = json.loads(summary_path.read_text())
            steps = meta.get("total_steps")
            baseline_score = aggregate_score(baseline, "no_prompt" if phase == 2 else "system_prompt")
            phase_score = aggregate_score(entries, condition)
            if steps and baseline_score is not None and phase_score is not None:
                eff = sample_efficiency(phase_score, baseline_score, steps)
                results[model][phase] = eff["sample_efficiency"]
                any_data = True

    if not any_data:
        log.append(f"SKIP sample_efficiency n{size}: insufficient data")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(models)), 5))
    x = range(len(models))
    width = 0.35
    for i, phase in enumerate(phases):
        vals = [results[m].get(phase, 0) for m in models]
        offset = (i - (len(phases) - 1) / 2) * width
        xs = [xi + offset for xi in x]
        bars = ax.bar(xs, vals, width, label=f"phase{phase}")
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], fontsize=8, padding=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("(phase score − baseline score) / training steps")
    ax.set_title(f"Sample efficiency — n{size:03d}")
    ax.legend()
    savefig(fig, out_dir / f"n{size:03d}" / "sample_efficiency.png")
    log.append(f"OK   sample_efficiency n{size}")


# ---------------------------------------------------------------------------
# 8. Prompt robustness (phase 3 only)
# ---------------------------------------------------------------------------

def chart_prompt_robustness(models: list[str], size: int, metric: str, out_dir: Path, log: list,
                             version: str = None):
    conditions = INFERENCE_CONDITIONS["phase3"]
    rows = {}
    for model in models:
        entries = load_scores(model, size, 3, version=version)
        if not entries:
            continue
        if metric == "aggregate":
            vals = {c: aggregate_score(entries, c) for c in conditions}
        else:
            vals = {c: dim_avg(entries, "system_prompt_dependency", c) for c in conditions}
        rows[model] = vals
    if not rows:
        log.append(f"SKIP prompt_robustness n{size}: no phase3 data")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(rows)), 5))
    x = range(len(rows))
    width = 0.8 / len(conditions)
    for i, cond in enumerate(conditions):
        vals = [rows[m][cond] if rows[m][cond] is not None else 0 for m in rows]
        offset = (i - (len(conditions) - 1) / 2) * width
        xs = [xi + offset for xi in x]
        bars = ax.bar(xs, vals, width, label=cond)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(rows.keys()), rotation=30, ha="right")
    ax.set_ylabel(f"{metric} score")
    ax.set_title(f"Prompt robustness, phase 3 ({metric}) — n{size:03d}")
    ax.legend()
    savefig(fig, out_dir / f"n{size:03d}" / f"prompt_robustness_{metric}.png")
    log.append(f"OK   prompt_robustness n{size}")


# ---------------------------------------------------------------------------
# 9. Phase 2 transfer test (no_prompt vs system_prompt)
# ---------------------------------------------------------------------------

def chart_phase2_transfer(models: list[str], size: int, metric: str, out_dir: Path, log: list,
                           version: str = None):
    rows = {}
    for model in models:
        entries = load_scores(model, size, 2, version=version)
        if not entries:
            continue
        if metric == "aggregate":
            vals = {c: aggregate_score(entries, c) for c in ("no_prompt", "system_prompt")}
        else:
            vals = {c: dim_avg(entries, "system_prompt_dependency", c) for c in ("no_prompt", "system_prompt")}
        rows[model] = vals
    if not rows:
        log.append(f"SKIP phase2_transfer n{size}: no phase2 data")
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(rows)), 5))
    x = range(len(rows))
    width = 0.35
    for i, cond in enumerate(("no_prompt", "system_prompt")):
        vals = [rows[m][cond] if rows[m][cond] is not None else 0 for m in rows]
        offset = (i - 0.5) * width
        xs = [xi + offset for xi in x]
        bars = ax.bar(xs, vals, width, label=cond)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals], fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(rows.keys()), rotation=30, ha="right")
    ax.set_ylabel(f"{metric} score")
    ax.set_title(f"Phase 2 transfer test: does an inference-time prompt help "
                 f"a model never trained with one? — n{size:03d}")
    ax.legend()
    savefig(fig, out_dir / f"n{size:03d}" / f"phase2_transfer_{metric}.png")
    log.append(f"OK   phase2_transfer n{size}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(out_dir: str = None, phases: list[int] = None, robustness_metric: str = "aggregate",
        dataset_sizes: list[int] = None, version: str = None) -> dict:
    """Generate every chart across all discovered (model, dataset_size)
    pairs for this pipeline generation, or only the given dataset_sizes if
    specified (e.g. right after a run-full for one particular size — no need
    to re-chart every size that happens to have old data lying around).
    out_dir defaults to charts/{version} — different generations' charts
    never mix in the same directory."""
    phases = phases or [2, 3]
    version = version or PIPELINE_VERSION
    out_dir = Path(out_dir) if out_dir else Path("charts") / version
    log = []

    pairs = discover_model_sizes(version=version)
    if dataset_sizes is not None:
        pairs = [(m, s) for m, s in pairs if s in dataset_sizes]
    if not pairs:
        print(f"No (model, dataset_size) pairs with real results found under results/{version}/ — nothing to chart.")
        return {"generated": 0, "skipped": 0, "out_dir": str(out_dir)}

    sizes = sorted({size for _, size in pairs})
    for size in sizes:
        models = sorted({m for m, s in pairs if s == size})
        print(f"\n=== n{size:03d}: {len(models)} model(s) ===")

        for model in models:
            for phase in phases:
                chart_loss_curves(model, size, phase, out_dir, log, version=version)
                chart_checkpoint_trajectory(model, phase, size, out_dir, log, version=version)

        chart_baseline_vs_phases(models, size, out_dir, log, version=version)
        for condition in INFERENCE_CONDITIONS["phase2"]:
            chart_cross_model_ranking(models, size, 2, condition, out_dir, log, version=version)
        for condition in INFERENCE_CONDITIONS["phase3"]:
            chart_cross_model_ranking(models, size, 3, condition, out_dir, log, version=version)
        chart_sample_efficiency(models, size, phases, out_dir, log, version=version)
        chart_prompt_robustness(models, size, robustness_metric, out_dir, log, version=version)
        chart_phase2_transfer(models, size, robustness_metric, out_dir, log, version=version)

    print("\n=== summary ===")
    for line in log:
        print(line)
    ok = sum(1 for l in log if l.startswith("OK"))
    skip = sum(1 for l in log if l.startswith("SKIP"))
    print(f"\n{ok} chart(s) generated, {skip} skipped. Output: {out_dir}/")
    return {"generated": ok, "skipped": skip, "out_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser(description="Generate documentation charts from experiment results.")
    parser.add_argument("--out-dir", default=None, help="Defaults to charts/{version}.")
    parser.add_argument("--phases", default="2,3", help="Comma-separated phases to include (training charts).")
    parser.add_argument("--version", default=PIPELINE_VERSION, help="Pipeline generation to chart (v1/v2/v3).")
    parser.add_argument("--robustness-metric", default="aggregate",
                         choices=["system_prompt_dependency", "aggregate"],
                         help="system_prompt_dependency is only ever scored for the LAST condition run "
                              "per test case (it needs sibling transcripts that don't exist yet when "
                              "earlier conditions are scored), so it's structurally incomplete — "
                              "'aggregate' (default) doesn't have that gap.")
    args = parser.parse_args()
    run(out_dir=args.out_dir, phases=[int(p) for p in args.phases.split(",")],
        robustness_metric=args.robustness_metric, version=args.version)


if __name__ == "__main__":
    main()
