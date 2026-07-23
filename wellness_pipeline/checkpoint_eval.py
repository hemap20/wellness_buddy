"""
Checkpoint-based evaluation + best-checkpoint selection (v3).

Reuses the pipeline's existing eval suite — simulator.py + judge.py — which
already scores exactly the 6 dimensions this needs
(persona_tone_adherence, empathy_vs_parroting, boundary_holding,
crisis_handling, multi_turn_drift, system_prompt_dependency). Nothing new is
built for scoring itself; this module only adds: running that suite against
each intermediate checkpoint (not just the final adapter), aggregating scores
into a trajectory, and picking which checkpoint to actually ship.

Scope per checkpoint (confirmed): all 8 test cases, 1 inference condition,
judge_cfg.num_runs overridden to config.CHECKPOINT_EVAL_JUDGE_RUNS (1) — a
deliberately cheaper pass than final phase2/3 scoring (which keeps
JudgeConfig.num_runs=3). This module never writes into the real results/
scores.json tree — checkpoint-eval transcripts/scores are scratch data, kept
under each checkpoint's own directory for inspection, not merged into
final results.

Selection rule: default to the final checkpoint. If boundary_holding or
crisis_handling at the final checkpoint has dropped more than
config.SAFETY_REGRESSION_THRESHOLD points below that dimension's own max
across the four checkpoints, select the max-scoring checkpoint instead and
record why.

Run standalone (after train.py has produced checkpoints for this run):
    python checkpoint_eval.py --model dialogpt-small --num-samples 50 --phase 2
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import config
import judge
import results_manager as rm
import simulator
from config import INFERENCE_CONDITIONS, JUDGE_CONFIG, system_prompt_for_condition
from inference import ModelRunner

SAFETY_DIMENSIONS = ("boundary_holding", "crisis_handling")
ALL_DIMENSIONS = (
    "persona_tone_adherence", "empathy_vs_parroting", "boundary_holding",
    "crisis_handling", "multi_turn_drift", "system_prompt_dependency",
)


def discover_checkpoints(ckpt_root: Path) -> list[Path]:
    """Returns intermediate step checkpoints sorted by step number, ending
    with 'final'. 'final' is included so the trajectory always covers what
    would otherwise ship by default."""
    steps = sorted(
        ckpt_root.glob("step_*_pct_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    final = ckpt_root / "final"
    return steps + ([final] if final.exists() else [])


def eval_checkpoint(model_cfg, ckpt_path: Path, phase: int, dataset_size: int) -> dict[str, float]:
    """Runs the eval suite (all 8 test cases, 1 condition, 1 judge run)
    against this checkpoint's adapter weights. Returns {dimension: avg_score}
    (system_prompt_dependency omitted from the average if judge returned
    "N/A" for every test case, since that dimension only applies when
    matched-condition sibling transcripts exist)."""
    runner = ModelRunner(model_cfg, adapter_path=str(ckpt_path))
    phase_key = f"phase{phase}"
    condition = INFERENCE_CONDITIONS[phase_key][0]  # single fixed condition — this is a cheap directional signal, not final scoring
    judge_cfg = dataclasses.replace(JUDGE_CONFIG, num_runs=config.CHECKPOINT_EVAL_JUDGE_RUNS)

    scores_by_dim = {dim: [] for dim in ALL_DIMENSIONS}
    eval_dir = ckpt_path / "checkpoint_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    for test_case_id in simulator.list_test_cases():
        test_case = simulator.load_test_case(test_case_id)
        system_prompt = system_prompt_for_condition(condition)
        result = simulator.run_conversation(
            runner.generate, test_case, system_prompt,
            metadata={"model_name": model_cfg.name, "phase": phase_key, "condition": condition, "adapter": str(ckpt_path)},
        )
        scoring = judge.score_transcript(result, related_records=None, judge_cfg=judge_cfg)
        with open(eval_dir / f"{test_case_id}_{condition}.json", "w", encoding="utf-8") as f:
            json.dump({"metadata": result["metadata"], **scoring}, f, indent=2)

        for run in scoring["judge_runs"]:
            for dim in ALL_DIMENSIONS:
                score = run.get(dim, {}).get("score")
                if isinstance(score, (int, float)):
                    scores_by_dim[dim].append(float(score))

    result = {dim: (sum(vals) / len(vals) if vals else None) for dim, vals in scores_by_dim.items()}

    # Explicit cleanup — run_checkpoint_eval_for_run calls this in a loop,
    # once per checkpoint (5 per phase: 4 fractions + final). HF/PEFT model
    # objects commonly hold reference cycles (autograd graph, parent-child
    # module refs) that Python's refcounting alone won't collect, so without
    # this, GPU memory from the previous checkpoint's full base-model load
    # can still be resident when the next one loads — a real OOM risk for
    # the 7-8B models specifically. Same pattern lora_target_module_diagnostic.py
    # already uses for the same reason.
    import gc

    import torch

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def select_best_checkpoint(trajectory: list[tuple[Path, dict]]) -> tuple[Path, str]:
    """trajectory: [(checkpoint_path, {dimension: avg_score}), ...] in step
    order, last entry is the final checkpoint. Returns (selected_path, reason)."""
    if not trajectory:
        raise ValueError("empty trajectory — nothing to select from")

    final_path, final_scores = trajectory[-1]
    reasons = []
    fallback_path = final_path

    for dim in SAFETY_DIMENSIONS:
        dim_scores = [(path, scores.get(dim)) for path, scores in trajectory if scores.get(dim) is not None]
        if len(dim_scores) < 2:
            continue
        max_path, max_score = max(dim_scores, key=lambda ps: ps[1])
        final_score = final_scores.get(dim)
        if final_score is None:
            continue
        if max_score - final_score > config.SAFETY_REGRESSION_THRESHOLD:
            reasons.append(
                f"{dim} declined from {max_score:.1f} (at {max_path.name}) to {final_score:.1f} "
                f"at the final checkpoint (> {config.SAFETY_REGRESSION_THRESHOLD}pt threshold)"
            )
            fallback_path = max_path  # if multiple safety dims regress, the last one checked wins the fallback — both are still logged

    if reasons:
        return fallback_path, "; ".join(reasons)
    return final_path, "final checkpoint selected: no safety-dimension regression beyond threshold"


def run_checkpoint_eval_for_run(model_cfg, phase: int, dataset_size: int, version: str = None) -> dict:
    version = version or config.PIPELINE_VERSION
    run_name = f"{model_cfg.name}_{version}_{rm.format_dataset_size(dataset_size)}_phase{phase}"
    ckpt_root = config.CHECKPOINT_DIR / version / run_name
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"No checkpoints found at {ckpt_root} — run train.py for this model/phase first.")

    checkpoints = discover_checkpoints(ckpt_root)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint dirs matching step_*_pct_*/final under {ckpt_root}")

    print(f"=== {model_cfg.name} phase{phase} — evaluating {len(checkpoints)} checkpoints ===")
    trajectory = []
    for ckpt_path in checkpoints:
        print(f"  evaluating {ckpt_path.name}...")
        scores = eval_checkpoint(model_cfg, ckpt_path, phase, dataset_size)
        trajectory.append((ckpt_path, scores))
        print(f"    {scores}")

    selected_path, reason = select_best_checkpoint(trajectory)
    print(f"[selected] {selected_path.name} — {reason}")

    report = {
        "model": model_cfg.name,
        "phase": phase,
        "dataset_size": dataset_size,
        "version": version,
        "trajectory": [{"checkpoint": p.name, "scores": s} for p, s in trajectory],
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_name": selected_path.name,
        "reason": reason,
    }
    _write_trajectory_report(model_cfg.name, report, version, phase, dataset_size)
    return report


def _write_trajectory_report(model_name: str, report: dict, version: str, phase: int, dataset_size: int) -> None:
    report_dir = config.REPORT_DIR / version
    report_dir.mkdir(parents=True, exist_ok=True)
    # Filename MUST include phase + dataset_size — without them, running
    # checkpoint-eval for phase2 then phase3 of the SAME model (exactly what
    # run-v3 does in one call) would silently overwrite phase2's report with
    # phase3's at the same path, and likewise for two different --dataset-size
    # runs of the same model/phase.
    base_name = f"{model_name}_{rm.format_dataset_size(dataset_size)}_phase{phase}_checkpoint_trajectory"
    path = report_dir / f"{base_name}.md"

    lines = [f"# Checkpoint trajectory — {model_name} phase{report['phase']} ({version})\n"]
    lines.append(f"| checkpoint | {' | '.join(ALL_DIMENSIONS)} |")
    lines.append(f"|---|{'---|' * len(ALL_DIMENSIONS)}")
    for entry in report["trajectory"]:
        scores = entry["scores"]
        row = [f"{scores.get(dim):.2f}" if scores.get(dim) is not None else "N/A" for dim in ALL_DIMENSIONS]
        lines.append(f"| {entry['checkpoint']} | {' | '.join(row)} |")
    lines.append(f"\n**Selected**: `{report['selected_checkpoint_name']}`\n\n**Reason**: {report['reason']}\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    with open(report_dir / f"{base_name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[report] wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--phase", type=int, required=True, choices=[2, 3])
    parser.add_argument("--version", type=str, default=config.PIPELINE_VERSION)
    args = parser.parse_args()

    model_cfg = next((m for m in config.MODELS_UNDER_TEST if m.name == args.model), None)
    if model_cfg is None:
        raise SystemExit(f"Unknown model: {args.model}")

    report = run_checkpoint_eval_for_run(model_cfg, args.phase, args.num_samples, args.version)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
