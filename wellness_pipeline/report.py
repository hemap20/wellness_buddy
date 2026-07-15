"""
Reporting / aggregation module — standalone runnable stage.

Primary mode: discovers all completed runs via results/manifest.jsonl (rather
than walking the results/ directory tree), then loads each run's scores.json
(phase 2/3) or phase1_baseline.json (phase 1) from the path recorded in that
manifest entry, expanding to one row per (model, dataset_size, phase,
condition, test_case) with all dimension scores as columns. Also writes a
summary view aggregated per (model, dataset_size, phase, condition) across
test cases.

Never re-runs upstream stages — only reads what orchestrator.py's run-phase
already wrote.

Legacy mode (--legacy-scores-dir): the old flat-directory glob over
scores/*.scores.json, kept for standalone simulator.py/judge.py debugging
runs that don't go through results_manager.py at all.

Run standalone:
    python report.py
    python report.py --out reports/run1
    python report.py --legacy-scores-dir scores
"""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Optional

from config import REPORT_DIR, SCORE_DIR
import results_manager as rm

DIMENSIONS = [
    "persona_tone_adherence", "empathy_vs_parroting", "boundary_holding",
    "crisis_handling", "system_prompt_dependency", "multi_turn_drift",
]


def _mean_score(runs: list[dict], dim: str) -> Optional[float]:
    scores = [r[dim]["score"] for r in runs if isinstance(r[dim]["score"], int)]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _variance(runs: list[dict], dim: str) -> Optional[float]:
    scores = [r[dim]["score"] for r in runs if isinstance(r[dim]["score"], int)]
    if len(scores) < 2:
        return 0.0
    return statistics.pvariance(scores)


def _phase_to_int(phase) -> int:
    """Transcript/scores metadata stores phase as the string 'phase2'/'phase3'
    (from simulator.py's CLI); the manifest stores it as a plain int. Accept
    either so callers don't need to know which source they're reading from."""
    if isinstance(phase, int):
        return phase
    return int(str(phase).removeprefix("phase"))


def _load_training_meta(model: str, dataset_size, phase: int) -> dict:
    """val_loss (final checkpoint) + possible_overfitting, read from
    checkpoints_meta.json. None/False for phase 1 (no training) or if that
    file doesn't exist yet."""
    if phase == 1 or dataset_size is None:
        return {"val_loss_final": None, "possible_overfitting": False}
    meta_path = rm.get_results_path(model, dataset_size, phase) / "checkpoints_meta.json"
    if not meta_path.exists():
        return {"val_loss_final": None, "possible_overfitting": False}
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    val_loss_final = None
    curve_path = rm.get_results_path(model, dataset_size, phase) / "loss_curve.json"
    if curve_path.exists():
        with open(curve_path, encoding="utf-8") as f:
            curve = json.load(f)
        val_losses = curve.get("val_loss")
        if val_losses:
            val_loss_final = val_losses[-1]
    return {"val_loss_final": val_loss_final, "possible_overfitting": meta.get("possible_overfitting", False)}


def _row_from_entry(meta: dict, runs: list[dict], dataset_size, phase_int: int, source_path: str) -> dict:
    training_meta = _load_training_meta(meta.get("model_name"), dataset_size, phase_int)
    row = {
        "model": meta.get("model_name"),
        "dataset_size": dataset_size,
        "phase": meta.get("phase"),
        "condition": meta.get("condition"),
        "thinking_mode": meta.get("thinking_mode"),  # None for models without a thinking toggle
        "test_case": meta.get("test_case"),
        "simulator_mode": meta.get("simulator_mode"),
        "num_judge_runs": len(runs),
        "flagged_for_human_review": any(r.get("flag_for_human_review") for r in runs),
        "val_loss_final": training_meta["val_loss_final"],
        "possible_overfitting": training_meta["possible_overfitting"],
        "source_file": source_path,
    }
    for dim in DIMENSIONS:
        row[f"{dim}_mean"] = _mean_score(runs, dim)
        row[f"{dim}_variance"] = _variance(runs, dim)
    return row


# ---------------------------------------------------------------------------
# Manifest-driven (primary)
# ---------------------------------------------------------------------------

def build_rows_from_manifest() -> list[dict]:
    rows = []
    seen_baseline_models = set()

    for manifest_entry in rm.load_manifest():
        model = manifest_entry["model"]
        phase = manifest_entry["phase"]
        dataset_size = manifest_entry["dataset_size"]

        if phase == 1:
            if model in seen_baseline_models:
                continue  # baseline file already fully expanded from an earlier manifest line
            seen_baseline_models.add(model)
            baseline = rm.load_baseline(model)
            if not baseline:
                continue
            for entry in baseline["results"]:
                rows.append(_row_from_entry(entry["metadata"], entry["judge_runs"], None, _phase_to_int(phase),
                                             manifest_entry["path"]))
        else:
            scores = rm.load_scores(model, dataset_size, phase)
            for entry in scores:
                rows.append(_row_from_entry(entry["metadata"], entry["judge_runs"], dataset_size, _phase_to_int(phase),
                                             manifest_entry["path"]))

    # de-dupe phase2/3 rows in case multiple manifest lines point at the same
    # scores.json (one manifest line per condition, but scores.json accumulates
    # every condition for that phase — dedupe by the natural row key)
    dedup = {}
    for row in rows:
        key = (row["model"], row["dataset_size"], row["phase"], row["condition"],
               row["thinking_mode"], row["test_case"])
        dedup[key] = row
    return list(dedup.values())


# ---------------------------------------------------------------------------
# Legacy flat-directory (for standalone simulator.py/judge.py debugging runs)
# ---------------------------------------------------------------------------

def build_rows_legacy(scores_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(Path(scores_dir) / "*.scores.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(_row_from_entry(data["metadata"], data["judge_runs"], None,
                                     _phase_to_int(data["metadata"].get("phase", 1)), path))
    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model"], row["dataset_size"], row["phase"], row["condition"], row["thinking_mode"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (model, dataset_size, phase, condition, thinking_mode), group_rows in sorted(
        groups.items(), key=lambda kv: [str(x) for x in kv[0]]
    ):
        entry = {
            "model": model, "dataset_size": dataset_size, "phase": phase, "condition": condition,
            "thinking_mode": thinking_mode,
            "num_test_cases": len(group_rows),
            "any_flagged_for_human_review": any(r["flagged_for_human_review"] for r in group_rows),
            # Same value on every row in this group (one training run per
            # model/dataset_size/phase) — just read it off the first row
            # rather than aggregating, since it's a fact about the training
            # run, not a per-test-case judge score.
            "val_loss_final": group_rows[0].get("val_loss_final"),
            "possible_overfitting": group_rows[0].get("possible_overfitting"),
        }
        for dim in DIMENSIONS:
            values = [r[f"{dim}_mean"] for r in group_rows if r[f"{dim}_mean"] is not None]
            entry[f"{dim}_avg"] = sum(values) / len(values) if values else None
        summary.append(entry)
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(legacy_scores_dir: str = None, out_prefix: str = None) -> dict:
    out_prefix = out_prefix or str(REPORT_DIR / "report")
    rows = build_rows_legacy(legacy_scores_dir) if legacy_scores_dir else build_rows_from_manifest()
    summary = build_summary(rows)

    detail_json = Path(f"{out_prefix}_detail.json")
    detail_csv = Path(f"{out_prefix}_detail.csv")
    summary_json = Path(f"{out_prefix}_summary.json")
    summary_csv = Path(f"{out_prefix}_summary.csv")

    detail_json.write_text(json.dumps(rows, indent=2))
    write_csv(rows, detail_csv)
    summary_json.write_text(json.dumps(summary, indent=2))
    write_csv(summary, summary_csv)

    return {
        "source": "legacy_scores_dir" if legacy_scores_dir else "manifest",
        "num_detail_rows": len(rows),
        "num_summary_rows": len(summary),
        "detail_json": str(detail_json),
        "detail_csv": str(detail_csv),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
    }


def main():
    parser = argparse.ArgumentParser(description="Reporting stage: aggregate scores + metadata into CSV/JSON.")
    parser.add_argument("--legacy-scores-dir", default=None,
                         help=f"Use the old flat-directory glob instead of results/manifest.jsonl "
                              f"(e.g. '{SCORE_DIR}' for standalone simulator.py/judge.py debug runs).")
    parser.add_argument("--out", default=None, help="Output path prefix (no extension)")
    args = parser.parse_args()

    result = run(args.legacy_scores_dir, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
