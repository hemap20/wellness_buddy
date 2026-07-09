"""
Reporting / aggregation module — standalone runnable stage.

Collects judge scores + automatic metrics + metadata into one row per
(model, phase, condition, test_case), writes CSV + JSON, and a summary view
aggregated per (model, phase, condition) across test cases.

Runs entirely against files already on disk (scores/*.scores.json,
transcripts/*.json) — never re-runs upstream stages.

Run standalone:
    python report.py
    python report.py --scores-dir scores --out reports/run1
"""
import argparse
import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Optional

from config import REPORT_DIR, SCORE_DIR

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


def build_rows(scores_dir: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(Path(scores_dir) / "*.scores.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"]
        runs = data["judge_runs"]

        row = {
            "model": meta.get("model_name"),
            "phase": meta.get("phase"),
            "condition": meta.get("condition"),
            "test_case": meta.get("test_case"),
            "simulator_mode": meta.get("simulator_mode"),
            "num_judge_runs": len(runs),
            "flagged_for_human_review": any(r.get("flag_for_human_review") for r in runs),
            "source_scores_file": path,
        }
        for dim in DIMENSIONS:
            row[f"{dim}_mean"] = _mean_score(runs, dim)
            row[f"{dim}_variance"] = _variance(runs, dim)
        rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["model"], row["phase"], row["condition"])
        groups.setdefault(key, []).append(row)

    summary = []
    for (model, phase, condition), group_rows in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        entry = {
            "model": model, "phase": phase, "condition": condition,
            "num_test_cases": len(group_rows),
            "any_flagged_for_human_review": any(r["flagged_for_human_review"] for r in group_rows),
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


def run(scores_dir: str = None, out_prefix: str = None) -> dict:
    scores_dir = scores_dir or str(SCORE_DIR)
    out_prefix = out_prefix or str(REPORT_DIR / "report")

    rows = build_rows(scores_dir)
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
        "num_detail_rows": len(rows),
        "num_summary_rows": len(summary),
        "detail_json": str(detail_json),
        "detail_csv": str(detail_csv),
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
    }


def main():
    parser = argparse.ArgumentParser(description="Reporting stage: aggregate scores + metadata into CSV/JSON.")
    parser.add_argument("--scores-dir", default=None)
    parser.add_argument("--out", default=None, help="Output path prefix (no extension)")
    args = parser.parse_args()

    result = run(args.scores_dir, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
