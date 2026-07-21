"""
Automatic (non-judge) metrics module — standalone runnable stage.

Computed directly from training/inference logs, kept separate from the
judge-scored dimensions in the final report:
  - training loss curve (+ flat/non-decreasing flag)
  - latency (avg / p95 / p99) from simulator-run transcripts
  - sample efficiency: (checkpoint score - phase-1 baseline score) / steps-or-examples-seen

Run standalone:
    python metrics.py loss --loss-log logs/dialogpt-small_phase2_loss.jsonl
    python metrics.py latency --transcripts "transcripts/dialogpt-small_phase2_*.json"
    python metrics.py efficiency --checkpoint-score 3.2 --baseline-score 1.8 --steps 15
"""
import argparse
import glob
import json
import math
from pathlib import Path


def loss_curve_summary(loss_log_path: str, tolerance: float = 1e-3) -> dict:
    values = []
    with open(loss_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(json.loads(line)["loss"])
    if not values:
        return {"values": [], "flat_or_nondecreasing": False, "note": "empty loss log"}
    flat_or_nondecreasing = values[-1] >= values[0] - tolerance
    return {
        "values": values,
        "first_loss": values[0],
        "last_loss": values[-1],
        "min_loss": min(values),
        "max_loss": max(values),
        "flat_or_nondecreasing": flat_or_nondecreasing,
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    k = (len(sorted_values) - 1) * (pct / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def latency_stats(transcript_glob: str) -> dict:
    latencies = []
    paths = sorted(glob.glob(transcript_glob))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        for turn in record["transcript"]:
            if turn.get("role") == "assistant" and "latency_seconds" in turn:
                latencies.append(turn["latency_seconds"])
    if not latencies:
        return {"count": 0, "avg": None, "p95": None, "p99": None, "transcripts_scanned": len(paths)}
    latencies.sort()
    return {
        "count": len(latencies),
        "avg": sum(latencies) / len(latencies),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "transcripts_scanned": len(paths),
    }


def sample_efficiency(checkpoint_score: float, baseline_score: float, steps_or_examples: int) -> dict:
    if steps_or_examples <= 0:
        raise ValueError("steps_or_examples must be > 0")
    delta = checkpoint_score - baseline_score
    return {
        "checkpoint_score": checkpoint_score,
        "baseline_score": baseline_score,
        "steps_or_examples": steps_or_examples,
        "delta": delta,
        "sample_efficiency": delta / steps_or_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Automatic metrics stage.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_loss = sub.add_parser("loss")
    p_loss.add_argument("--loss-log", required=True)

    p_lat = sub.add_parser("latency")
    p_lat.add_argument("--transcripts", required=True, help="glob pattern, e.g. 'transcripts/*.json'")

    p_eff = sub.add_parser("efficiency")
    p_eff.add_argument("--checkpoint-score", type=float, required=True)
    p_eff.add_argument("--baseline-score", type=float, required=True)
    p_eff.add_argument("--steps", type=int, required=True)

    args = parser.parse_args()

    if args.cmd == "loss":
        result = loss_curve_summary(args.loss_log)
        if result.get("flat_or_nondecreasing"):
            print("WARNING: loss curve is flat or non-decreasing.")
    elif args.cmd == "latency":
        result = latency_stats(args.transcripts)
    elif args.cmd == "efficiency":
        result = sample_efficiency(args.checkpoint_score, args.baseline_score, args.steps)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

'''
- remove the thinking mode testing altogether. even if the model has a thinking mode, we'll not use it
- keep the original 6 metrics, remove the ones we added for thinking mode
- train.py needs actual resume logic — load the last saved intermediate checkpoint's weights/optimizer state and continue from that step, rather than just skip-if-done. Right now that logic isn't there. This is a real code change, not just a storage config change. you add a --resume-from-checkpoint path that loads the latest intermediate checkpoint and picks the step counter back up.
- a background loop that commits every N minutes regardless of pipeline state — simpler to bolt on, less precise about when it commits.
- i'd want .gitignore already excluding checkpoints/
- i want the code to first do phase 1 testing, then do phase 2 training, then phase 2 testing, then phase 3 training and then phase 3 testing. 
- the timings file is getting very cluttered, if the model is skipped, no need to log it.  
- remove the concept of "any_flagged_for_human_review"
- remove existing traces of dry-run
- If you're mid-run when you git commit, future reruns become hard to attribute correctly. Try to commit before kicking off a batch of runs, not after — keeps git_commit entries clean (no -dirty suffix) and trustworthy as your source of truth for "what code produced this."
- i want the judge LLM to give a subjective opinion on how the model's performance is after each phase
it should tell what went right and what went wrong, a small answer. 
'''
