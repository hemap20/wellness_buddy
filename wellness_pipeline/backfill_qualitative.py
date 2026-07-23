"""
One-off backfill: adds the "qualitative" (3 strengths / 3 weaknesses) field
to every already-scored transcript that predates judge.py's qualitative pass.

Never re-runs the numeric rubric (existing scores/justifications are left
untouched) — for each entry missing "qualitative", it re-loads the saved
transcript, reuses the existing judge_runs as grounding context, and makes
one new LLM call to generate the qualitative read, then rewrites it in place.

Requires JUDGE_CONFIG's provider to be reachable (ANTHROPIC_API_KEY or
GEMINI_API_KEY/GOOGLE_API_KEY set) — see llm_client.py.

Run standalone:
    python backfill_qualitative.py                       # backfill everything missing it, 2s between calls
    python backfill_qualitative.py --sleep-seconds 5      # slower, for a tighter rate limit
    python backfill_qualitative.py --dry-run              # just report what would run
"""
import argparse
import json
import time

import judge
import results_manager as rm
from config import JUDGE_CONFIG
from llm_client import LLMClient


def _load_phase2_3_transcript(model: str, dataset_size: int, phase: int, meta: dict) -> dict:
    transcripts_dir = rm.get_results_path(model, dataset_size, phase, "transcripts")
    base = f"{meta['test_case']}_{meta['condition']}_{meta['simulator_mode']}"

    # thinking-mode probe runs (some n050 phase2/3 entries for gemma/qwen3
    # variants) save two transcripts per test_case/condition/simulator_mode —
    # one per thinking_mode value — with that appended to the filename, which
    # the base pattern above doesn't capture. Metadata carries thinking_mode,
    # so build the exact name when present; fall back to a glob for anything
    # else unanticipated rather than hard-failing the whole backfill run.
    thinking_mode = meta.get("thinking_mode")
    if thinking_mode is not None:
        base += "_thinking_on" if thinking_mode else "_thinking_off"

    path = transcripts_dir / f"{base}.json"
    if not path.exists():
        matches = list(transcripts_dir.glob(f"{meta['test_case']}_{meta['condition']}_*.json"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one transcript for {meta['test_case']}/{meta['condition']} "
                f"in {transcripts_dir}, found {len(matches)}: {matches}"
            )
        path = matches[0]

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def backfill_model_phase(model: str, dataset_size, phase: int, dry_run: bool, sleep_seconds: float) -> tuple[int, list[str]]:
    """Returns (num_filled, list_of_failure_descriptions). A single transcript
    that the judge can't produce valid qualitative JSON for (rare, but a real
    LLM failure mode after exhausting max_json_retries) must not abort the
    other 90+ transcripts still queued behind it — log it and move on; rerun
    the script afterward to retry just the ones still missing "qualitative"."""
    client = None if dry_run else LLMClient(JUDGE_CONFIG.provider, JUDGE_CONFIG.model)
    filled = 0
    failures = []

    if phase == 1:
        baseline = rm.load_baseline(model)
        if not baseline:
            return 0, []
        for entry in baseline["results"]:
            if "qualitative" in entry:
                continue
            if dry_run:
                filled += 1
                continue
            try:
                prompt = judge._build_judge_prompt({"metadata": entry["metadata"], "transcript": entry["transcript"]}, None)
                entry["qualitative"] = judge._generate_qualitative(client, JUDGE_CONFIG, prompt, entry["judge_runs"])
            except Exception as exc:
                failures.append(f"{model} n=None phase1 {entry['metadata'].get('test_case')}/"
                                 f"{entry['metadata'].get('condition')}: {type(exc).__name__}: {exc}")
                time.sleep(sleep_seconds)
                continue
            filled += 1
            # Save after every single entry, not once at the end of the model —
            # a rate-limit error 40 calls into a 456-call run must not lose the
            # 39 already paid for; a re-run just skips whatever already has
            # "qualitative" set.
            rm.save_baseline(model, baseline["results"])
            time.sleep(sleep_seconds)
        return filled, failures

    scores = rm.load_scores(model, dataset_size, phase)
    if not scores:
        return 0, []
    scores_path = rm.get_results_path(model, dataset_size, phase) / "scores.json"
    for entry in scores:
        if "qualitative" in entry:
            continue
        if dry_run:
            filled += 1
            continue
        try:
            record = _load_phase2_3_transcript(model, dataset_size, phase, entry["metadata"])
            prompt = judge._build_judge_prompt(record, None)
            entry["qualitative"] = judge._generate_qualitative(client, JUDGE_CONFIG, prompt, entry["judge_runs"])
        except Exception as exc:
            failures.append(f"{model} n={dataset_size} phase{phase} {entry['metadata'].get('test_case')}/"
                             f"{entry['metadata'].get('condition')}: {type(exc).__name__}: {exc}")
            time.sleep(sleep_seconds)
            continue
        filled += 1
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)
        time.sleep(sleep_seconds)
    return filled, failures


def main():
    parser = argparse.ArgumentParser(description="Backfill qualitative strengths/weaknesses onto existing scores.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be backfilled without calling the judge LLM")
    parser.add_argument("--sleep-seconds", type=float, default=2.0,
                         help="Pause between judge LLM calls to stay under rate limits (default: 2.0s)")
    args = parser.parse_args()

    seen = set()
    total_filled = 0
    all_failures = []
    for manifest_entry in rm.load_manifest():
        model = manifest_entry["model"]
        dataset_size = manifest_entry["dataset_size"]
        phase = manifest_entry["phase"]
        key = (model, dataset_size, phase)
        if key in seen:
            continue
        seen.add(key)

        filled, failures = backfill_model_phase(model, dataset_size, phase, args.dry_run, args.sleep_seconds)
        if filled:
            verb = "would backfill" if args.dry_run else "backfilled"
            print(f"[{model} n={dataset_size} phase{phase}] {verb} {filled} transcript(s)")
        total_filled += filled
        all_failures.extend(failures)

    print(f"\nTotal: {total_filled} transcript(s) {'need' if args.dry_run else 'received'} qualitative backfill.")
    if all_failures:
        print(f"\n{len(all_failures)} transcript(s) FAILED and were skipped (rerun the script to retry just these):")
        for f in all_failures:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
