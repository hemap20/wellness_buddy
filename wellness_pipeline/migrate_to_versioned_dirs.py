"""
One-time migration: move today's flat results/checkpoints/logs/reports/charts
trees into a versioned v1/v2/v3 layout matching the new version-aware path
builders in results_manager.py and config.py.

NOT part of the pipeline — run once by hand, inspect the --dry-run output
first, then run for real. Idempotent: re-running after a successful move is
a no-op (nothing left at the old locations to move).

Layout produced:
    results/v1/{model}_v1/...        <- the two pre-existing "_v1" dirs, just moved
    results/v2/{model}_v2/...        <- every other current unsuffixed model dir, moved + suffixed
    results/v2/manifest.jsonl        <- current manifest.jsonl copied verbatim (see NOTE below)
    results/v2/timings.jsonl         <- current timings.jsonl copied verbatim
    checkpoints/v1/{model}_v1_...    <- the two "_v1"-tagged checkpoint dirs
    checkpoints/v2/{model}_v2_...    <- every other current checkpoint dir (including the old
                                         pre-size-namespaced "{model}_phase2"/"_phase3" dirs), suffixed
    logs/v1/{model}_v1_..._loss.jsonl
    logs/v2/{model}_v2_..._loss.jsonl
    charts/v2/{n050,n500,n1228}      <- current chart data, all generated from v2-era models
    reports/v2/*                     <- current reports/ moved wholesale (ambiguous filenames,
                                         per your instruction to skip per-file attribution)

NOTE on manifest.jsonl/timings.jsonl: inspecting the actual content shows it's
an append-only historical log spanning multiple environments (some entries
reference "/workspace/wellness_buddy/..." paths from a different machine, and
model names like "dialogpt-small-v2"/"dialogpt-small-dryrun" that don't
correspond to any directory that exists on this disk today). There is no
clean, non-guessing way to split it into v1-only vs v2-only content, so both
files are copied verbatim into results/v2/ — this migration does not attempt
to reconstruct history it can't verify.

Usage:
    python migrate_to_versioned_dirs.py --dry-run
    python migrate_to_versioned_dirs.py
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
CHART_DIR = ROOT / "charts"
REPORT_DIR = ROOT / "reports"

V1_MODELS = {"dialogpt-small", "gemma-4-e2b-it"}  # the only two models with pre-existing "_v1" data


def plan_moves() -> list[tuple[Path, Path, str]]:
    """Returns [(src, dst, description), ...]. Pure planning, no filesystem writes."""
    moves = []

    # --- results/ ---
    if RESULTS_DIR.is_dir():
        for entry in sorted(RESULTS_DIR.iterdir()):
            if entry.name in ("manifest.jsonl", "timings.jsonl"):
                dst = RESULTS_DIR / "v2" / entry.name
                moves.append((entry, dst, "results log file -> v2/ (copied verbatim, see NOTE in docstring)"))
                continue
            if not entry.is_dir():
                continue
            if entry.name.endswith("_v1"):
                dst = RESULTS_DIR / "v1" / entry.name
                moves.append((entry, dst, "pre-existing v1 model results -> v1/"))
            else:
                dst = RESULTS_DIR / "v2" / f"{entry.name}_v2"
                moves.append((entry, dst, "current model results -> v2/ (suffix added)"))

    # --- checkpoints/ ---
    if CHECKPOINT_DIR.is_dir():
        for entry in sorted(CHECKPOINT_DIR.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if "_v1_" in name or name.endswith("_v1"):
                dst = CHECKPOINT_DIR / "v1" / name
                moves.append((entry, dst, "pre-existing v1 checkpoint dir -> v1/"))
            else:
                # e.g. "dialogpt-small_n050_phase2" -> "dialogpt-small_v2_n050_phase2"
                # e.g. "dialogpt-small_phase2" (old pre-size-namespaced) -> "dialogpt-small_v2_phase2"
                model_part, _, rest = name.partition("_")
                new_name = f"{model_part}_v2_{rest}" if rest else f"{name}_v2"
                dst = CHECKPOINT_DIR / "v2" / new_name
                moves.append((entry, dst, "current checkpoint dir -> v2/ (suffix inserted)"))

    # --- logs/ ---
    if LOG_DIR.is_dir():
        for entry in sorted(LOG_DIR.iterdir()):
            if not entry.is_file():
                continue
            name = entry.name
            if "_v1_" in name or "_v1." in name:
                dst = LOG_DIR / "v1" / name
                moves.append((entry, dst, "pre-existing v1 log file -> v1/"))
            else:
                model_part, _, rest = name.partition("_")
                new_name = f"{model_part}_v2_{rest}" if rest else f"{name}"
                dst = LOG_DIR / "v2" / new_name
                moves.append((entry, dst, "current log file -> v2/ (suffix inserted)"))

    # --- charts/ ---
    if CHART_DIR.is_dir():
        for entry in sorted(CHART_DIR.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in ("v1", "v2", "v3"):
                continue
            dst = CHART_DIR / "v2" / entry.name
            moves.append((entry, dst, "current chart dir -> v2/ (all v2-era data)"))

    # --- reports/ ---
    if REPORT_DIR.is_dir():
        for entry in sorted(REPORT_DIR.iterdir()):
            if entry.name in ("v1", "v2", "v3"):
                continue
            dst = REPORT_DIR / "v2" / entry.name
            moves.append((entry, dst, "current report file -> v2/ (moved wholesale, no per-file attribution)"))

    return moves


def execute_moves(moves: list[tuple[Path, Path, str]]) -> None:
    for src, dst, _ in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            print(f"[skip] {dst} already exists — not overwriting", file=sys.stderr)
            continue
        if src.name in ("manifest.jsonl", "timings.jsonl"):
            shutil.copy2(src, dst)  # copy, not move — keep the original in place as a safety net
        else:
            shutil.move(str(src), str(dst))
        print(f"[done] {src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned moves without touching the filesystem")
    args = parser.parse_args()

    moves = plan_moves()
    if not moves:
        print("Nothing to migrate — results/checkpoints/logs/charts/reports already look migrated or empty.")
        return

    print(f"{'DRY RUN — ' if args.dry_run else ''}Planned moves ({len(moves)}):\n")
    for src, dst, desc in moves:
        rel_src = src.relative_to(ROOT)
        rel_dst = dst.relative_to(ROOT)
        print(f"  {rel_src}\n    -> {rel_dst}   [{desc}]")
    print()

    if args.dry_run:
        print("Dry run only — nothing was moved. Re-run without --dry-run to execute.")
        return

    execute_moves(moves)
    print(f"\nMigration complete — {len(moves)} moves.")


if __name__ == "__main__":
    main()
