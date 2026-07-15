"""
Results storage & manifest logic — standalone, reusable module.

Every experiment run (train + inference + judge-scoring for a given model +
dataset_size + phase [+ condition]) saves to a deterministic path under
`results/`, and gets one line appended to `results/manifest.jsonl`. Nothing
else in the pipeline should build these paths by hand — import from here.

Directory layout produced:

    results/
      {model_name}/
        phase1_baseline.json          <- per-model only, NOT per dataset size
        n{dataset_size}/               <- zero-padded, e.g. n050
          phase2/
            transcripts/
            scores.json                <- ALL judge runs kept, never averaged down
            loss_curve.json
            checkpoints_meta.json      <- step numbers + paths, not weights
          phase3/
            (same structure)
      manifest.jsonl                   <- flat, append-only index

Run standalone to see a self-contained demo/test (uses a temp directory, never
touches your real `results/`):

    python3 results_manager.py
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Naming / path helpers
# ---------------------------------------------------------------------------

def sanitize_model_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\\/]+", "-", name)      # slashes (e.g. "microsoft/DialoGPT-small") -> "-"
    name = re.sub(r"\s+", "_", name)          # whitespace -> "_"
    name = re.sub(r"[^a-z0-9._-]", "", name)  # drop anything else non-filesystem-safe
    return name or "unnamed-model"


def format_dataset_size(n: int) -> str:
    width = max(3, len(str(n)))
    return f"n{str(n).zfill(width)}"


def get_git_commit(cwd: Optional[Path] = None) -> str:
    cwd = cwd or Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    try:
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = False
    return f"{commit}-dirty" if dirty else commit


def make_run_id(model_name: str, dataset_size: Optional[int], phase: int,
                 condition: Optional[str], timestamp: Optional[str] = None,
                 thinking_mode: Optional[bool] = None) -> str:
    timestamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    size_part = format_dataset_size(dataset_size) if dataset_size is not None else "nNA"
    cond_part = condition or "na"
    thinking_part = "" if thinking_mode is None else ("_thinking_on" if thinking_mode else "_thinking_off")
    return f"{sanitize_model_name(model_name)}_{size_part}_phase{phase}_{cond_part}{thinking_part}_{timestamp}"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_results_path(model_name: str, dataset_size: Optional[int], phase: int,
                      subfolder: Optional[str] = None, root: Path = None) -> Path:
    """Returns the path for this model/dataset_size/phase, creating parent
    directories as needed.

    phase == 1 (baseline) is a special case: returns the FILE path
    `results/{model}/phase1_baseline.json` directly, ignoring dataset_size
    and subfolder entirely (baseline has no per-condition/per-subfolder
    breakdown — it's one JSON file covering every test case/condition).
    """
    root = root or DEFAULT_RESULTS_ROOT
    model_dir = root / sanitize_model_name(model_name)

    if phase == 1:
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir / "phase1_baseline.json"

    if dataset_size is None:
        raise ValueError("dataset_size is required for phase 2/3 paths.")

    phase_dir = model_dir / format_dataset_size(dataset_size) / f"phase{phase}"
    target = phase_dir / subfolder if subfolder else phase_dir
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_manifest_path(root: Path = None) -> Path:
    root = root or DEFAULT_RESULTS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root / "manifest.jsonl"


# ---------------------------------------------------------------------------
# Baseline dedup
# ---------------------------------------------------------------------------

def baseline_exists(model_name: str, root: Path = None) -> bool:
    return get_results_path(model_name, dataset_size=None, phase=1, root=root).exists()


# ---------------------------------------------------------------------------
# Overwrite guard
# ---------------------------------------------------------------------------

class ExistingRunError(RuntimeError):
    pass


def check_existing_run(model_name: str, dataset_size: int, phase: int, root: Path = None) -> dict:
    """Returns {"exists": bool, "path": str, "has_content": bool}. `has_content`
    is True if scores.json (or, for phase1, the baseline file) already has
    at least one recorded result — that's the signal an accidental re-run
    would clobber real data, not just an empty directory `get_results_path`
    already created."""
    if phase == 1:
        path = get_results_path(model_name, None, 1, root=root)
        exists = path.exists()
        has_content = exists and path.stat().st_size > 0
        return {"exists": exists, "path": str(path), "has_content": has_content}

    phase_dir = get_results_path(model_name, dataset_size, phase, root=root)
    scores_path = phase_dir / "scores.json"
    has_content = scores_path.exists() and scores_path.stat().st_size > 0
    return {"exists": phase_dir.exists(), "path": str(phase_dir), "has_content": has_content}


def require_no_existing_run(model_name: str, dataset_size: int, phase: int,
                             force: bool = False, root: Path = None) -> None:
    """Raise ExistingRunError if this model/dataset_size/phase already has
    results on disk, unless `force=True`. Call this before starting a run
    that would write to that path."""
    info = check_existing_run(model_name, dataset_size, phase, root=root)
    if info["has_content"] and not force:
        raise ExistingRunError(
            f"Results already exist at {info['path']} for model={model_name!r}, "
            f"dataset_size={dataset_size}, phase={phase}. Re-running would overwrite "
            f"previous transcripts/scores. Pass force=True (or --force) to overwrite "
            f"deliberately, or pick a different dataset_size to version this run separately."
        )


# ---------------------------------------------------------------------------
# scores.json — append-only within a phase folder, ALL judge runs kept
# ---------------------------------------------------------------------------

def append_scores_entry(model_name: str, dataset_size: int, phase: int, entry: dict,
                         root: Path = None) -> Path:
    """entry is one {"metadata": {...}, "judge_runs": [...]} record (same shape
    judge.py already produces per transcript). Appended to the phase's shared
    scores.json — read-modify-write is safe here since each phase folder is
    only ever written by one run at a time, unlike the global manifest."""
    phase_dir = get_results_path(model_name, dataset_size, phase, root=root)
    scores_path = phase_dir / "scores.json"

    existing = []
    if scores_path.exists() and scores_path.stat().st_size > 0:
        with open(scores_path, encoding="utf-8") as f:
            existing = json.load(f)

    existing.append(entry)
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    return scores_path


def load_scores(model_name: str, dataset_size: int, phase: int, root: Path = None) -> list[dict]:
    scores_path = get_results_path(model_name, dataset_size, phase, root=root) / "scores.json"
    if not scores_path.exists():
        return []
    with open(scores_path, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(model_name: str, entries: list[dict], root: Path = None) -> Path:
    """entries: list of {"metadata": {...}, "judge_runs": [...]} records
    covering every test_case/condition run in phase 1. Written once — see
    baseline_exists()/require_no_existing_run() for the dedup guard."""
    path = get_results_path(model_name, None, 1, root=root)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "phase": 1, "generated_at": _now_iso(), "results": entries}, f, indent=2)
    return path


def load_baseline(model_name: str, root: Path = None) -> Optional[dict]:
    path = get_results_path(model_name, None, 1, root=root)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Manifest — append-only, one line per completed (model, dataset_size, phase,
# condition) run. Phase-1 baseline entries use dataset_size=null, condition=null.
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_manifest_entry(model_name: str, dataset_size: Optional[int], phase: int,
                          condition: Optional[str], path: Path, num_test_cases: int,
                          status: str = "completed", run_id: Optional[str] = None,
                          thinking_mode: Optional[bool] = None) -> dict:
    return {
        "run_id": run_id or make_run_id(model_name, dataset_size, phase, condition, thinking_mode=thinking_mode),
        "model": model_name,
        "dataset_size": dataset_size,
        "phase": phase,
        "condition": condition,
        "thinking_mode": thinking_mode,  # None for models without a thinking toggle
        "path": str(path),
        "timestamp": _now_iso(),
        "git_commit": get_git_commit(),
        "num_test_cases": num_test_cases,
        "status": status,
    }


def append_manifest_entry(entry: dict, root: Path = None) -> Path:
    """Append-only — never reads-then-rewrites the whole file, so concurrent
    runs can't truncate each other's entries (worst case with true concurrent
    writers is interleaved-but-intact lines, since each write is one line
    with a trailing newline via a single write() call under the GIL/OS)."""
    path = get_manifest_path(root=root)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return path


def load_manifest(root: Path = None) -> list[dict]:
    path = get_manifest_path(root=root)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_manifest_table(entries: list[dict]) -> None:
    columns = ["run_id", "model", "dataset_size", "phase", "condition", "status", "timestamp"]
    widths = {c: max(len(c), *(len(str(e.get(c, ""))) for e in entries)) if entries else len(c) for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    print(header)
    print("  ".join("-" * widths[c] for c in columns))
    for e in entries:
        print("  ".join(str(e.get(c, "")).ljust(widths[c]) for c in columns))


# ---------------------------------------------------------------------------
# Standalone demo / test — uses a temp directory, never touches real results/
# ---------------------------------------------------------------------------

def _demo():
    import shutil
    import tempfile

    tmp_root = Path(tempfile.mkdtemp(prefix="results_manager_demo_"))
    print(f"Using isolated demo root: {tmp_root}\n")

    model = "dummy-model"

    # 1. Baseline doesn't exist yet.
    assert not baseline_exists(model, root=tmp_root)
    print("[ok] baseline does not exist yet")

    fake_baseline_entries = [
        {"metadata": {"test_case": "T1", "condition": "no_prompt"}, "judge_runs": [{"score": "fake"}]},
        {"metadata": {"test_case": "T1", "condition": "system_prompt"}, "judge_runs": [{"score": "fake"}]},
    ]
    baseline_path = save_baseline(model, fake_baseline_entries, root=tmp_root)
    print(f"[ok] wrote baseline to {baseline_path}")

    # 2. Calling baseline generation again should be a no-op / skip signal.
    assert baseline_exists(model, root=tmp_root)
    print("[ok] baseline now exists — a second call should skip re-generating it")

    # 3. Write fake phase2/phase3 results at a couple of dataset sizes.
    for size, phase in [(5, 2), (5, 3), (50, 2)]:
        require_no_existing_run(model, size, phase, root=tmp_root)  # should not raise (nothing there yet)
        entry = {
            "metadata": {"test_case": "T1", "condition": "no_prompt"},
            "judge_runs": [{"crisis_handling": {"score": 3}}, {"crisis_handling": {"score": 4}}],
        }
        scores_path = append_scores_entry(model, size, phase, entry, root=tmp_root)
        manifest_entry = build_manifest_entry(
            model, size, phase, condition="no_prompt", path=scores_path.parent, num_test_cases=1,
        )
        append_manifest_entry(manifest_entry, root=tmp_root)
        print(f"[ok] wrote phase{phase} n={size} results + manifest entry")

    baseline_manifest_entry = build_manifest_entry(
        model, None, 1, condition=None, path=baseline_path, num_test_cases=len(fake_baseline_entries),
    )
    append_manifest_entry(baseline_manifest_entry, root=tmp_root)
    print("[ok] appended baseline manifest entry")

    # 4. Confirm manifest has all 4 entries, appended not overwritten.
    manifest = load_manifest(root=tmp_root)
    assert len(manifest) == 4, f"expected 4 manifest entries, got {len(manifest)}"
    print(f"[ok] manifest has {len(manifest)} entries (appended correctly)\n")

    # 5. Confirm overwrite protection raises without force=True.
    try:
        require_no_existing_run(model, 5, 2, root=tmp_root)
        raise AssertionError("expected ExistingRunError to be raised")
    except ExistingRunError as exc:
        print(f"[ok] re-running model={model} n=5 phase=2 correctly raised: {exc}\n")

    # ...but force=True allows it through.
    require_no_existing_run(model, 5, 2, force=True, root=tmp_root)
    print("[ok] force=True correctly bypassed the guard\n")

    # 6. Print the manifest as a table.
    print("Manifest contents:")
    print_manifest_table(manifest)

    shutil.rmtree(tmp_root)
    print(f"\n[cleanup] removed {tmp_root}")
    print("\nAll results_manager.py checks passed.")


if __name__ == "__main__":
    _demo()
