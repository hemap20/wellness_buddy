"""
Thin CLI orchestrator.

Each stage (data-prep, train, inference, simulate, score, report) is a fully
independent module you can run and inspect on its own — see the __main__
block of each file. This orchestrator is just a convenience wrapper that
calls the same underlying functions in sequence; it contains no logic of
its own beyond "which order to call things in".

Examples:
    python orchestrator.py data-prep
    python orchestrator.py train --phase 2
    python orchestrator.py simulate --test-case T1 --phase phase2 --condition no_prompt
    python orchestrator.py score --transcript transcripts/dialogpt-small_phase2_no_prompt_T1_fixed.json
    python orchestrator.py report

    # Convenience: run every test case x condition for a whole phase, then
    # score and report. Still just calls the modules above in a loop. Results
    # land under results/{model}/n{size}/phase{N}/ (or results/{model}/phase1_baseline.json
    # for phase 1) via results_manager.py, with one line appended per
    # (model, dataset_size, phase, condition) to results/manifest.jsonl.
    python orchestrator.py run-phase --phase 1
    python orchestrator.py run-phase --phase 2 --checkpoint checkpoints/dialogpt-small_phase2/final --dataset-size 50
    python orchestrator.py run-phase --phase 3 --checkpoint checkpoints/dialogpt-small_phase3/final --dataset-size 50

    python orchestrator.py manifest   # print results/manifest.jsonl as a table
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    DataConfig,
    INFERENCE_CONDITIONS,
    LOG_DIR,
    MODELS_UNDER_TEST,
    SIMULATOR_CONFIG,
    SimulatorConfig,
    system_prompt_for_condition,
)
import results_manager as rm


def cmd_data_prep(args):
    import data_prep

    cfg = data_prep.DataConfig()
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    print(json.dumps(data_prep.run(cfg), indent=2))


def cmd_train(args):
    import train

    model_cfg = _resolve_model(args.model, tag=args.tag)
    data_cfg = DataConfig(num_samples=args.num_samples) if args.num_samples is not None else None
    print(json.dumps(train.run(args.phase, model_cfg=model_cfg, data_cfg=data_cfg, force=args.force), indent=2))


def cmd_inference(args):
    from inference import ModelRunner

    model_cfg = _resolve_model(args.model)
    runner = ModelRunner(model_cfg, adapter_path=args.adapter)
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.user_message})
    text, latency = runner.generate(messages)
    print(json.dumps({"response": text, "latency_seconds": latency}, indent=2))


def cmd_simulate(args):
    import simulator
    from inference import ModelRunner

    model_cfg = _resolve_model(args.model)
    sim_cfg = SimulatorConfig(**{**SIMULATOR_CONFIG.__dict__, **({"mode": args.mode} if args.mode else {})})
    test_case = simulator.load_test_case(args.test_case)
    system_prompt = system_prompt_for_condition(args.condition)
    runner = ModelRunner(model_cfg, adapter_path=args.adapter)

    result = simulator.run_conversation(
        runner.generate, test_case, system_prompt, sim_cfg=sim_cfg,
        metadata={"model_name": model_cfg.name, "phase": args.phase, "condition": args.condition, "adapter": args.adapter},
    )
    path = simulator.save_transcript(result)
    print(f"Transcript saved to {path}")
    return path


def cmd_score(args):
    import judge

    record = judge.load_transcript(args.transcript)
    related = None if args.no_related else judge.find_related_transcripts(record)
    runs = judge.score_transcript(record, related_records=related)
    path = judge.save_scores(record, runs)
    print(f"Scores saved to {path}")


def cmd_report(args):
    import report

    print(json.dumps(report.run(args.legacy_scores_dir, args.out), indent=2))


def _resolve_dataset_size(args) -> int:
    if args.dataset_size is not None:
        return args.dataset_size
    return DataConfig().num_samples


def _append_timing(model_name: str, dataset_size: int, step_name: str, duration_seconds: float, status: str) -> None:
    """Appended-only log of how long each run-all step took, per model/dataset
    size — a near-zero duration on a skipped (already-done) step is expected
    and informative on its own, not a bug. Kept separate from manifest.jsonl
    (which is about completed test results, not step wall-clock time)."""
    path = rm.DEFAULT_RESULTS_ROOT / "timings.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model_name, "dataset_size": dataset_size, "step": step_name,
        "duration_seconds": round(duration_seconds, 2), "status": status,
        "timestamp": rm._now_iso(),
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_training_meta(model_name: str, dataset_size: int, phase: int) -> None:
    """Copies the already-written train_summary.json / loss log into this
    run's results/ folder as loss_curve.json + checkpoints_meta.json (step
    numbers + paths, not the weights themselves)."""
    # Must match train.py's naming exactly (namespaced by dataset size so
    # different sample counts never collide on the same checkpoint dir).
    run_name = f"{model_name}_{rm.format_dataset_size(dataset_size)}_phase{phase}"
    summary_path = CHECKPOINT_DIR / run_name / "train_summary.json"
    loss_log_path = LOG_DIR / f"{run_name}_loss.jsonl"
    if not summary_path.exists():
        print(f"[warn] no train_summary.json at {summary_path} — skipping checkpoints_meta.json/loss_curve.json")
        return

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    phase_dir = rm.get_results_path(model_name, dataset_size, phase)
    phase_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = CHECKPOINT_DIR / run_name
    checkpoints_meta = {
        "checkpoint_steps": summary.get("checkpoint_steps", []),
        "checkpoint_paths": {
            str(step): str(next(iter(ckpt_root.glob(f"step_{step}_pct_*")), ""))
            for step in summary.get("checkpoint_steps", [])
        },
        "final_checkpoint": summary.get("final_checkpoint"),
        "total_steps": summary.get("total_steps"),
        # Diagnostic signal from train.py, not something the pipeline acts on
        # automatically — surfaced here so report.py can pick it up per phase.
        "possible_overfitting": summary.get("possible_overfitting", False),
        "val_examples_used": summary.get("val_examples_used", 0),
    }
    with open(phase_dir / "checkpoints_meta.json", "w", encoding="utf-8") as f:
        json.dump(checkpoints_meta, f, indent=2)

    # Prefer train.py's own checkpoint-level curve (train_loss + val_loss,
    # aligned to the same checkpoint steps) — falls back to the raw per-step
    # train-only log for older train_summary.json files that predate val_loss.
    checkpoint_curve = summary.get("checkpoint_curve")
    if checkpoint_curve is not None:
        with open(phase_dir / "loss_curve.json", "w", encoding="utf-8") as f:
            json.dump(checkpoint_curve, f, indent=2)
    elif loss_log_path.exists():
        with open(loss_log_path, encoding="utf-8") as f:
            loss_records = [json.loads(line) for line in f if line.strip()]
        with open(phase_dir / "loss_curve.json", "w", encoding="utf-8") as f:
            json.dump({"steps": [r["step"] for r in loss_records],
                       "train_loss": [r["loss"] for r in loss_records], "val_loss": None}, f, indent=2)


def cmd_run_phase(args):
    import simulator
    import judge
    from inference import ModelRunner

    model_cfg = _resolve_model(args.model, tag=args.tag)
    phase_key = f"phase{args.phase}"
    conditions = INFERENCE_CONDITIONS[phase_key]
    test_case_ids = simulator.list_test_cases()
    modes = ["fixed", "adaptive"] if args.both_modes else [args.mode]

    if args.phase == 1:
        if rm.baseline_exists(model_cfg.name) and not args.force:
            path = rm.get_results_path(model_cfg.name, None, 1)
            print(f"Baseline for {model_cfg.name} already exists at {path}, skipping. Pass --force to regenerate.")
            return
        dataset_size = None
    else:
        dataset_size = _resolve_dataset_size(args)
        # Skip (don't crash) if this exact (model, dataset_size, phase) already
        # has results — same "already done" semantics as the phase-1 baseline
        # check above. This is what makes `run-full`/`run-batch`/`run-all`
        # resume-safe: stopping mid-run and re-running the same command later
        # skips every already-completed phase instead of erroring out.
        existing = rm.check_existing_run(model_cfg.name, dataset_size, args.phase)
        if existing["has_content"] and not args.force:
            print(f"Results for {model_cfg.name} n{dataset_size} phase{args.phase} already exist "
                  f"at {existing['path']}, skipping. Pass --force to regenerate.")
            return
        _write_training_meta(model_cfg.name, dataset_size, args.phase)

    runner = ModelRunner(model_cfg, adapter_path=args.checkpoint)

    baseline_entries = []  # only populated for phase 1
    failed_runs = []  # (test_case/condition/mode) labels that raised — see except block below

    for condition in conditions:
        for mode in modes:
                sim_cfg = SimulatorConfig(**{**SIMULATOR_CONFIG.__dict__, "mode": mode})
                generate_fn = runner.generate

                num_succeeded = 0
                for test_case_id in test_case_ids:
                    label = f"{test_case_id} / {condition} / {mode}"
                    try:
                        test_case = simulator.load_test_case(test_case_id)
                        system_prompt = system_prompt_for_condition(condition)

                        result = simulator.run_conversation(
                            generate_fn, test_case, system_prompt, sim_cfg=sim_cfg,
                            metadata={
                                "model_name": model_cfg.name, "phase": phase_key,
                                "condition": condition,
                                "adapter": args.checkpoint,
                            },
                        )

                        related = None
                        if args.phase != 1:
                            transcripts_dir = rm.get_results_path(
                                model_cfg.name, dataset_size, args.phase, "transcripts"
                            )
                            related = []
                            for sibling_path in transcripts_dir.glob(f"{test_case_id}_*.json"):
                                with open(sibling_path, encoding="utf-8") as f:
                                    sibling = json.load(f)
                                sibling_meta = sibling["metadata"]
                                if sibling_meta.get("condition") != condition:
                                    related.append(sibling)
                        judge_runs = judge.score_transcript(result, related_records=related)

                        if args.phase == 1:
                            baseline_entries.append({
                                "metadata": result["metadata"], "transcript": result["transcript"],
                                "judge_runs": judge_runs,
                            })
                        else:
                            phase_dir = rm.get_results_path(model_cfg.name, dataset_size, args.phase)
                            transcripts_dir = phase_dir / "transcripts"
                            transcripts_dir.mkdir(parents=True, exist_ok=True)
                            transcript_path = transcripts_dir / f"{test_case_id}_{condition}_{mode}.json"
                            with open(transcript_path, "w", encoding="utf-8") as f:
                                json.dump(result, f, indent=2)
                            rm.append_scores_entry(
                                model_cfg.name, dataset_size, args.phase,
                                {"metadata": result["metadata"], "judge_runs": judge_runs},
                            )

                        num_succeeded += 1
                        print(f"[{label}] done")
                    except Exception as exc:
                        # One bad conversation (simulator crash, judge exhausting
                        # retries, etc.) must not cost every other test case's
                        # already-spent API calls — log and move on. Failed
                        # combos are simply absent from scores.json/manifest;
                        # re-run this specific (phase, condition)
                        # later with --force once the underlying issue is fixed.
                        failed_runs.append(label)
                        print(f"[{label}] FAILED: {type(exc).__name__}: {exc}")

                if args.phase == 1 and baseline_entries:
                    # Checkpoint incrementally — baseline_entries otherwise only
                    # hits disk once, at the very end of the whole loop, so a
                    # crash anywhere in a long baseline run would lose every
                    # already-completed conversation. save_baseline() is a full
                    # overwrite of the one baseline file, so re-calling it here
                    # with everything accumulated so far is safe and idempotent.
                    rm.save_baseline(model_cfg.name, baseline_entries)

                # num_succeeded (not len(test_case_ids)) — a manifest entry
                # must reflect what actually landed in scores.json/baseline,
                # not what was merely attempted. Otherwise a fully- or
                # partially-failed group (e.g. every test case hitting a
                # ConnectError) still gets recorded as if it succeeded,
                # and a later "already exists" check treats corrupt/empty
                # results as done — this bit us three times in one run
                # before being caught here.
                if args.phase == 1:
                    path_for_manifest = rm.get_results_path(model_cfg.name, None, 1)
                else:
                    path_for_manifest = rm.get_results_path(model_cfg.name, dataset_size, args.phase)
                entry = rm.build_manifest_entry(
                    model_cfg.name, dataset_size, args.phase, condition, path_for_manifest, num_succeeded,
                )
                rm.append_manifest_entry(entry)
                print(f"[manifest] appended {entry['run_id']} ({num_succeeded}/{len(test_case_ids)} succeeded)")

    if args.phase == 1:
        baseline_path = rm.save_baseline(model_cfg.name, baseline_entries)
        print(f"Baseline saved to {baseline_path}")

    if failed_runs:
        print(f"\n{len(failed_runs)} combo(s) FAILED and were skipped (not in scores.json/manifest):")
        for label in failed_runs:
            print(f"  - {label}")
        print("Re-run this same run-phase call (with --force if data already exists) to retry just these.")

    import report

    print(json.dumps(report.run(), indent=2))

    if failed_runs:
        # Raised (not just printed) so callers like cmd_run_all's run_step
        # correctly mark this step FAILED instead of OK — a step with even
        # one dropped test case must not be indistinguishable from a clean
        # run to anything reading the summary/manifest.
        raise RuntimeError(
            f"{len(failed_runs)} test case(s) failed for {model_cfg.name} phase{args.phase} "
            f"— see the FAILED lines above. Re-run with --force once the underlying issue is fixed."
        )


def cmd_run_all(args):
    """Single-command full pipeline for one model: phase1 baseline -> train
    phase2 -> test phase2 -> train phase3 -> test phase3. Meant to be launched
    once (e.g. in the background) and left running unattended — each step is
    isolated in its own try/except so one failure doesn't abort the rest, and
    a clear pass/fail summary prints at the end."""
    import train

    model_cfg = _resolve_model(args.model, tag=args.tag)
    dataset_size = args.num_samples if args.num_samples is not None else DataConfig().num_samples

    def make_run_phase_args(phase, checkpoint=None):
        # Pass the base (untagged) model name + tag separately, not
        # model_cfg.name — cmd_run_phase's own _resolve_model looks up the
        # name in MODELS_UNDER_TEST and would fail to find "model-v2".
        return argparse.Namespace(
            phase=phase, model=args.model, checkpoint=checkpoint, mode=args.mode,
            both_modes=args.both_modes, dataset_size=None if phase == 1 else dataset_size,
            force=args.force, tag=args.tag,
        )

    steps_ok = {}

    def run_step(step_name, fn):
        print(f"[run-all] starting {step_name}")
        start = time.time()
        try:
            fn()
            steps_ok[step_name] = True
            status = "ok"
        except Exception as exc:
            steps_ok[step_name] = False
            status = f"FAILED: {type(exc).__name__}: {exc}"
        duration = time.time() - start
        # A skipped step (already-done check firing) returns near-instantly;
        # logging it just clutters timings.jsonl with meaningless ~0s rows.
        # Only genuine work (real training/testing, always seconds-to-hours)
        # is worth timing.
        if duration >= 2.0:
            _append_timing(model_cfg.name, dataset_size, step_name, duration, status)
        if steps_ok[step_name]:
            print(f"[run-all] finished {step_name} ({duration:.1f}s)")
        else:
            print(f"[run-all] {status} ({step_name}, {duration:.1f}s)")

    run_step("phase1_baseline", lambda: cmd_run_phase(make_run_phase_args(1)))

    ckpt2 = None
    def _train_phase2():
        nonlocal ckpt2
        summary = train.run(2, model_cfg=model_cfg, data_cfg=DataConfig(num_samples=dataset_size), force=args.force)
        ckpt2 = summary["final_checkpoint"]
    run_step("train_phase2", _train_phase2)

    if ckpt2:
        run_step("test_phase2", lambda: cmd_run_phase(make_run_phase_args(2, checkpoint=ckpt2)))
    else:
        steps_ok["test_phase2"] = False
        print("[run-all] SKIPPED test_phase2 (no checkpoint — train_phase2 failed)")

    ckpt3 = None
    def _train_phase3():
        nonlocal ckpt3
        summary = train.run(3, model_cfg=model_cfg, data_cfg=DataConfig(num_samples=dataset_size), force=args.force)
        ckpt3 = summary["final_checkpoint"]
    run_step("train_phase3", _train_phase3)

    if ckpt3:
        run_step("test_phase3", lambda: cmd_run_phase(make_run_phase_args(3, checkpoint=ckpt3)))
    else:
        steps_ok["test_phase3"] = False
        print("[run-all] SKIPPED test_phase3 (no checkpoint — train_phase3 failed)")

    print(f"\n[run-all] summary for model={model_cfg.name}, dataset_size={dataset_size}:")
    for step_name, ok in steps_ok.items():
        print(f"  {'OK  ' if ok else 'FAIL'}  {step_name}")
    if not all(steps_ok.values()):
        raise SystemExit(1)


def cmd_run_batch(args):
    """Convenience: runs cmd_run_all sequentially for a list of models, one
    after another (never in parallel — everything shares one GPU/unified
    memory pool). One bad model must not abort the rest, so each model's
    run-all failure is caught and logged; the batch always finishes."""
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_names:
        raise SystemExit("--models must be a non-empty comma-separated list")

    batch_results = {}
    for model_name in model_names:
        print(f"\n[run-batch] ===== starting model={model_name} =====")
        sub_args = argparse.Namespace(
            model=model_name, num_samples=args.num_samples, tag=args.tag,
            mode=args.mode, both_modes=args.both_modes, force=args.force,
        )
        try:
            cmd_run_all(sub_args)
            batch_results[model_name] = "OK"
        except SystemExit as exc:
            # cmd_run_all raises SystemExit(1) when one of its own steps
            # failed — that's already logged step-by-step by cmd_run_all;
            # here we just record it and move on to the next model.
            batch_results[model_name] = "PARTIAL_FAILURE" if exc.code else "OK"
        except Exception as exc:
            batch_results[model_name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"[run-batch] model={model_name} FAILED: {type(exc).__name__}: {exc}")
        print(f"[run-batch] ===== finished model={model_name} =====")

    print(f"\n[run-batch] FINAL SUMMARY (num_samples={args.num_samples}, tag={args.tag}):")
    for model_name, status in batch_results.items():
        print(f"  {status:16s} {model_name}")


def cmd_run_phase1_batch(model_names, tag):
    """Runs (or skips, if already done) phase-1 baseline for every listed
    model, one after another. Baseline is dataset-size-independent and needs
    no training, so this is fast — pulling it into its own pass up front
    means every model's zero-shot reference point is available quickly,
    instead of model N's baseline being stuck behind model 1..N-1's full
    training+testing cycle finishing first."""
    for model_name in model_names:
        print(f"\n[run-full] ----- phase1 baseline: {model_name} -----")
        args = argparse.Namespace(
            phase=1, model=model_name, checkpoint=None, mode="adaptive",
            both_modes=False, dataset_size=None, force=False, tag=tag,
        )
        try:
            cmd_run_phase(args)
        except Exception as exc:
            print(f"[run-full] phase1 baseline FAILED for {model_name}: {type(exc).__name__}: {exc}")


def cmd_run_full(args):
    """The single unattended command: data-prep for each listed dataset size
    (skipped if already present), then phase-1 baseline for EVERY listed
    model up front (so no model's baseline waits behind another model's full
    training cycle), then run-batch at the first size for every listed
    model, then the next size, etc. Sizes always run in the given order, and
    ALL models finish a given size before the next size starts."""
    import data_prep

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    if not sizes:
        raise SystemExit("--sizes must be a non-empty comma-separated list")

    for num_samples in sizes:
        cfg = data_prep.DataConfig(num_samples=num_samples)
        if cfg.raw_path().exists():
            print(f"[run-full] data/n{num_samples:03d} already exists, skipping data-prep")
        else:
            print(f"[run-full] running data-prep for num_samples={num_samples}")
            data_prep.run(cfg)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    cmd_run_phase1_batch(model_names, args.tag)

    for num_samples in sizes:
        print(f"\n[run-full] ########## num_samples={num_samples} ##########")
        batch_args = argparse.Namespace(
            models=args.models, num_samples=num_samples, tag=args.tag,
            mode=args.mode, both_modes=args.both_modes, force=args.force,
        )
        cmd_run_batch(batch_args)

    if not args.no_charts:
        import generate_charts

        print(f"\n[run-full] ########## generating charts ##########")
        generate_charts.run(dataset_sizes=sizes)


def cmd_manifest(args):
    entries = rm.load_manifest()
    if not entries:
        print("Manifest is empty — no runs recorded yet.")
        return
    rm.print_manifest_table(entries)


def _resolve_model(name, tag=None):
    if not name:
        model_cfg = MODELS_UNDER_TEST[0]
    else:
        matches = [m for m in MODELS_UNDER_TEST if m.name == name]
        if not matches:
            raise SystemExit(f"Unknown model name: {name}. Known: {[m.name for m in MODELS_UNDER_TEST]}")
        model_cfg = matches[0]
    if tag:
        # Renaming only affects checkpoint/results paths (both keyed off
        # model_cfg.name) — hf_model_id/lora_target_modules/etc. are
        # untouched, so a "-v2" retrain never collides with the original run.
        model_cfg = dataclasses.replace(model_cfg, name=f"{model_cfg.name}-{tag}")
    return model_cfg


def build_parser():
    parser = argparse.ArgumentParser(description="Wellness-buddy eval pipeline orchestrator.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data-prep")
    p.add_argument("--num-samples", type=int, default=None)
    p.set_defaults(func=cmd_data_prep)

    p = sub.add_parser("train")
    p.add_argument("--phase", type=int, required=True, choices=[2, 3])
    p.add_argument("--model", default=None)
    p.add_argument("--num-samples", type=int, default=None,
                    help="Must match what you passed to data_prep.py --num-samples.")
    p.add_argument("--tag", default=None, help="Appends '-{tag}' to the model name for checkpoint naming.")
    p.add_argument("--force", action="store_true", help="Redo training even if a completed checkpoint already exists.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("inference")
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--system-prompt", default=None)
    p.add_argument("--user-message", default="I've had a really rough day.")
    p.set_defaults(func=cmd_inference)

    p = sub.add_parser("simulate")
    p.add_argument("--test-case", required=True)
    p.add_argument("--phase", required=True, choices=["phase1", "phase2", "phase3"])
    p.add_argument("--condition", required=True, choices=["no_prompt", "system_prompt", "matched", "paraphrased"])
    p.add_argument("--model", default=None)
    p.add_argument("--adapter", default=None)
    p.add_argument("--mode", default=None, choices=["adaptive", "fixed"])
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("score")
    p.add_argument("--transcript", required=True)
    p.add_argument("--no-related", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("report")
    p.add_argument("--legacy-scores-dir", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run-phase", help="Convenience: simulate+score+report across all test cases/conditions for one phase.")
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default=None)
    p.add_argument("--checkpoint", default=None, help="LoRA adapter path (phase 2/3 only; omit for phase 1 baseline)")
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--both-modes", action="store_true", help="Run both fixed and adaptive simulator modes")
    p.add_argument("--dataset-size", type=int, default=None,
                    help="Overrides DataConfig.num_samples for labeling results/{model}/n{size}/ — "
                         "use this if you ran data_prep.py with a different --num-samples than the config default.")
    p.add_argument("--force", action="store_true",
                    help="Overwrite an existing phase result (or regenerate an existing baseline) instead of skipping/erroring.")
    p.add_argument("--tag", default=None, help="Appends '-{tag}' to the model name for results naming.")
    p.set_defaults(func=cmd_run_phase)

    p = sub.add_parser("manifest", help="Print results/manifest.jsonl as a table.")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser(
        "run-all",
        help="Convenience: run phase 1 baseline, then train+test phase 2, then train+test phase 3 "
             "for one model in a single command — meant to be launched once and left running.",
    )
    p.add_argument("--model", default=None)
    p.add_argument("--num-samples", type=int, default=None,
                    help="Used for both data_prep-derived phase2/3 datasets and results/checkpoint naming. "
                         "Must already have been produced by data_prep.py --num-samples.")
    p.add_argument("--tag", default=None, help="Appends '-{tag}' to the model name everywhere (checkpoints + results).")
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--both-modes", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser(
        "run-batch",
        help="Convenience: run-all for each of several models, one after another, at one dataset size.",
    )
    p.add_argument("--models", required=True, help="Comma-separated model names, e.g. dialogpt-small,gemma-4-e2b-it")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--both-modes", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_run_batch)

    p = sub.add_parser(
        "run-full",
        help="Convenience: run-batch at each --sizes entry, in order, for every listed model, "
             "then generate charts. This is the single command meant to be launched once and "
             "left running unattended, covering phase1 -> train2 -> test2 -> train3 -> test3 "
             "-> report -> charts for every model at every dataset size.",
    )
    p.add_argument("--models", required=True, help="Comma-separated model names.")
    p.add_argument("--sizes", default="50,500",
                    help="Comma-separated dataset sizes, run in order (all models finish size N "
                         "before size N+1 starts). Default '50,500'.")
    p.add_argument("--tag", default=None)
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--both-modes", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation at the end.")
    p.set_defaults(func=cmd_run_full)

    p = sub.add_parser(
        "run-phase1-batch",
        help="Convenience: phase-1 baseline only, for every listed model, one after another. "
             "No training involved — fast. Useful to get every model's zero-shot reference point "
             "before deciding where/how to run the (much longer) phase 2/3 training.",
    )
    p.add_argument("--models", required=True, help="Comma-separated model names.")
    p.add_argument("--tag", default=None)
    p.set_defaults(func=lambda args: cmd_run_phase1_batch(
        [m.strip() for m in args.models.split(",") if m.strip()], args.tag
    ))

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
