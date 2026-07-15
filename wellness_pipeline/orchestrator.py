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
import json
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

    model_cfg = _resolve_model(args.model)
    data_cfg = DataConfig(num_samples=args.num_samples) if args.num_samples is not None else None
    print(json.dumps(train.run(args.phase, model_cfg=model_cfg, data_cfg=data_cfg), indent=2))


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


def _write_training_meta(model_name: str, dataset_size: int, phase: int) -> None:
    """Copies the already-written train_summary.json / loss log into this
    run's results/ folder as loss_curve.json + checkpoints_meta.json (step
    numbers + paths, not the weights themselves)."""
    run_name = f"{model_name}_phase{phase}"
    summary_path = CHECKPOINT_DIR / run_name / "train_summary.json"
    loss_log_path = LOG_DIR / f"{run_name}_loss.jsonl"
    if not summary_path.exists():
        print(f"[warn] no train_summary.json at {summary_path} — skipping checkpoints_meta.json/loss_curve.json")
        return

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    phase_dir = rm.get_results_path(model_name, dataset_size, phase)

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

    model_cfg = _resolve_model(args.model)
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
        try:
            rm.require_no_existing_run(model_cfg.name, dataset_size, args.phase, force=args.force)
        except rm.ExistingRunError as exc:
            raise SystemExit(str(exc))
        _write_training_meta(model_cfg.name, dataset_size, args.phase)

    runner = ModelRunner(model_cfg, adapter_path=args.checkpoint)

    baseline_entries = []  # only populated for phase 1
    failed_runs = []  # (test_case/condition/mode/thinking) labels that raised — see except block below
    # Models without a thinking toggle just get a single None pass (unchanged
    # behavior); models with one run every condition/mode/test_case twice.
    thinking_modes = [True, False] if model_cfg.supports_thinking else [None]

    for condition in conditions:
        for thinking_mode in thinking_modes:
            for mode in modes:
                sim_cfg = SimulatorConfig(**{**SIMULATOR_CONFIG.__dict__, "mode": mode})
                # Bind thinking_mode now (default-arg trick) so each closure
                # captures its own value rather than the loop variable.
                generate_fn = (lambda messages, _t=thinking_mode: runner.generate(messages, thinking=_t))

                for test_case_id in test_case_ids:
                    label = (f"{test_case_id} / {condition} / {mode}"
                             f"{'' if thinking_mode is None else f' / thinking={thinking_mode}'}")
                    try:
                        test_case = simulator.load_test_case(test_case_id)
                        system_prompt = system_prompt_for_condition(condition)

                        result = simulator.run_conversation(
                            generate_fn, test_case, system_prompt, sim_cfg=sim_cfg,
                            metadata={
                                "model_name": model_cfg.name, "phase": phase_key,
                                "condition": condition, "thinking_mode": thinking_mode,
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
                                # Only compare against siblings under the SAME
                                # thinking mode — system_prompt_dependency is about
                                # prompt-condition consistency, not thinking mode.
                                if (sibling_meta.get("condition") != condition
                                        and sibling_meta.get("thinking_mode") == thinking_mode):
                                    related.append(sibling)
                        judge_runs = judge.score_transcript(result, related_records=related)

                        if args.phase == 1:
                            baseline_entries.append({
                                "metadata": result["metadata"], "transcript": result["transcript"],
                                "judge_runs": judge_runs,
                            })
                        else:
                            phase_dir = rm.get_results_path(model_cfg.name, dataset_size, args.phase)
                            thinking_suffix = "" if thinking_mode is None else (
                                "_thinking_on" if thinking_mode else "_thinking_off"
                            )
                            transcript_path = (
                                phase_dir / "transcripts" / f"{test_case_id}_{condition}_{mode}{thinking_suffix}.json"
                            )
                            with open(transcript_path, "w", encoding="utf-8") as f:
                                json.dump(result, f, indent=2)
                            rm.append_scores_entry(
                                model_cfg.name, dataset_size, args.phase,
                                {"metadata": result["metadata"], "judge_runs": judge_runs},
                            )

                        print(f"[{label}] done")
                    except Exception as exc:
                        # One bad conversation (simulator crash, judge exhausting
                        # retries, etc.) must not cost every other test case's
                        # already-spent API calls — log and move on. Failed
                        # combos are simply absent from scores.json/manifest;
                        # re-run this specific (phase, condition, thinking_mode)
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

                num_test_cases = len(test_case_ids)
                if args.phase == 1:
                    path_for_manifest = rm.get_results_path(model_cfg.name, None, 1)
                else:
                    path_for_manifest = rm.get_results_path(model_cfg.name, dataset_size, args.phase)
                entry = rm.build_manifest_entry(
                    model_cfg.name, dataset_size, args.phase, condition, path_for_manifest, num_test_cases,
                    thinking_mode=thinking_mode,
                )
                rm.append_manifest_entry(entry)
                print(f"[manifest] appended {entry['run_id']}")

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


def cmd_manifest(args):
    entries = rm.load_manifest()
    if not entries:
        print("Manifest is empty — no runs recorded yet.")
        return
    rm.print_manifest_table(entries)


def _resolve_model(name):
    if not name:
        return MODELS_UNDER_TEST[0]
    matches = [m for m in MODELS_UNDER_TEST if m.name == name]
    if not matches:
        raise SystemExit(f"Unknown model name: {name}. Known: {[m.name for m in MODELS_UNDER_TEST]}")
    return matches[0]


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
    p.set_defaults(func=cmd_run_phase)

    p = sub.add_parser("manifest", help="Print results/manifest.jsonl as a table.")
    p.set_defaults(func=cmd_manifest)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
