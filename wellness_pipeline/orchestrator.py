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
    # score and report. Still just calls the modules above in a loop.
    python orchestrator.py run-phase --phase 1
    python orchestrator.py run-phase --phase 2 --checkpoint checkpoints/dialogpt-small_phase2/final
    python orchestrator.py run-phase --phase 3 --checkpoint checkpoints/dialogpt-small_phase3/final
"""
import argparse
import json

from config import (
    CHECKPOINT_DIR,
    INFERENCE_CONDITIONS,
    MODELS_UNDER_TEST,
    SIMULATOR_CONFIG,
    SimulatorConfig,
    system_prompt_for_condition,
)


def cmd_data_prep(args):
    import data_prep

    cfg = data_prep.DataConfig()
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    print(json.dumps(data_prep.run(cfg), indent=2))


def cmd_train(args):
    import train

    model_cfg = _resolve_model(args.model)
    print(json.dumps(train.run(args.phase, model_cfg=model_cfg), indent=2))


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

    print(json.dumps(report.run(args.scores_dir, args.out), indent=2))


def cmd_run_phase(args):
    import simulator
    import judge

    model_cfg = _resolve_model(args.model)
    phase_key = f"phase{args.phase}"
    conditions = INFERENCE_CONDITIONS[phase_key]
    test_case_ids = simulator.list_test_cases()

    transcript_paths = []
    for test_case_id in test_case_ids:
        for condition in conditions:
            for mode in (["fixed", "adaptive"] if args.both_modes else [args.mode]):
                sim_cfg = SimulatorConfig(**{**SIMULATOR_CONFIG.__dict__, "mode": mode})
                test_case = simulator.load_test_case(test_case_id)
                system_prompt = system_prompt_for_condition(condition)
                from inference import ModelRunner

                runner = ModelRunner(model_cfg, adapter_path=args.checkpoint)
                result = simulator.run_conversation(
                    runner.generate, test_case, system_prompt, sim_cfg=sim_cfg,
                    metadata={
                        "model_name": model_cfg.name, "phase": phase_key,
                        "condition": condition, "adapter": args.checkpoint,
                    },
                )
                path = simulator.save_transcript(result)
                transcript_paths.append(path)
                print(f"[simulate] {test_case_id} / {condition} / {mode} -> {path}")

    for path in transcript_paths:
        record = judge.load_transcript(str(path))
        related = judge.find_related_transcripts(record)
        runs = judge.score_transcript(record, related_records=related)
        score_path = judge.save_scores(record, runs)
        print(f"[score] {path} -> {score_path}")

    import report

    print(json.dumps(report.run(), indent=2))


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
    p.add_argument("--scores-dir", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("run-phase", help="Convenience: simulate+score+report across all test cases/conditions for one phase.")
    p.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--model", default=None)
    p.add_argument("--checkpoint", default=None, help="LoRA adapter path (phase 2/3 only; omit for phase 1 baseline)")
    p.add_argument("--mode", default="adaptive", choices=["adaptive", "fixed"])
    p.add_argument("--both-modes", action="store_true", help="Run both fixed and adaptive simulator modes")
    p.set_defaults(func=cmd_run_phase)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
