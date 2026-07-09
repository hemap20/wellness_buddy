"""
LLM-as-user-simulator module — standalone runnable stage.

Drives a multi-turn conversation against an arbitrary "model under test"
generate function, using a different model (Claude) to play the user.

Two modes, toggled by SimulatorConfig.mode:
  - "fixed":    sends the test case's turn_beats verbatim, in order, ignoring
                the bot's actual replies. Strictly comparable across models/phases.
  - "adaptive": Claude decides the next user message given the persona, the
                turn_beats as a loose script, and the transcript so far.

Logs the full transcript + metadata to a JSON file under transcripts/.

Run standalone (against the local model, phase 2, no system prompt):
    python simulator.py --test-case T1 --phase phase2 --condition no_prompt --model dialogpt-small
"""
import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

from config import (
    SIMULATOR_CONFIG,
    TEST_CASE_DIR,
    TRANSCRIPT_DIR,
    SimulatorConfig,
    system_prompt_for_condition,
)
from llm_client import LLMClient

_SIMULATOR_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_message": {"type": "string", "description": "The next message the simulated user sends."},
        "end_conversation": {"type": "boolean", "description": "True if the scenario is naturally complete."},
    },
    "required": ["user_message", "end_conversation"],
    "additionalProperties": False,
}


def load_test_case(test_case_id: str) -> dict:
    matches = list(TEST_CASE_DIR.glob(f"{test_case_id}_*.json"))
    if not matches:
        raise FileNotFoundError(f"No test case file found for id '{test_case_id}' in {TEST_CASE_DIR}")
    with open(matches[0], encoding="utf-8") as f:
        return json.load(f)


def list_test_cases() -> list[str]:
    return sorted(p.stem.split("_")[0] for p in TEST_CASE_DIR.glob("*.json"))


def _simulator_system_prompt(test_case: dict) -> str:
    beats = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(test_case["turn_beats"]))
    return (
        f"{test_case['persona']}\n\n"
        "You are role-playing this persona in a conversation with a wellness chatbot. "
        "Here are the loose turn beats to follow, adapting your exact wording to what "
        "the chatbot actually says (don't just recite them verbatim; react naturally):\n"
        f"{beats}\n\n"
        "Stay fully in character as the human user. Never break character, never mention "
        "that you are an AI or a simulator. Produce only your next message to the chatbot."
    )


def _next_adaptive_user_message(client: LLMClient, sim_cfg: SimulatorConfig,
                                 test_case: dict, transcript: list[dict], turn_idx: int) -> tuple[str, bool]:
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in transcript) or "(conversation not yet started)"
    user_prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"You are on turn beat {min(turn_idx + 1, len(test_case['turn_beats']))} of "
        f"{len(test_case['turn_beats'])}. Produce your next message now."
    )
    text = client.generate_json(
        system=_simulator_system_prompt(test_case),
        user_content=user_prompt,
        schema=_SIMULATOR_OUTPUT_SCHEMA,
        max_tokens=300,
    )
    data = json.loads(text)
    return data["user_message"], data["end_conversation"]


def run_conversation(
    generate_fn: Callable[[list[dict]], tuple[str, float]],
    test_case: dict,
    system_prompt: Optional[str],
    sim_cfg: SimulatorConfig = None,
    metadata: Optional[dict] = None,
) -> dict:
    """generate_fn(messages) -> (assistant_text, latency_seconds). `messages`
    passed to generate_fn always starts with the optional system message
    (per `system_prompt`) followed by the alternating conversation."""
    sim_cfg = sim_cfg or SIMULATOR_CONFIG
    client = LLMClient(sim_cfg.provider, sim_cfg.model)

    bot_messages: list[dict] = []
    if system_prompt:
        bot_messages.append({"role": "system", "content": system_prompt})

    transcript: list[dict] = []
    max_turns = min(sim_cfg.max_turns, len(test_case["turn_beats"])) if sim_cfg.mode == "fixed" else sim_cfg.max_turns

    for turn_idx in range(max_turns):
        if sim_cfg.mode == "fixed":
            if turn_idx >= len(test_case["turn_beats"]):
                break
            user_text = test_case["turn_beats"][turn_idx]
            end_conversation = False
        else:
            user_text, end_conversation = _next_adaptive_user_message(client, sim_cfg, test_case, transcript, turn_idx)

        transcript.append({"role": "user", "content": user_text})
        bot_messages.append({"role": "user", "content": user_text})

        assistant_text, latency = generate_fn(bot_messages)
        transcript.append({"role": "assistant", "content": assistant_text, "latency_seconds": latency})
        bot_messages.append({"role": "assistant", "content": assistant_text})

        if end_conversation:
            break

    result = {
        "metadata": {
            "test_case": test_case["id"],
            "test_case_name": test_case["name"],
            "simulator_mode": sim_cfg.mode,
            "simulator_model": sim_cfg.model,
            "system_prompt": system_prompt,
            "timestamp_run_at_epoch": time.time(),
            **(metadata or {}),
        },
        "transcript": transcript,
    }
    return result


def save_transcript(result: dict) -> Path:
    meta = result["metadata"]
    fname = "_".join(
        str(meta.get(k, "na")) for k in ("model_name", "phase", "condition", "test_case", "simulator_mode")
    )
    path = TRANSCRIPT_DIR / f"{fname}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Simulator stage: run a standalone transcript.")
    parser.add_argument("--test-case", required=True, help="e.g. T1")
    parser.add_argument("--phase", required=True, choices=["phase1", "phase2", "phase3"])
    parser.add_argument("--condition", required=True, choices=["no_prompt", "system_prompt", "matched", "paraphrased"])
    parser.add_argument("--model", default=None, help="Model name from MODELS_UNDER_TEST")
    parser.add_argument("--adapter", default=None, help="Path to a LoRA adapter checkpoint")
    parser.add_argument("--mode", default=None, choices=["adaptive", "fixed"])
    args = parser.parse_args()

    from config import MODELS_UNDER_TEST
    from inference import ModelRunner

    model_cfg = MODELS_UNDER_TEST[0]
    if args.model:
        matches = [m for m in MODELS_UNDER_TEST if m.name == args.model]
        if not matches:
            raise SystemExit(f"Unknown model: {args.model}")
        model_cfg = matches[0]

    sim_cfg = SimulatorConfig(**{**SIMULATOR_CONFIG.__dict__, **({"mode": args.mode} if args.mode else {})})

    test_case = load_test_case(args.test_case)
    system_prompt = system_prompt_for_condition(args.condition)

    runner = ModelRunner(model_cfg, adapter_path=args.adapter)

    result = run_conversation(
        runner.generate,
        test_case,
        system_prompt,
        sim_cfg=sim_cfg,
        metadata={"model_name": model_cfg.name, "phase": args.phase, "condition": args.condition, "adapter": args.adapter},
    )
    path = save_transcript(result)
    print(f"Transcript saved to {path}")
    print(json.dumps(result, indent=2)[:2000])


if __name__ == "__main__":
    main()
