"""
Data-prep module — standalone runnable stage.

Downloads N raw samples from the configured HF dataset, normalizes whatever
schema it has into {user, assistant} pairs, and writes two variants:
  - phase2_pairs.jsonl: single-turn {user, assistant} pairs, no system role
  - phase3_pairs.jsonl: same pairs, with the fixed system prompt prepended

Fails loudly (raises) on malformed data rather than silently training on it.

Run standalone:
    python data_prep.py
    python data_prep.py --num-samples 5
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from config import DataConfig, FIXED_SYSTEM_PROMPT


class DataValidationError(ValueError):
    pass


def _clean_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise DataValidationError(f"{field_name} is not a string: {type(value)!r}")
    text = value.strip()
    if not text:
        raise DataValidationError(f"{field_name} is empty after stripping whitespace")
    try:
        text.encode("utf-8").decode("utf-8")
    except UnicodeError as exc:
        raise DataValidationError(f"{field_name} has an encoding issue: {exc}")
    return text


def _extract_pairs_from_conversation(turns: list) -> list[tuple[str, str]]:
    """Given a list of {role/from, content/value} turns, return every
    adjacent (user, assistant) exchange as an independent single-turn pair.
    Any system turns are stripped; any turns with unrecognized roles fail
    loudly rather than being silently dropped."""
    normalized = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise DataValidationError(f"conversation turn is not an object: {turn!r}")
        role_raw = turn.get("role", turn.get("from"))
        content_raw = turn.get("content", turn.get("value"))
        if role_raw is None or content_raw is None:
            raise DataValidationError(f"conversation turn missing role/content: {turn!r}")
        role = str(role_raw).strip().lower()
        role = {"human": "user", "gpt": "assistant", "bot": "assistant", "ai": "assistant"}.get(role, role)
        if role == "system":
            continue
        if role not in ("user", "assistant"):
            raise DataValidationError(f"malformed role '{role_raw}' in conversation turn: {turn!r}")
        normalized.append((role, _clean_text(content_raw, f"conversation.{role}.content")))

    pairs = []
    for i in range(len(normalized) - 1):
        role_a, text_a = normalized[i]
        role_b, text_b = normalized[i + 1]
        if role_a == "user" and role_b == "assistant":
            pairs.append((text_a, text_b))
    if not pairs:
        raise DataValidationError(f"no user->assistant exchange found in conversation: {turns!r}")
    return pairs


# Field-name patterns we try, in order, to normalize an unknown HF schema
# into a list of (user, assistant) pairs. Add new patterns here rather than
# writing ad-hoc parsing elsewhere if the source schema differs.
_CONVERSATION_KEYS = ("conversations", "messages", "dialogue", "turns")
_DIRECT_PAIR_KEYS = (
    ("input", "output"),
    ("prompt", "response"),
    ("instruction", "response"),
    ("question", "answer"),
    ("user", "assistant"),
)


def normalize_example(example: dict) -> list[tuple[str, str]]:
    for key in _CONVERSATION_KEYS:
        if key in example and isinstance(example[key], list) and example[key]:
            return _extract_pairs_from_conversation(example[key])

    for user_key, assistant_key in _DIRECT_PAIR_KEYS:
        if user_key in example and assistant_key in example:
            return [(
                _clean_text(example[user_key], user_key),
                _clean_text(example[assistant_key], assistant_key),
            )]

    raise DataValidationError(
        f"Could not recognize schema for example with keys {list(example.keys())}. "
        f"Extend _CONVERSATION_KEYS / _DIRECT_PAIR_KEYS in data_prep.py."
    )


def load_raw_samples(cfg: DataConfig) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(cfg.hf_dataset_id, split=cfg.hf_split)
    ds = ds.shuffle(seed=cfg.seed) if cfg.num_samples < len(ds) else ds
    n = min(cfg.num_samples, len(ds))
    samples = [ds[i] for i in range(n)]
    if not samples:
        raise DataValidationError("Dataset returned zero samples.")
    return samples


def build_pairs(raw_samples: list[dict]) -> list[tuple[str, str]]:
    all_pairs: list[tuple[str, str]] = []
    for idx, example in enumerate(raw_samples):
        try:
            pairs = normalize_example(example)
        except DataValidationError as exc:
            raise DataValidationError(f"Sample #{idx} failed validation: {exc}") from exc
        all_pairs.extend(pairs)
    return all_pairs


def to_phase2_record(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def to_phase3_record(user: str, assistant: str, system_prompt: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def validate_record(record: dict) -> None:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise DataValidationError(f"record has no messages list: {record!r}")
    roles_seen = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            raise DataValidationError(f"malformed role in record: {msg!r}")
        _clean_text(content, f"{role}.content")
        roles_seen.append(role)
    if roles_seen.count("user") != 1 or roles_seen.count("assistant") != 1:
        raise DataValidationError(f"record is not single-turn (roles={roles_seen}): {record!r}")


def write_jsonl(records: Iterable[dict], path: Path) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            validate_record(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def run(cfg: DataConfig = None) -> dict:
    cfg = cfg or DataConfig()

    raw_samples = load_raw_samples(cfg)
    with open(cfg.raw_path, "w", encoding="utf-8") as f:
        for example in raw_samples:
            f.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")

    pairs = build_pairs(raw_samples)

    phase2_records = (to_phase2_record(u, a) for u, a in pairs)
    phase3_records = (to_phase3_record(u, a, FIXED_SYSTEM_PROMPT) for u, a in pairs)

    n_phase2 = write_jsonl(phase2_records, cfg.phase2_path)
    n_phase3 = write_jsonl(phase3_records, cfg.phase3_path)

    summary = {
        "raw_samples": len(raw_samples),
        "pairs_extracted": len(pairs),
        "phase2_records": n_phase2,
        "phase3_records": n_phase3,
        "raw_path": str(cfg.raw_path),
        "phase2_path": str(cfg.phase2_path),
        "phase3_path": str(cfg.phase3_path),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Data-prep stage: download + reformat + validate.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    args = parser.parse_args()

    cfg = DataConfig()
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.dataset_id is not None:
        cfg.hf_dataset_id = args.dataset_id

    try:
        summary = run(cfg)
    except DataValidationError as exc:
        print(f"FATAL: data validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
