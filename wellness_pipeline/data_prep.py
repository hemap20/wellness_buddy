"""
Data-prep module — standalone runnable stage.

Downloads N raw samples from the configured HF dataset, normalizes whatever
schema it has into {user, assistant} pairs, optionally splits into
train/val (seeded, reproducible), and writes to
data/n{size}/phase{2,3}/{train,val}.jsonl — val.jsonl is omitted when the
split is skipped (num_samples below DataConfig.min_samples_for_val_split, or
val_split_ratio=0.0 and val_samples=None, which is the default/backward-compatible
behavior).

Fails loudly (raises) on malformed data rather than silently training on it.

Run standalone:
    python data_prep.py
    python data_prep.py --num-samples 5
    python data_prep.py --num-samples 100 --val-split-ratio 0.1
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

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


def compute_val_count(n_examples: int, cfg: DataConfig) -> tuple[int, str]:
    """Returns (val_count, reason). val_count=0 means no split — either
    because none was requested, or because n_examples is below
    min_samples_for_val_split (which always wins, regardless of the other
    settings)."""
    if n_examples < cfg.min_samples_for_val_split:
        if cfg.val_samples or cfg.val_split_ratio > 0:
            return 0, (
                f"num_samples={n_examples} is below min_samples_for_val_split="
                f"{cfg.min_samples_for_val_split} — training on full set, no validation split."
            )
        return 0, "val_split_ratio=0.0 and val_samples=None — no validation split requested."

    if cfg.val_samples is not None:
        n_val = cfg.val_samples
    else:
        n_val = round(n_examples * cfg.val_split_ratio)
    n_val = max(0, min(n_val, n_examples - 1))  # always keep >=1 training example

    if n_val == 0:
        return 0, "val_split_ratio=0.0 and val_samples=None — no validation split requested."
    return n_val, (
        f"num_samples={n_examples}, val_split_ratio={cfg.val_split_ratio}, "
        f"val_samples={cfg.val_samples} -> val_count={n_val}"
    )


def split_pairs(pairs: list[tuple[str, str]], cfg: DataConfig) -> tuple[list, list, str]:
    """Splits (user, assistant) pairs into (train, val) using a seeded
    shuffle — same seed as everything else in DataConfig, so the split is
    reproducible across repeated runs at the same num_samples. Splitting
    happens on the extracted pairs, before Phase 2/3 reformatting, so both
    phases share an identical train/val assignment for apples-to-apples
    comparison."""
    n_val, reason = compute_val_count(len(pairs), cfg)
    if n_val == 0:
        return pairs, [], reason

    indices = list(range(len(pairs)))
    random.Random(cfg.seed).shuffle(indices)
    val_idx = set(indices[:n_val])

    train = [p for i, p in enumerate(pairs) if i not in val_idx]
    val = [p for i, p in enumerate(pairs) if i in val_idx]

    # cheap insurance — should be guaranteed by construction (disjoint index sets)
    assert set(train).isdisjoint(set(val)), "train/val leakage detected after split"

    return train, val, reason


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
    raw_path = cfg.raw_path()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        for example in raw_samples:
            f.write(json.dumps(example, ensure_ascii=False, default=str) + "\n")

    pairs = build_pairs(raw_samples)
    train_pairs, val_pairs, split_reason = split_pairs(pairs, cfg)

    print(f"num_samples={cfg.num_samples}, train={len(train_pairs)}, val={len(val_pairs)}, seed={cfg.seed}")
    print(split_reason)

    summary = {
        "raw_samples": len(raw_samples),
        "pairs_extracted": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "seed": cfg.seed,
        "split_reason": split_reason,
        "raw_path": str(raw_path),
    }

    phase_formatters = {
        2: lambda u, a: to_phase2_record(u, a),
        3: lambda u, a: to_phase3_record(u, a, FIXED_SYSTEM_PROMPT),
    }
    for phase, formatter in phase_formatters.items():
        train_path = cfg.train_path(phase)
        train_path.parent.mkdir(parents=True, exist_ok=True)
        n_train = write_jsonl((formatter(u, a) for u, a in train_pairs), train_path)
        summary[f"phase{phase}_train_path"] = str(train_path)
        summary[f"phase{phase}_train_records"] = n_train

        if val_pairs:
            val_path = cfg.val_path(phase)
            n_val = write_jsonl((formatter(u, a) for u, a in val_pairs), val_path)
            summary[f"phase{phase}_val_path"] = str(val_path)
            summary[f"phase{phase}_val_records"] = n_val
        else:
            summary[f"phase{phase}_val_path"] = None
            summary[f"phase{phase}_val_records"] = 0

    return summary


def main():
    parser = argparse.ArgumentParser(description="Data-prep stage: download + reformat + validate.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset-id", type=str, default=None)
    parser.add_argument("--val-split-ratio", type=float, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = DataConfig()
    if args.num_samples is not None:
        cfg.num_samples = args.num_samples
    if args.dataset_id is not None:
        cfg.hf_dataset_id = args.dataset_id
    if args.val_split_ratio is not None:
        cfg.val_split_ratio = args.val_split_ratio
    if args.val_samples is not None:
        cfg.val_samples = args.val_samples

    try:
        summary = run(cfg)
    except DataValidationError as exc:
        print(f"FATAL: data validation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
