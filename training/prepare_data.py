#!/usr/bin/env python3
"""Validate Torun chat-format JSONL data and write normalized JSONL."""

import argparse
import json
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}


def normalize_record(record: object, line_number: int) -> dict:
    if not isinstance(record, dict):
        raise ValueError(f"line {line_number}: each JSONL row must be an object")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: 'messages' must be a non-empty list")

    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"line {line_number}, message {index}: expected an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"line {line_number}, message {index}: unsupported role {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"line {line_number}, message {index}: content must be a non-empty string"
            )
        normalized.append({"role": role, "content": content.strip()})

    if normalized[-1]["role"] != "assistant":
        raise ValueError(f"line {line_number}: final message must be from assistant")
    if not any(message["role"] == "user" for message in normalized):
        raise ValueError(f"line {line_number}: at least one user message is required")
    return {"messages": normalized}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                clean = normalize_record(record, line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                raise SystemExit(str(exc)) from exc
            target.write(json.dumps(clean, ensure_ascii=False) + "\n")
            count += 1

    print(f"Validated {count} conversations -> {args.output}")


if __name__ == "__main__":
    main()
