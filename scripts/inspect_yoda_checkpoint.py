#!/usr/bin/env python3
"""Inspect a YODA checkpoint without materializing its tensor payloads."""

from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--examples", type=int, default=8)
    return parser.parse_args()


def unwrap_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for wrapper in ("state_dict", "net", "model"):
        if wrapper in checkpoint:
            checkpoint = checkpoint[wrapper]
    consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
    return checkpoint


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(
        args.checkpoint,
        map_location="meta",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint root is {type(checkpoint).__name__}, not dict")
    checkpoint = unwrap_state_dict(checkpoint)

    print(f"checkpoint={args.checkpoint.resolve()}")
    print(f"size_bytes={args.checkpoint.stat().st_size}")
    print(f"top_level_keys={list(checkpoint)}")
    for key in sorted(checkpoint):
        value = checkpoint[key]
        if not isinstance(value, dict):
            print(f"{key}: type={type(value).__name__} value={value!r}")
            continue
        tensor_count = sum(torch.is_tensor(item) for item in value.values())
        prefixes = collections.Counter(
            str(name).split(".", maxsplit=1)[0] for name in value
        )
        print(
            f"{key}: entries={len(value)} tensors={tensor_count} "
            f"prefixes={dict(prefixes.most_common(args.examples))}"
        )
        for name in list(value)[: args.examples]:
            item = value[name]
            shape = tuple(item.shape) if torch.is_tensor(item) else type(item).__name__
            print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
