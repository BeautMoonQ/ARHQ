"""Minimal dataset loading for ARHQ evaluation."""

from __future__ import annotations

import json
import os
from typing import Any

SUPPORTED_DATASETS = ["MATH-500", "ZebraLogic"]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_parquet(path: str) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    names = table.schema.names
    return [
        {name: table[name][idx].as_py() for name in names}
        for idx in range(table.num_rows)
    ]


def _math_prompt(question: str, think_mode: bool) -> str:
    if think_mode:
        suffix = "Please reason step by step, and put your final answer within \\boxed{}."
    else:
        suffix = "Please directly give the final answer within \\boxed{}."
    return f"{question}\n\n{suffix}"


def _zebra_prompt(puzzle: str, think_mode: bool) -> str:
    if think_mode:
        suffix = (
            "Please reason step by step, then provide your answer as a JSON "
            "object matching the puzzle solution format."
        )
    else:
        suffix = (
            "Please provide your answer as a JSON object matching the puzzle "
            "solution format."
        )
    return f"{puzzle}\n\n{suffix}"


def load_dataset(name: str, think_mode: bool, data_dir: str,
                 max_samples: int = -1, gpqa_subset: str | None = None):
    """Load a small set of datasets used by the ARHQ experiments."""
    del gpqa_subset
    if name == "MATH-500":
        path = os.path.join(data_dir, "MATH-500", "test.jsonl")
        samples = [
            {
                "question": _math_prompt(row["problem"], think_mode),
                "answer": str(row.get("answer", "")),
                "meta": {k: v for k, v in row.items() if k not in ("problem", "answer")},
            }
            for row in _load_jsonl(path)
        ]
    elif name == "ZebraLogic":
        path = os.path.join(data_dir, "ZebraLogic", "test-00000-of-00001.parquet")
        samples = [
            {
                "question": _zebra_prompt(row["puzzle"], think_mode),
                "answer": row.get("solution"),
                "meta": {k: v for k, v in row.items() if k not in ("puzzle", "solution")},
            }
            for row in _load_parquet(path)
        ]
    else:
        raise ValueError(
            f"Unsupported dataset {name!r}. Minimal ARHQ supports {SUPPORTED_DATASETS}."
        )

    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]
    return samples
