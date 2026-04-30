"""Minimal evaluation utilities for ARHQ generation runs."""

from __future__ import annotations

import json
import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_thinking_and_answer(tokenizer, output_ids: list[int]) -> tuple[str, str]:
    if not output_ids:
        return "", ""

    think_end_id: Optional[int] = None
    try:
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if isinstance(think_end_id, list):
            think_end_id = think_end_id[0] if think_end_id else None
    except Exception:
        think_end_id = None

    split_idx = 0
    if isinstance(think_end_id, int) and think_end_id in output_ids:
        split_idx = len(output_ids) - output_ids[::-1].index(think_end_id)

    thinking = tokenizer.decode(
        output_ids[:split_idx], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[split_idx:], skip_special_tokens=True
    ).strip("\n")
    return thinking, content


def _extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth != 0:
        return None
    return text[start:pos - 1].strip()


def extract_answer(text: str, dataset_name: str):
    """Lightweight extraction used for online progress only.

    ZebraLogic accuracy is computed offline by ``calc_acc.py`` style logic, so
    this function returns ``None`` for ZebraLogic instead of trying to judge a
    JSON table during generation.
    """
    if not text:
        return None
    if dataset_name == "MATH-500":
        return _extract_boxed(text)
    if dataset_name == "ZebraLogic":
        return None
    match = re.findall(r"\b([A-D])\b", text)
    return match[-1] if match else None


def _result_path(output_dir: str, idx: int) -> str:
    return os.path.join(output_dir, f"{idx:06d}.json")


def save_result(output_dir: str, idx: int, dataset_name: str, question: str,
                gold_answer, repeat_num: int, results: list[dict]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "idx": idx,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "repeat_num": repeat_num,
        "results": results,
    }
    with open(_result_path(output_dir, idx), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_completed_set(output_dir: str, num_questions: int, repeat_num: int) -> set[int]:
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    for idx in range(num_questions):
        path = _result_path(output_dir, idx)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            repeats = {r.get("repeat_index") for r in payload.get("results", [])}
            if all(i in repeats for i in range(repeat_num)):
                completed.add(idx)
        except Exception:
            continue
    return completed


def load_partial_results(output_dir: str, idx: int) -> list[dict]:
    path = _result_path(output_dir, idx)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("results", [])
    except Exception:
        return []
