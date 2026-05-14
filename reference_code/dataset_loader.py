"""
dataset_loader.py
-----------------
数据集加载与 prompt 构造模块。

支持的数据集（对应 ~/work/data/ 下各子目录）：
  - AIME2025         : 数学竞赛题，问题字段 question，答案字段 answer
  - MATH-500         : 数学题，问题字段 problem，答案字段 answer
  - gsm8k            : 小学数学，问题字段 question，答案字段 answer
  - code_generation_lite : LiveCodeBench 代码生成，问题字段 question_content
  - gpqa             : 科学多选题，默认使用 gpqa_diamond 子集
  - IFEval           : 指令跟随，问题字段 prompt
  - MuSR             : 多步推理选择题，问题字段 question（含 narrative 上下文）
  - ZebraLogic       : 斑马逻辑谜题，问题字段 puzzle

每个数据集返回标准化的 list[dict]，每条含：
  - "question"  : 给模型的问题文本（已拼好 narrative/choices 等）
  - "answer"    : 标准答案（str 或 None）
  - "meta"      : 原始样本的其余字段（dict）
"""

import ast
import csv
import json
import os
import random
from typing import Dict, List, Optional, Tuple

DATA_DIR_DEFAULT = os.path.expanduser("~/work/data")

# ─────────────────────────────────────────────────────────────────────────────
# 底层文件读取
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_csv_dicts(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_parquet(path: str) -> List[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    names = table.schema.names
    rows = []
    for i in range(table.num_rows):
        rows.append({k: table[k][i].as_py() for k in names})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────────────────────────

def _make_math_prompt(question: str, think_mode: bool) -> str:
    think_hint = (
        "Please reason step by step, and put your final answer within \\boxed{}."
        if think_mode else
        "Please directly give the final answer within \\boxed{}."
    )
    return f"{question}\n\n{think_hint}"


def _make_mc_prompt(question: str, choices: List[str], think_mode: bool) -> str:
    choices_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    think_hint = (
        "Please reason step by step, and give your final answer as a single letter (A/B/C/D)."
        if think_mode else
        "Please directly give the final answer as a single letter (A/B/C/D)."
    )
    return f"{question}\n\nChoices:\n{choices_str}\n\n{think_hint}"


def _make_code_prompt(title: str, content: str, starter_code: str, think_mode: bool) -> str:
    think_hint = (
        "Please reason step by step, then provide your complete solution within a ```python``` code block."
        if think_mode else
        "Please provide your complete solution within a ```python``` code block."
    )
    parts = [f"## {title}\n\n{content}"]
    if starter_code and starter_code.strip():
        parts.append(f"**Starter code:**\n```python\n{starter_code.strip()}\n```")
    parts.append(think_hint)
    return "\n\n".join(parts)


def _make_ifeval_prompt(prompt: str) -> str:
    # IFEval 的 prompt 本身就是完整指令，不加额外 hint
    return prompt


def _make_zebralogic_prompt(puzzle: str, think_mode: bool) -> str:
    think_hint = (
        "Please reason step by step, then provide your answer as a JSON object matching the puzzle solution format."
        if think_mode else
        "Please provide your answer as a JSON object matching the puzzle solution format."
    )
    return f"{puzzle}\n\n{think_hint}"


def _make_musr_prompt(narrative: str, question: str, choices: List[str], think_mode: bool) -> str:
    choices_str = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    think_hint = (
        "Please reason step by step, and give your final answer as a single letter (A/B/C/D)."
        if think_mode else
        "Please directly give the final answer as a single letter (A/B/C/D)."
    )
    return f"{narrative}\n\nQuestion: {question}\n\nChoices:\n{choices_str}\n\n{think_hint}"


def _make_gpqa_prompt(question: str, choices: List[str], think_mode: bool) -> str:
    return _make_mc_prompt(question, choices, think_mode)


# ─────────────────────────────────────────────────────────────────────────────
# 各数据集加载
# ─────────────────────────────────────────────────────────────────────────────

def _load_aime2025(data_dir: str, think_mode: bool) -> List[dict]:
    ds_dir = os.path.join(data_dir, "AIME2025")
    samples = []
    for fname in sorted(os.listdir(ds_dir)):
        if fname.endswith(".jsonl"):
            for row in _load_jsonl(os.path.join(ds_dir, fname)):
                q_text = row["question"]
                samples.append({
                    "question": _make_math_prompt(q_text, think_mode),
                    "answer": str(row.get("answer", "")),
                    "meta": {k: v for k, v in row.items() if k not in ("question", "answer")},
                })
    return samples


def _load_math500(data_dir: str, think_mode: bool) -> List[dict]:
    path = os.path.join(data_dir, "MATH-500", "test.jsonl")
    samples = []
    for row in _load_jsonl(path):
        samples.append({
            "question": _make_math_prompt(row["problem"], think_mode),
            "answer": str(row.get("answer", "")),
            "meta": {k: v for k, v in row.items() if k not in ("problem", "answer")},
        })
    return samples


def _load_gsm8k(data_dir: str, think_mode: bool) -> List[dict]:
    path = os.path.join(data_dir, "gsm8k", "main", "test-00000-of-00001.parquet")
    samples = []
    for row in _load_parquet(path):
        # gsm8k answer 字段含推理过程，最终答案在 #### 后
        raw_ans = row.get("answer", "")
        if "####" in raw_ans:
            final_ans = raw_ans.split("####")[-1].strip()
        else:
            final_ans = raw_ans
        samples.append({
            "question": _make_math_prompt(row["question"], think_mode),
            "answer": final_ans,
            "meta": {},
        })
    return samples


def _load_code_generation_lite(data_dir: str, think_mode: bool) -> List[dict]:
    ds_dir = os.path.join(data_dir, "code_generation_lite")
    samples = []
    for fname in sorted(os.listdir(ds_dir)):
        if not fname.endswith(".jsonl"):
            continue
        for row in _load_jsonl(os.path.join(ds_dir, fname)):
            title = row.get("question_title", "")
            content = row.get("question_content", "")
            starter = row.get("starter_code", "")
            samples.append({
                "question": _make_code_prompt(title, content, starter, think_mode),
                "answer": None,  # 代码题答案靠测试用例，不做字符串匹配
                "meta": {k: v for k, v in row.items()
                         if k not in ("question_title", "question_content", "starter_code")},
            })
    return samples


def _load_gpqa(data_dir: str, think_mode: bool, subset: str = "gpqa_diamond") -> List[dict]:
    path = os.path.join(data_dir, "gpqa", f"{subset}.csv")
    samples = []
    for row in _load_csv_dicts(path):
        question = row["Question"]
        correct = row["Correct Answer"]
        wrong = [row[f"Incorrect Answer {i}"] for i in range(1, 4) if row.get(f"Incorrect Answer {i}", "").strip()]
        # 打乱选项顺序，记录正确答案的字母
        choices = [correct] + wrong
        random.shuffle(choices)
        correct_letter = chr(65 + choices.index(correct))
        samples.append({
            "question": _make_gpqa_prompt(question, choices, think_mode),
            "answer": correct_letter,
            "meta": {
                "subdomain": row.get("Subdomain", ""),
                "domain": row.get("High-level domain", ""),
                "correct_text": correct,
                "choices": choices,
            },
        })
    return samples


def _load_ifeval(data_dir: str, think_mode: bool) -> List[dict]:
    path = os.path.join(data_dir, "IFEval", "ifeval_input_data.jsonl")
    samples = []
    for row in _load_jsonl(path):
        samples.append({
            "question": _make_ifeval_prompt(row["prompt"]),
            "answer": None,  # IFEval 答案靠规则检查，不做字符串匹配
            "meta": {
                "key": row.get("key"),
                "instruction_id_list": row.get("instruction_id_list", []),
                "kwargs": row.get("kwargs", []),
                "prompt": row["prompt"],
            },
        })
    return samples


def _load_musr(data_dir: str, think_mode: bool) -> List[dict]:
    path = os.path.join(data_dir, "MuSR", "all.csv")
    samples = []
    for row in _load_csv_dicts(path):
        narrative = row.get("narrative", "")
        question = row.get("question", "")
        # choices 字段是字符串化的列表
        choices_raw = row.get("choices", "[]")
        try:
            choices = ast.literal_eval(choices_raw)
        except Exception:
            choices = [choices_raw]
        answer_choice = row.get("answer_choice", "")
        # 正确答案转为字母
        if answer_choice in choices:
            correct_letter = chr(65 + choices.index(answer_choice))
        else:
            correct_letter = answer_choice
        samples.append({
            "question": _make_musr_prompt(narrative, question, choices, think_mode),
            "answer": correct_letter,
            "meta": {
                "answer_index": row.get("answer_index"),
                "answer_choice": answer_choice,
                "choices": choices,
            },
        })
    return samples


def _load_zebralogic(data_dir: str, think_mode: bool) -> List[dict]:
    path = os.path.join(data_dir, "ZebraLogic", "test-00000-of-00001.parquet")
    samples = []
    for row in _load_parquet(path):
        samples.append({
            "question": _make_zebralogic_prompt(row["puzzle"], think_mode),
            "answer": str(row.get("solution", "")),
            "meta": {
                "id": row.get("id"),
                "size": row.get("size"),
            },
        })
    return samples


# ZebraSelect：从 ZebraLogic 中按固定 idx 列表挑选的 30 条样本子集
_ZEBRASELECT_INDICES = [
    0, 2, 3, 11, 15, 27, 31, 32, 34, 42, 48, 51, 53, 57, 77,
    210, 211, 337, 419, 427, 522, 549, 617, 629, 651, 655, 731, 824, 843, 877,
]


def _load_zebraselect(data_dir: str, think_mode: bool) -> List[dict]:
    all_samples = _load_zebralogic(data_dir, think_mode)
    samples = [all_samples[i] for i in _ZEBRASELECT_INDICES]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 统一加载入口
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_DATASETS = [
    "AIME2025",
    "MATH-500",
    "gsm8k",
    "code_generation_lite",
    "gpqa",
    "IFEval",
    "MuSR",
    "ZebraLogic",
    "ZebraSelect",
]


def load_dataset(
    name: str,
    think_mode: bool = True,
    data_dir: str = DATA_DIR_DEFAULT,
    max_samples: int = -1,
    gpqa_subset: str = "gpqa_diamond",
) -> List[dict]:
    """
    加载数据集，返回标准化 list[dict]，每条包含 question / answer / meta。

    参数：
        name         : 数据集名称，见 SUPPORTED_DATASETS
        think_mode   : 是否在 prompt 中开启思维链
        data_dir     : 数据根目录
        max_samples  : 最大样本数，-1 表示全部
        gpqa_subset  : gpqa 子集名（不含 .csv），默认 gpqa_diamond
    """
    loaders = {
        "AIME2025":              lambda: _load_aime2025(data_dir, think_mode),
        "MATH-500":              lambda: _load_math500(data_dir, think_mode),
        "gsm8k":                 lambda: _load_gsm8k(data_dir, think_mode),
        "code_generation_lite":  lambda: _load_code_generation_lite(data_dir, think_mode),
        "gpqa":                  lambda: _load_gpqa(data_dir, think_mode, gpqa_subset),
        "IFEval":                lambda: _load_ifeval(data_dir, think_mode),
        "MuSR":                  lambda: _load_musr(data_dir, think_mode),
        "ZebraLogic":            lambda: _load_zebralogic(data_dir, think_mode),
        "ZebraSelect":           lambda: _load_zebraselect(data_dir, think_mode),
    }

    if name not in loaders:
        raise ValueError(f"不支持的数据集: {name}，可选: {SUPPORTED_DATASETS}")

    samples = loaders[name]()

    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]

    return samples
