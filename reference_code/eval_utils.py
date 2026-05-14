"""
eval_utils.py
-------------
评测通用工具：
  - 随机种子设置
  - 模型加载
  - thinking/content 拆分
  - 答案抽取（各数据集）
  - 结果保存与断点续跑
"""

import json
import math
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device_map: str = "auto"):
    """以 bfloat16 加载 Qwen3 模型和 tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    # 左填充（批量生成需要）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def print_device_info(model: torch.nn.Module) -> None:
    device = next(model.parameters()).device
    print(f"[INFO] Device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name()}")
        cap = torch.cuda.get_device_capability()
        print(f"[INFO] Compute Capability: {cap[0]}.{cap[1]}")
        is_blackwell = cap[0] >= 10
        print(f"[INFO] Blackwell GPU: {is_blackwell}")
        if not is_blackwell:
            print("[INFO] 非 Blackwell GPU，模拟量化无硬件加速")


# ─────────────────────────────────────────────────────────────────────────────
# 输出解析
# ─────────────────────────────────────────────────────────────────────────────

def split_thinking_and_answer(tokenizer, output_ids: List[int]) -> Tuple[str, str]:
    """
    将 output_ids 拆分为 thinking 部分和正文部分。
    返回 (thinking_content, answer_content)。

    注意：apply_chat_template 会在 prompt 末尾自动追加 <think>\n，
    因此模型实际输出的 output_ids 不含开头的 <think> token，
    只有 [思考内容] + [</think>] + [答案]。
    必须通过查找 </think> token id 来切分，而不能依赖正则匹配 <think>。
    """
    if not output_ids:
        return "", ""

    think_end_id: Optional[int] = None
    try:
        think_end_id = tokenizer.convert_tokens_to_ids("</think>")
        if isinstance(think_end_id, list):
            think_end_id = think_end_id[0] if think_end_id else None
    except Exception:
        think_end_id = None

    index = 0
    if isinstance(think_end_id, int) and think_end_id in output_ids:
        # 找最后一个 </think>，index 指向其后一位
        index = len(output_ids) - output_ids[::-1].index(think_end_id)

    thinking_ids = output_ids[:index]
    answer_ids = output_ids[index:]

    thinking_text = tokenizer.decode(thinking_ids, skip_special_tokens=True).strip("\n")
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip("\n")
    return thinking_text, answer_text


# ─────────────────────────────────────────────────────────────────────────────
# 答案抽取
# ─────────────────────────────────────────────────────────────────────────────

def _extract_boxed(text: str) -> Optional[str]:
    """抽取 \\boxed{...} 中的内容（支持嵌套括号）。"""
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start: i - 1].strip() if depth == 0 else None


def _extract_letter(text: str) -> Optional[str]:
    """从文本末尾抽取单字母选项（A/B/C/D）。"""
    # 优先找"The answer is X"模式
    m = re.search(r"(?:answer is|答案是|final answer)[:\s]*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 找最后一个单独出现的大写字母
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1].upper()
    return None


def _extract_code_block(text: str) -> Optional[str]:
    """抽取最后一个 ```python ... ``` 代码块。"""
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # fallback：任意代码块
    blocks = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return None


def extract_answer(text: str, dataset_name: str) -> Optional[str]:
    """
    根据数据集类型从模型输出中抽取答案。
    返回抽取到的答案字符串，或 None。
    """
    if not text:
        return None

    if dataset_name in ("AIME2025", "MATH-500", "gsm8k"):
        return _extract_boxed(text)

    if dataset_name in ("gpqa", "MuSR", "ZebraLogic"):
        return _extract_letter(text)

    if dataset_name == "code_generation_lite":
        return _extract_code_block(text)

    if dataset_name == "IFEval":
        # IFEval 不做字符串匹配，保存完整输出供后续规则检查
        return text

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 结果保存与断点续跑
# ─────────────────────────────────────────────────────────────────────────────

def _result_path(output_dir: str, idx: int) -> str:
    return os.path.join(output_dir, f"{idx:06d}.json")


def save_result(
    output_dir: str,
    idx: int,
    dataset_name: str,
    question: str,
    gold_answer: Optional[str],
    repeat_num: int,
    results: List[dict],
) -> None:
    """
    将单个样本的所有 repeat 结果保存到 {output_dir}/{idx:06d}.json。
    每次调用都覆盖写入（保留历史所有 repeat）。
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "idx": idx,
        "dataset_name": dataset_name,
        "question": question,
        "gold_answer": gold_answer,
        "repeat_num": repeat_num,
        "results": results,
    }
    path = _result_path(output_dir, idx)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_completed_set(output_dir: str, num_questions: int, repeat_num: int) -> set:
    """
    扫描 output_dir，找出已完成全部 repeat_num 次推理的题目 idx 集合。
    """
    completed = set()
    if not os.path.isdir(output_dir):
        return completed
    for idx in range(num_questions):
        path = _result_path(output_dir, idx)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data.get("results", [])) >= repeat_num:
                completed.add(idx)
        except Exception:
            pass
    return completed


def load_partial_results(output_dir: str, idx: int) -> List[dict]:
    """
    加载已保存的部分 repeat 结果（用于断点续跑时恢复进度）。
    """
    path = _result_path(output_dir, idx)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    except Exception:
        return []
