#!/usr/bin/env python3
"""
eval_vllm.py
------------
使用 vLLM 离线推理引擎的统一评测脚本。

适用于 compressed-tensors / AWQ / GPTQ / fp16 等 vLLM 原生支持的模型格式。
也支持 --nvfp4 模式：使用 transformers FPQuantConfig 进行 NVFP4 模拟量化推理
（不依赖 vLLM，任意 CUDA GPU 均可运行）。
支持 --quarot / --quip 模式：加载由 llm-compressor 量化的 compressed-tensors 格式
模型（vLLM 原生支持，与普通 vLLM 加载路径完全相同，标志仅用于命名/识别）。

功能与 eval_sglang.py 对齐：
  - 多数据集评测
  - 断点续跑
  - repeat_num 多次推理
  - 多进程切分 (rank/world_size)

用法示例
--------
# 单数据集
python claude/eval_vllm.py --datasets MATH-500

# 多数据集
python claude/eval_vllm.py --datasets AIME2025 MATH-500 gsm8k

# 自定义模型路径
python claude/eval_vllm.py --datasets MATH-500 --model_path /path/to/model

# 指定 tensor parallel
python claude/eval_vllm.py --datasets MATH-500 --tp 2

# 多进程切分
python claude/eval_vllm.py --datasets MATH-500 --rank 0 --world_size 4

# NVFP4 模拟量化模式（使用 transformers FPQuantConfig）
python claude/eval_vllm.py --datasets MATH-500 --nvfp4 --model_path Qwen/Qwen3-8B

# QuaRot INT4 量化模型（compressed-tensors 格式，vLLM 原生加载）
python claude/eval_vllm.py --datasets MATH-500 --quarot
python claude/eval_vllm.py --datasets MATH-500 --quarot --model_path /path/to/quarot-model

# QuIP INT4 量化模型（compressed-tensors 格式，vLLM 原生加载）
python claude/eval_vllm.py --datasets MATH-500 --quip
python claude/eval_vllm.py --datasets MATH-500 --quip --model_path /path/to/quip-model

依赖: pip install vllm  (nvfp4 模式仅需 transformers torch accelerate)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set

# 将 claude/ 目录加入路径，复用同目录模块
_CLAUDE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CLAUDE_DIR not in sys.path:
    sys.path.insert(0, _CLAUDE_DIR)

from dataset_loader import load_dataset, SUPPORTED_DATASETS
from eval_utils import (
    set_seed,
    split_thinking_and_answer,
    extract_answer,
    save_result,
    get_completed_set,
    load_partial_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# 默认值
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH        = os.path.expanduser("~/work/models/Qwen3-4B-Thinking-2507-AWQ-4bit")
DEFAULT_QUAROT_MODEL_PATH = os.path.expanduser("~/work/models/Qwen3-4B-Thinking-2507-quarot-w4a16")
DEFAULT_QUIP_MODEL_PATH   = os.path.expanduser("~/work/models/Qwen3-4B-Thinking-2507-quip-w4a16")
DEFAULT_DATA_DIR = os.path.expanduser("~/work/data")
DEFAULT_OUTPUT_BASE = os.path.join(_CLAUDE_DIR, "eval_result")

# ─────────────────────────────────────────────────────────────────────────────
# 参数解析
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM 离线推理评测脚本（支持 compressed-tensors / AWQ / GPTQ 等格式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 数据集 ----
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        choices=SUPPORTED_DATASETS,
        metavar="DS",
        help=f"要评测的数据集，可多选。可选: {SUPPORTED_DATASETS}",
    )
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="每个数据集最多评测样本数（-1 = 全部）")
    parser.add_argument("--gpqa_subset", type=str, default="gpqa_diamond",
                        choices=["gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts"])

    # ---- 模型 ----
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径（支持 vLLM 支持的所有格式）")
    parser.add_argument("--think_mode", type=str, default="think",
                        choices=["think", "no-think"])
    parser.add_argument("--tp", type=int, default=1,
                        help="tensor parallel 数（vLLM tensor_parallel_size）")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.88,
                        help="GPU 显存利用率")

    # ---- 推理参数 ----
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repeat_num", type=int, default=1,
                        help="每题重复推理次数")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="每批并发请求数（vLLM 内部会自动 continuous batching）")

    # ---- 输出 ----
    parser.add_argument("--output_base", type=str, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--output_name", type=str, default=None,
                        help="实验名称子目录（默认自动生成）")

    # ---- NVFP4 模拟量化 ----
    parser.add_argument("--nvfp4", action="store_true",
                        help="使用 transformers FPQuantConfig 进行 NVFP4 模拟量化推理"
                             "（不使用 vLLM，任意 CUDA GPU 可运行）")
    parser.add_argument("--nvfp4_format", type=str, default="nvfp4",
                        choices=["nvfp4", "mxfp4"],
                        help="FP4 量化格式（nvfp4 质量更好，mxfp4 兼容性更广）")
    parser.add_argument("--nvfp4_method", type=str, default="abs_max",
                        choices=["abs_max", "quest"],
                        help="量化缩放方法（PTQ 推荐 abs_max，QAT 推荐 quest）")

    # ---- QuaRot / QuIP compressed-tensors 量化 ----
    parser.add_argument("--quarot", action="store_true",
                        help="加载 QuaRot INT4 量化模型（compressed-tensors 格式，vLLM 原生支持）；"
                             f"默认模型路径: {DEFAULT_QUAROT_MODEL_PATH}")
    parser.add_argument("--quip", action="store_true",
                        help="加载 QuIP INT4 量化模型（compressed-tensors 格式，vLLM 原生支持）；"
                             f"默认模型路径: {DEFAULT_QUIP_MODEL_PATH}")

    # ---- 多进程切分 ----
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# prompt 构造
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompt(tokenizer, question: str) -> str:
    """用 tokenizer 的 chat_template 构造完整 prompt 文本。"""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# 推理核心
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_eval(
    dataset_name: str,
    dataset: List[dict],
    llm,
    tokenizer,
    sampling_params,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """对单个数据集执行评测。"""

    num_questions = len(dataset)
    if args.nvfp4:
        seed_list = list(range(1, args.repeat_num + 1))
    else:
        seed_list = list(range(args.repeat_num))
    

    # 断点续跑
    completed_set = get_completed_set(output_dir, num_questions, args.repeat_num)
    if len(completed_set) >= num_questions:
        print(f"  [{dataset_name}] 已全部完成，跳过。")
        return
    print(f"  [{dataset_name}] 已完成 {len(completed_set)}/{num_questions} 题")

    # 待处理问题列表
    remaining: List[int] = [i for i in range(num_questions) if i not in completed_set]

    # 多进程切分
    if args.world_size > 1:
        total_q = len(remaining)
        per_rank = (total_q + args.world_size - 1) // args.world_size
        start = args.rank * per_rank
        end = min(start + per_rank, total_q)
        if start >= total_q:
            print(f"  [{dataset_name}] rank {args.rank} 无任务")
            return
        remaining = remaining[start:end]
        print(f"  [{dataset_name}] rank {args.rank}/{args.world_size}: "
              f"负责 [{start}:{end})，共 {len(remaining)} 题")

    print(f"  [{dataset_name}] 待处理: {len(remaining)} 题，"
          f"每题 {args.repeat_num} 次推理")

    # prompt 缓存
    prompt_cache: Dict[int, str] = {}

    def get_prompt(q_idx: int) -> str:
        if q_idx not in prompt_cache:
            prompt_cache[q_idx] = build_chat_prompt(
                tokenizer, dataset[q_idx]["question"]
            )
        return prompt_cache[q_idx]

    # 按 batch_size 分批；每批问题先跑完全部 repeat，再保存
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        batch_qs = remaining[batch_idx * args.batch_size:
                             (batch_idx + 1) * args.batch_size]
        actual_bs = len(batch_qs)

        print(f"\n  {'='*70}")
        print(f"  [{dataset_name}] batch {batch_idx+1}/{total_batches}，"
              f"问题索引 {batch_qs[:5]}{'...' if len(batch_qs) > 5 else ''}，"
              f"共 {actual_bs} 题")

        # 加载已有部分结果（断点续跑）
        pending: Dict[int, List[dict]] = {}
        for q_idx in batch_qs:
            pending[q_idx] = load_partial_results(output_dir, q_idx)

        # 依次跑每个 repeat
        now = time.time()

        for r_idx in range(args.repeat_num):
            # 检查哪些题还需要跑这个 r_idx
            need_qs = [
                q_idx for q_idx in batch_qs
                if not any(r["repeat_index"] == r_idx for r in pending[q_idx])
            ]
            if not need_qs:
                print(f"  repeat {r_idx+1}/{args.repeat_num}: 全部已完成，跳过")
                continue

            print(f"\n  ---- repeat {r_idx+1}/{args.repeat_num} "
                  f"(seed={seed_list[r_idx]}) ----")

            set_seed(seed_list[r_idx])
            
            print(f"cost time: {time.time() - now:.2f}s")
            now = time.time()

            # 构造批量 prompts
            prompts = [get_prompt(q_idx) for q_idx in need_qs]

            # vLLM 批量推理
            tic = time.time()
            outputs = llm.generate(prompts, sampling_params)
            toc = time.time()
            print(f"  推理耗时: {toc - tic:.2f}s ({len(need_qs)} 题)")

            # 处理结果
            for task_i, q_idx in enumerate(need_qs):
                # vLLM 返回 RequestOutput，取第一个 CompletionOutput
                completion = outputs[task_i].outputs[0]
                output_ids = list(completion.token_ids)

                # 使用 token id 精确切分 thinking / answer
                thinking, content = split_thinking_and_answer(tokenizer, output_ids)
                extracted = extract_answer(content or thinking, dataset_name)

                print(f"\n  --- 题 {q_idx}, repeat {r_idx+1} ---")
                thinking_preview = (
                    (thinking[:150] + "...") if len(thinking) > 150
                    else (thinking or "[无思维内容]")
                )
                content_preview = (
                    (content[:300] + "...") if len(content) > 300
                    else content
                )
                print(f"  [thinking] {thinking_preview}")
                print(f"  [content]  {content_preview}")
                if extracted is not None:
                    print(f"  [extracted] {str(extracted)[:120]}")

                pending[q_idx].append({
                    "repeat_index": r_idx,
                    "seed": seed_list[r_idx],
                    "output_ids": output_ids,
                    "thinking_content": thinking,
                    "content": content,
                    "extracted_answer": extracted,
                })

            # 每跑完一次 repeat，立即更新 json
            for q_idx in batch_qs:
                results_so_far = sorted(
                    pending[q_idx], key=lambda r: r["repeat_index"]
                )
                save_result(
                    output_dir=output_dir,
                    idx=q_idx,
                    dataset_name=dataset_name,
                    question=dataset[q_idx]["question"],
                    gold_answer=dataset[q_idx]["answer"],
                    repeat_num=args.repeat_num,
                    results=results_so_far,
                )

        # batch 完成后打印正确率
        correct = 0
        total_with_ans = 0
        for q_idx in batch_qs:
            gold = dataset[q_idx]["answer"]
            if gold is None:
                continue
            total_with_ans += 1
            extracted_list = [r.get("extracted_answer") for r in pending[q_idx]]
            if any(e == gold for e in extracted_list):
                correct += 1
        if total_with_ans > 0:
            print(f"\n  [{dataset_name}] batch {batch_idx+1} "
                  f"正确率: {correct}/{total_with_ans}")

    print(f"\n  [{dataset_name}] 评测完成，结果保存于: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# NVFP4 模拟量化推理核心（使用 transformers model.generate）
# ─────────────────────────────────────────────────────────────────────────────

def run_dataset_eval_nvfp4(
    dataset_name: str,
    dataset: List[dict],
    model,
    tokenizer,
    args: argparse.Namespace,
    output_dir: str,
) -> None:
    """使用 transformers model.generate 对单个数据集执行 NVFP4 模拟量化评测。"""
    import torch

    num_questions = len(dataset)
    if args.nvfp4:
        seed_list = list(range(10, args.repeat_num + 10))
    else:
        seed_list = list(range(args.repeat_num))

    # 断点续跑
    completed_set = get_completed_set(output_dir, num_questions, args.repeat_num)
    if len(completed_set) >= num_questions:
        print(f"  [{dataset_name}] 已全部完成，跳过。")
        return
    print(f"  [{dataset_name}] 已完成 {len(completed_set)}/{num_questions} 题")

    # 待处理问题列表
    remaining: List[int] = [i for i in range(num_questions) if i not in completed_set]

    # 多进程切分
    if args.world_size > 1:
        total_q = len(remaining)
        per_rank = (total_q + args.world_size - 1) // args.world_size
        start = args.rank * per_rank
        end = min(start + per_rank, total_q)
        if start >= total_q:
            print(f"  [{dataset_name}] rank {args.rank} 无任务")
            return
        remaining = remaining[start:end]
        print(f"  [{dataset_name}] rank {args.rank}/{args.world_size}: "
              f"负责 [{start}:{end})，共 {len(remaining)} 题")

    print(f"  [{dataset_name}] 待处理: {len(remaining)} 题，"
          f"每题 {args.repeat_num} 次推理")

    # prompt 缓存
    prompt_cache: Dict[int, str] = {}

    def get_prompt(q_idx: int) -> str:
        if q_idx not in prompt_cache:
            prompt_cache[q_idx] = build_chat_prompt(
                tokenizer, dataset[q_idx]["question"]
            )
        return prompt_cache[q_idx]

    # 生成参数
    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False

    # 设置 tokenizer 左填充（批量生成需要）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 按 batch_size 分批
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        batch_qs = remaining[batch_idx * args.batch_size:
                             (batch_idx + 1) * args.batch_size]
        actual_bs = len(batch_qs)

        print(f"\n  {'='*70}")
        print(f"  [{dataset_name}] batch {batch_idx+1}/{total_batches}，"
              f"问题索引 {batch_qs[:5]}{'...' if len(batch_qs) > 5 else ''}，"
              f"共 {actual_bs} 题")

        # 加载已有部分结果（断点续跑）
        pending: Dict[int, List[dict]] = {}
        for q_idx in batch_qs:
            pending[q_idx] = load_partial_results(output_dir, q_idx)

        # 依次跑每个 repeat
        for r_idx in range(args.repeat_num):
            need_qs = [
                q_idx for q_idx in batch_qs
                if not any(r["repeat_index"] == r_idx for r in pending[q_idx])
            ]
            if not need_qs:
                print(f"  repeat {r_idx+1}/{args.repeat_num}: 全部已完成，跳过")
                continue

            print(f"\n  ---- repeat {r_idx+1}/{args.repeat_num} "
                  f"(seed={seed_list[r_idx]}) ----")

            set_seed(seed_list[r_idx])

            # 构造批量 prompts
            prompts = [get_prompt(q_idx) for q_idx in need_qs]

            # tokenize 并 padding
            model_inputs = tokenizer(
                prompts, return_tensors="pt", padding=True,
            ).to(model.device)
            input_lengths = [len(ids) for ids in tokenizer(prompts)["input_ids"]]

            # transformers model.generate
            tic = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, **gen_kwargs)
            toc = time.time()
            print(f"  推理耗时: {toc - tic:.2f}s ({len(need_qs)} 题)")

            # 处理结果
            padded_len = model_inputs.input_ids.shape[1]
            for task_i, q_idx in enumerate(need_qs):
                # 左填充，真正输入在 [pad_len : padded_len)
                output_ids = generated_ids[task_i][padded_len:].tolist()

                thinking, content = split_thinking_and_answer(tokenizer, output_ids)
                extracted = extract_answer(content or thinking, dataset_name)

                print(f"\n  --- 题 {q_idx}, repeat {r_idx+1} ---")
                thinking_preview = (
                    (thinking[:150] + "...") if len(thinking) > 150
                    else (thinking or "[无思维内容]")
                )
                content_preview = (
                    (content[:300] + "...") if len(content) > 300
                    else content
                )
                print(f"  [thinking] {thinking_preview}")
                print(f"  [content]  {content_preview}")
                if extracted is not None:
                    print(f"  [extracted] {str(extracted)[:120]}")

                pending[q_idx].append({
                    "repeat_index": r_idx,
                    "seed": seed_list[r_idx],
                    "output_ids": output_ids,
                    "thinking_content": thinking,
                    "content": content,
                    "extracted_answer": extracted,
                })

            # 每跑完一次 repeat，立即更新 json
            for q_idx in batch_qs:
                results_so_far = sorted(
                    pending[q_idx], key=lambda r: r["repeat_index"]
                )
                save_result(
                    output_dir=output_dir,
                    idx=q_idx,
                    dataset_name=dataset_name,
                    question=dataset[q_idx]["question"],
                    gold_answer=dataset[q_idx]["answer"],
                    repeat_num=args.repeat_num,
                    results=results_so_far,
                )

        # batch 完成后打印正确率
        correct = 0
        total_with_ans = 0
        for q_idx in batch_qs:
            gold = dataset[q_idx]["answer"]
            if gold is None:
                continue
            total_with_ans += 1
            extracted_list = [r.get("extracted_answer") for r in pending[q_idx]]
            if any(e == gold for e in extracted_list):
                correct += 1
        if total_with_ans > 0:
            print(f"\n  [{dataset_name}] batch {batch_idx+1} "
                  f"正确率: {correct}/{total_with_ans}")

    print(f"\n  [{dataset_name}] 评测完成，结果保存于: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.world_size <= 0:
        raise ValueError(f"world_size 必须为正整数，当前 {args.world_size}")
    if not (0 <= args.rank < args.world_size):
        raise ValueError(f"rank 必须在 [0, {args.world_size-1}]，当前 {args.rank}")

    # --quarot / --quip 设置默认模型路径（仅当用户未显式指定时）
    if args.quarot and args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = DEFAULT_QUAROT_MODEL_PATH
    elif args.quip and args.model_path == DEFAULT_MODEL_PATH:
        args.model_path = DEFAULT_QUIP_MODEL_PATH

    # 自动生成 output_name
    if args.output_name is None:
        model_basename = os.path.basename(args.model_path.rstrip("/"))
        if args.nvfp4:
            prefix = "nvfp4"
        elif args.quarot:
            prefix = "quarot"
        elif args.quip:
            prefix = "quip"
        else:
            prefix = "vllm"
        args.output_name = f"{prefix}_{model_basename}"

    # ── 打印配置 ──
    print("=" * 80)
    if args.nvfp4:
        print("NVFP4 模拟量化评测配置 (transformers FPQuantConfig)")
    elif args.quarot:
        print("QuaRot INT4 量化评测配置 (compressed-tensors, vLLM)")
    elif args.quip:
        print("QuIP INT4 量化评测配置 (compressed-tensors, vLLM)")
    else:
        print("vLLM 评测配置")
    print(f"  数据集:      {args.datasets}")
    print(f"  模型:        {args.model_path}")
    if args.nvfp4:
        print(f"  量化格式:    {args.nvfp4_format}")
        print(f"  缩放方法:    {args.nvfp4_method}")
    else:
        print(f"  tp_size:     {args.tp}")
    print(f"  think_mode:  {args.think_mode}")
    print(f"  repeat_num:  {args.repeat_num}")
    print(f"  batch_size:  {args.batch_size}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  rank/world:  {args.rank}/{args.world_size}")
    print(f"  output_base: {args.output_base}")
    print(f"  output_name: {args.output_name}")
    print(f"  gen: max_new_tokens={args.max_new_tokens}, "
          f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print("=" * 80)

    # ── 创建推理引擎 ──
    if args.nvfp4:
        # NVFP4 模拟量化模式：使用 transformers + FPQuantConfig
        print("\n[Step 1] 加载模型 (NVFP4 pseudoquantization)...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig

        quant_config = FPQuantConfig(
            forward_dtype=args.nvfp4_format,
            pseudoquantization=True,
            forward_method=args.nvfp4_method,
            transform_init="hadamard",
        )
        print(f"  量化配置: {quant_config}")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"  模型加载完成，内存占用: {model.get_memory_footprint() / 1e9:.2f} GB")
    else:
        # vLLM 模式
        print("\n[Step 1] 创建 vLLM LLM...")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_new_tokens * 2,
        )
        tokenizer = llm.get_tokenizer()
        print("  LLM 创建完成")

    # ── 构造 SamplingParams（仅 vLLM 模式需要）──
    sampling_params = None
    if not args.nvfp4:
        from vllm import SamplingParams
        sp_kwargs = {
            "max_tokens": args.max_new_tokens,
        }
        if args.temperature > 0:
            sp_kwargs["temperature"] = args.temperature
            sp_kwargs["top_p"] = args.top_p
            if args.top_k > 0:
                sp_kwargs["top_k"] = args.top_k
        else:
            sp_kwargs["temperature"] = 0  # greedy

        sampling_params = SamplingParams(**sp_kwargs)

    # ── 逐数据集评测 ──
    set_seed(42)

    for ds_name in args.datasets:
        print(f"\n{'='*80}")
        print(f"[Step 2] 数据集: {ds_name}")

        think_mode_flag = args.think_mode == "think"
        dataset = load_dataset(
            name=ds_name,
            think_mode=think_mode_flag,
            data_dir=args.data_dir,
            max_samples=args.max_samples,
            gpqa_subset=args.gpqa_subset,
        )
        print(f"  样本数: {len(dataset)}")

        output_dir = os.path.join(args.output_base, args.output_name, ds_name)
        os.makedirs(output_dir, exist_ok=True)

        if args.nvfp4:
            run_dataset_eval_nvfp4(
                dataset_name=ds_name,
                dataset=dataset,
                model=model,
                tokenizer=tokenizer,
                args=args,
                output_dir=output_dir,
            )
        else:
            run_dataset_eval(
                dataset_name=ds_name,
                dataset=dataset,
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                args=args,
                output_dir=output_dir,
            )

    print("\n" + "=" * 80)
    print("全部评测完成。")
    print(f"结果目录: {os.path.join(args.output_base, args.output_name)}")


if __name__ == "__main__":
    main()
