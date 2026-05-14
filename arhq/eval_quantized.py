"""Quantized inference with ARHQ/SVDQuant low-rank compensation.

Supports two modes:
  1. Dataset evaluation (--datasets MATH-500 AIME2025 ...)
  2. Single question inference (--question "...")

Usage examples:
  # Single question
  conda run -n llmc python -m arhq.eval_quantized \
      --method arhq --setting raw --rank 128 --device cuda:2 \
      --decomp_dir results/layer_results \
      --question "What is 2+2?" --max_new_tokens 4096

  # Dataset evaluation
  conda run -n llmc python -m arhq.eval_quantized \
      --method arhq --setting raw --rank 128 --device cuda:2 \
      --datasets MATH-500 --batch_size 4

  # Compare with baseline (no low-rank)
  conda run -n llmc python -m arhq.eval_quantized \
      --method baseline --device cuda:2 \
      --question "Solve x^2 - 5x + 6 = 0"
"""

import argparse
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from arhq.data import load_dataset, SUPPORTED_DATASETS
from arhq.eval_utils import (
    set_seed,
    split_thinking_and_answer,
    extract_answer,
    save_result,
    get_completed_set,
    load_partial_results,
)

from arhq.quant import nvfp4_quantize, nvfp4_quantize_2d

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.expanduser("~/work/models/Qwen3-4B-Thinking-2507")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
DEFAULT_DECOMP_DIR = os.path.join(RESULTS_DIR, "layer_results")
ATTN_PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_PROJ_TYPES = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {
    "attn": ATTN_PROJ_TYPES,
    "ffn": FFN_PROJ_TYPES,
    "all": ATTN_PROJ_TYPES + FFN_PROJ_TYPES,
}
NUM_LAYERS = 36
DEFAULT_DATA_DIR = os.path.expanduser("~/work/data")


# ─────────────────────────────────────────────────────────────────────────────
# Quantized Linear Modules
# ─────────────────────────────────────────────────────────────────────────────

class QuantizedLowRankLinear(nn.Module):
    """Simulated nvfp4 quantized linear with low-rank compensation.

    With smoothing:
      Y = Q(A / scale) @ Q(W_res)^T + (A / scale @ A_fac) @ B_r^T

    Without smoothing:
      Y = Q(A) @ Q(W_res)^T + (A @ A_fac) @ B_r^T

    All heavy matmuls use nvfp4 simulation. The low-rank branch stays in
    the model's compute dtype (bf16/fp16) for accuracy.
    """

    def __init__(self, W_res: torch.Tensor, B_r: torch.Tensor,
                 A_fac: torch.Tensor, scale: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("W_res", W_res)     # [D_out, D_in] fp16
        self.register_buffer("B_r", B_r)         # [D_out, rank] fp16
        self.register_buffer("A_fac", A_fac)     # [D_in, rank] fp16
        self.has_scale = scale is not None
        if scale is not None:
            self.register_buffer("scale", scale)  # [D_in] fp16
        else:
            self.scale = None

        # Pre-quantize W_res (it's static) to avoid recomputing every forward
        self.register_buffer("W_res_q", nvfp4_quantize_2d(W_res.float()).to(W_res.dtype))
        # W_res itself is no longer needed after pre-quantization. Keeping only
        # W_res_q matters for all-layer FFN runs where memory is tight.
        self.W_res = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., D_in]
        orig_shape = x.shape[:-1]
        D_in = x.shape[-1]
        x_flat = x.reshape(-1, D_in)  # [N, D_in]

        # Optional SmoothQuant scaling
        x_s = x_flat / self.scale if self.has_scale else x_flat

        # Quantized main branch
        x_q = nvfp4_quantize(x_s.float()).to(x_s.dtype)  # [N, D_in]
        y_main = x_q @ self.W_res_q.T  # [N, D_out]

        # Low-rank compensation (full precision)
        y_lr = (x_s @ self.A_fac) @ self.B_r.T  # [N, D_out]

        y = y_main + y_lr
        return y.reshape(*orig_shape, -1)


class QuantizedOnlyLinear(nn.Module):
    """Simulated nvfp4 quantized linear without low-rank (baseline)."""

    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.register_buffer("W_q", nvfp4_quantize_2d(W.float()).to(W.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[:-1]
        D_in = x.shape[-1]
        x_flat = x.reshape(-1, D_in)

        x_q = nvfp4_quantize(x_flat.float()).to(x_flat.dtype)
        y = x_q @ self.W_q.T

        return y.reshape(*orig_shape, -1)


def build_runtime_residual(orig_weight: torch.Tensor, decomp: dict,
                           setting: str) -> tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, Optional[torch.Tensor]]:
    """Recompute runtime residual from current model weight and saved low-rank factors.

    The saved `.pt` files provide the low-rank branch parameters. At inference
    time we rebuild the residual from the current model weight so deployment
    stays consistent even if the source `weights.pt` snapshot differs from the
    loaded model checkpoint.
    """
    W_orig = orig_weight.float()
    B_r = decomp["B_r"].float().to(W_orig.device)
    A_fac = decomp["A_fac"].float().to(W_orig.device)

    scale = None
    if setting == "smoothing":
        if "scale" not in decomp:
            raise ValueError("Missing `scale` in smoothing decomposition file")
        scale = decomp["scale"].float().to(W_orig.device)
        W_target = W_orig * scale
    else:
        W_target = W_orig

    # Rebuild residual against the currently loaded checkpoint weight.
    W_res = W_target - B_r @ A_fac.T
    return W_res, B_r, A_fac, scale


# ─────────────────────────────────────────────────────────────────────────────
# Model patching
# ─────────────────────────────────────────────────────────────────────────────

def get_projection_module(layer, proj: str) -> nn.Module:
    if proj in ATTN_PROJ_TYPES:
        return getattr(layer.self_attn, proj)
    if proj in FFN_PROJ_TYPES:
        return getattr(layer.mlp, proj)
    raise ValueError(f"Unknown projection: {proj}")


def set_projection_module(layer, proj: str, new_mod: nn.Module):
    if proj in ATTN_PROJ_TYPES:
        setattr(layer.self_attn, proj, new_mod)
    elif proj in FFN_PROJ_TYPES:
        setattr(layer.mlp, proj, new_mod)
    else:
        raise ValueError(f"Unknown projection: {proj}")


def replace_linear_projections(model, method: str, setting: str,
                               rank: int, device: str, module_set: str,
                               decomp_dir: str):
    """Replace selected attention/FFN projections with quantized modules."""
    if module_set not in MODULE_SETS:
        raise ValueError(f"Unknown module_set={module_set}, valid={sorted(MODULE_SETS)}")
    proj_types = MODULE_SETS[module_set]
    replaced = 0
    missing = 0

    for layer_idx in range(NUM_LAYERS):
        layer = model.model.layers[layer_idx]
        for proj in proj_types:
            orig_linear = get_projection_module(layer, proj)
            W_orig = orig_linear.weight.data

            if method == "baseline":
                new_mod = QuantizedOnlyLinear(W_orig)
            else:
                # Load saved decomposition
                method_candidates = [method]
                if method == "arhq":
                    method_candidates.append("r_only")
                elif method == "r_only":
                    method_candidates.append("arhq")
                path = None
                for method_name in method_candidates:
                    fname = f"{proj}_{method_name}_{setting}_rank{rank}.pt"
                    candidate = os.path.join(
                        decomp_dir, f"layer_{layer_idx}", fname
                    )
                    if os.path.exists(candidate):
                        path = candidate
                        break
                if path is None:
                    expected = os.path.join(
                        decomp_dir, f"layer_{layer_idx}",
                        f"{proj}_{method}_{setting}_rank{rank}.pt",
                    )
                    print(f"  WARN: missing {expected}, using baseline quant")
                    new_mod = QuantizedOnlyLinear(W_orig)
                    missing += 1
                else:
                    decomp = torch.load(path, map_location="cpu", weights_only=True)
                    dtype = W_orig.dtype
                    W_res, B_r, A_fac, scale = build_runtime_residual(
                        W_orig, decomp, setting
                    )
                    new_mod = QuantizedLowRankLinear(
                        W_res=W_res.to(dtype),
                        B_r=B_r.to(dtype),
                        A_fac=A_fac.to(dtype),
                        scale=scale.to(dtype) if scale is not None else None,
                    )
                    replaced += 1

            new_mod = new_mod.to(device)
            set_projection_module(layer, proj, new_mod)

    total = NUM_LAYERS * len(proj_types)
    if method == "baseline":
        print(f"Replaced {total}/{total} projections with baseline "
              f"(module_set={module_set})")
    else:
        print(f"Replaced {replaced}/{total} projections with {method} "
              f"(setting={setting}, rank={rank}, module_set={module_set})")
    if missing:
        print(f"  {missing} projections fell back to baseline (missing .pt files)")


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_quantized_model(method: str, setting: str, rank: int, device: str,
                         module_set: str, decomp_dir: str,
                         model_path: str = MODEL_PATH):
    """Load model, replace selected projections, return (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    if method != "original":
        replace_linear_projections(
            model, method, setting, rank, device, module_set, decomp_dir
        )

    # Print memory
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"GPU memory used: {mem_gb:.2f} GB")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Chat prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_chat_prompt(tokenizer, question: str, think_mode: str = "think") -> str:
    """Build chat prompt with optional thinking mode."""
    if think_mode == "no-think":
        messages = [
            {"role": "system", "content": "You should answer directly without thinking."},
            {"role": "user", "content": question},
        ]
    else:
        messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single question inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_single_question(model, tokenizer, args):
    """Run inference on a single question."""
    prompt = build_chat_prompt(tokenizer, args.question, args.think_mode)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
    }
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False

    print(f"\nGenerating (max_new_tokens={args.max_new_tokens})...")
    tic = time.time()
    output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - tic

    new_ids = output_ids[0, input_len:].tolist()
    num_tokens = len(new_ids)
    tps = num_tokens / elapsed if elapsed > 0 else 0

    thinking, content = split_thinking_and_answer(tokenizer, new_ids)

    print(f"\n{'='*70}")
    print(f"Generated {num_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)")
    print(f"{'='*70}")

    if thinking:
        print(f"\n[Thinking]\n{thinking}")
    if content:
        print(f"\n[Answer]\n{content}")
    elif not thinking:
        full_text = tokenizer.decode(new_ids, skip_special_tokens=True)
        print(f"\n[Output]\n{full_text}")

    # Save result to JSON
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "eval_result", args.output_name, "single")
    os.makedirs(out_dir, exist_ok=True)
    import json
    result_path = os.path.join(out_dir, f"q_{int(time.time())}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "question": args.question,
            "thinking_content": thinking,
            "content": content,
            "num_tokens": num_tokens,
            "elapsed_s": round(elapsed, 2),
            "tok_per_s": round(tps, 2),
            "method": args.method,
            "rank": args.rank,
            "seed": args.seed,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResult saved to: {result_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_dataset_eval(dataset_name: str, dataset: List[dict],
                     model, tokenizer, args, output_dir: str):
    """Evaluate on a dataset with checkpoint resumption."""
    num_questions = len(dataset)
    seed_list = list(range(args.seed, args.seed + args.repeat_num))

    # Checkpoint
    completed_set = get_completed_set(output_dir, num_questions, args.repeat_num)
    if len(completed_set) >= num_questions:
        print(f"  [{dataset_name}] All done, skipping.")
        return
    print(f"  [{dataset_name}] Completed {len(completed_set)}/{num_questions}")

    remaining = [i for i in range(num_questions) if i not in completed_set]

    # Multi-process split
    if args.world_size > 1:
        total_q = len(remaining)
        per_rank = (total_q + args.world_size - 1) // args.world_size
        start = args.rank_proc * per_rank
        end = min(start + per_rank, total_q)
        if start >= total_q:
            print(f"  [{dataset_name}] rank {args.rank_proc} has no work")
            return
        remaining = remaining[start:end]
        print(f"  [{dataset_name}] rank {args.rank_proc}/{args.world_size}: "
              f"[{start}:{end}), {len(remaining)} questions")

    print(f"  [{dataset_name}] Remaining: {len(remaining)}, "
          f"repeat_num={args.repeat_num}")

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.top_k > 0:
            gen_kwargs["top_k"] = args.top_k
    else:
        gen_kwargs["do_sample"] = False

    # Batch processing
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(total_batches):
        batch_qs = remaining[batch_idx * args.batch_size:
                             (batch_idx + 1) * args.batch_size]

        print(f"\n  {'='*70}")
        print(f"  [{dataset_name}] batch {batch_idx+1}/{total_batches}, "
              f"{len(batch_qs)} questions")

        pending: Dict[int, List[dict]] = {}
        for q_idx in batch_qs:
            pending[q_idx] = load_partial_results(output_dir, q_idx)

        for r_idx in range(args.repeat_num):
            need_qs = [
                q_idx for q_idx in batch_qs
                if not any(r["repeat_index"] == r_idx for r in pending[q_idx])
            ]
            if not need_qs:
                continue

            print(f"\n  ---- repeat {r_idx+1}/{args.repeat_num} "
                  f"(seed={seed_list[r_idx]}) ----")
            set_seed(seed_list[r_idx])

            # Build prompts and tokenize with left-padding
            prompts = [
                build_chat_prompt(tokenizer, dataset[q_idx]["question"],
                                  args.think_mode)
                for q_idx in need_qs
            ]
            model_inputs = tokenizer(
                prompts, return_tensors="pt", padding=True,
            ).to(model.device)

            tic = time.time()
            generated_ids = model.generate(**model_inputs, **gen_kwargs)
            elapsed = time.time() - tic
            num_new = generated_ids.shape[1] - model_inputs.input_ids.shape[1]
            print(f"  Inference: {elapsed:.1f}s, {len(need_qs)} questions, "
                  f"~{num_new} new tokens/sample")

            padded_len = model_inputs.input_ids.shape[1]
            for task_i, q_idx in enumerate(need_qs):
                output_ids = generated_ids[task_i][padded_len:].tolist()
                thinking, content = split_thinking_and_answer(tokenizer, output_ids)
                extracted = extract_answer(content or thinking, dataset_name)

                thinking_preview = (thinking[:120] + "...") if len(thinking) > 120 else (thinking or "")
                content_preview = (content[:200] + "...") if len(content) > 200 else content
                print(f"  Q{q_idx} r{r_idx+1}: think={thinking_preview[:60]}... "
                      f"ans={content_preview[:80]}")
                if extracted:
                    print(f"         extracted={str(extracted)[:80]}")

                pending[q_idx].append({
                    "repeat_index": r_idx,
                    "seed": seed_list[r_idx],
                    "output_ids": output_ids,
                    "thinking_content": thinking,
                    "content": content,
                    "extracted_answer": extracted,
                })

            # Save after each repeat
            for q_idx in batch_qs:
                results_so_far = sorted(pending[q_idx],
                                        key=lambda r: r["repeat_index"])
                save_result(
                    output_dir=output_dir,
                    idx=q_idx,
                    dataset_name=dataset_name,
                    question=dataset[q_idx]["question"],
                    gold_answer=dataset[q_idx]["answer"],
                    repeat_num=args.repeat_num,
                    results=results_so_far,
                )

        # Batch accuracy
        correct, total_with_ans = 0, 0
        for q_idx in batch_qs:
            gold = dataset[q_idx]["answer"]
            if gold is None:
                continue
            extracted_answers = [
                r.get("extracted_answer") for r in pending[q_idx]
                if r.get("extracted_answer") is not None
            ]
            if not extracted_answers:
                continue
            total_with_ans += 1
            if any(ans == gold for ans in extracted_answers):
                correct += 1
        if total_with_ans:
            print(f"\n  [{dataset_name}] batch {batch_idx+1} "
                  f"accuracy: {correct}/{total_with_ans} "
                  f"({100*correct/total_with_ans:.1f}%)")

    print(f"\n  [{dataset_name}] Done. Results: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="ARHQ/SVDQuant simulated nvfp4 quantized inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Method
    parser.add_argument("--method", type=str, default="arhq",
                        choices=["arhq", "r_only", "svdquant", "baseline", "original"],
                        help="Quantization method. `r_only` is a backward-compatible alias for ARHQ.")
    parser.add_argument("--setting", type=str, default="smoothing",
                        choices=["raw", "smoothing"],
                        help="Saved decomposition setting for low-rank methods")
    parser.add_argument("--rank", type=int, default=128,
                        help="Low-rank dimension (ignored for baseline/original)")
    parser.add_argument("--module_set", type=str, default="attn",
                        choices=["attn", "ffn", "all"],
                        help="Which linear projections to quantize: attention only, FFN only, or both")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--decomp_dir", type=str, default=DEFAULT_DECOMP_DIR,
                        help="Directory containing layer_{i}/{proj}_{method}_{setting}_rank{rank}.pt")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Inference mode: single question
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to answer (mutually exclusive with --datasets)")

    # Inference mode: dataset evaluation
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=SUPPORTED_DATASETS, metavar="DS",
                        help=f"Datasets to evaluate. Options: {SUPPORTED_DATASETS}")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--gpqa_subset", type=str, default="gpqa_diamond")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Questions per batch (transformers generate, not vLLM)")
    parser.add_argument("--repeat_num", type=int, default=1)
    parser.add_argument("--output_base", type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))), "eval_result"))
    parser.add_argument("--output_name", type=str, default=None)

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=52768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--think_mode", type=str, default="think",
                        choices=["think", "no-think"])
    parser.add_argument("--seed", type=int, default=0)

    # Multi-process
    parser.add_argument("--rank_id", type=int, default=0, dest="rank_proc")
    parser.add_argument("--world_size", type=int, default=1)

    args = parser.parse_args()

    if args.question is None and args.datasets is None:
        parser.error("Must specify --question or --datasets")

    return args


def main():
    args = parse_args()

    set_seed(args.seed)

    # Auto output name
    if args.output_name is None:
        if args.method in ("baseline", "original"):
            args.output_name = args.method
        else:
            method_name = "arhq" if args.method == "r_only" else args.method
            args.output_name = f"{method_name}_{args.setting}_rank{args.rank}"
        if args.method != "original" and args.module_set != "attn":
            args.output_name = f"{args.output_name}_{args.module_set}"

    # Print config
    print("=" * 70)
    print("ARHQ Quantized Inference")
    print(f"  Method:     {args.method}")
    if args.method not in ("baseline", "original"):
        print(f"  Setting:    {args.setting}")
        print(f"  Rank:       {args.rank}")
    if args.method != "original":
        print(f"  Module set: {args.module_set}")
        if args.method not in ("baseline",):
            print(f"  Decomp dir: {args.decomp_dir}")
    print(f"  Model:      {args.model_path}")
    print(f"  Device:     {args.device}")
    print(f"  Think mode: {args.think_mode}")
    if args.question:
        print(f"  Mode:       Single question")
    else:
        print(f"  Mode:       Dataset eval ({args.datasets})")
        print(f"  Batch size: {args.batch_size}")
    print(f"  Generation: max_new_tokens={args.max_new_tokens}, "
          f"temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    print("=" * 70)

    # Load model
    model, tokenizer = load_quantized_model(
        args.method, args.setting, args.rank, args.device, args.module_set,
        args.decomp_dir, args.model_path
    )

    if args.question:
        # Single question mode
        run_single_question(model, tokenizer, args)
    else:
        # Dataset evaluation mode
        for ds_name in args.datasets:
            print(f"\n{'='*70}")
            print(f"Dataset: {ds_name}")

            dataset = load_dataset(
                name=ds_name,
                think_mode=(args.think_mode == "think"),
                data_dir=args.data_dir,
                max_samples=args.max_samples,
                gpqa_subset=args.gpqa_subset,
            )
            print(f"  Samples: {len(dataset)}")

            output_dir = os.path.join(args.output_base, args.output_name, ds_name)
            os.makedirs(output_dir, exist_ok=True)

            run_dataset_eval(ds_name, dataset, model, tokenizer, args, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
