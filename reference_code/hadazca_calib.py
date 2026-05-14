"""
hadazca_calib.py
----------------
逐层收集目标 Linear 层的输入激活值和权重，供 HadaZCA / ARHQ 使用。

基于 calib.py 的数据重建逻辑，逐层处理以控制内存：
  1. 加载模型和 eval result JSON
  2. 对每个 transformer layer，注册目标 linear 的 hook
  3. Forward pass 收集激活
  4. 提取对应权重
  5. 按层保存到磁盘

保存格式:
  {calib_dir}/{precision}_{dataset}[_save_tag]/samples_{start}/layer_{i}/
    ├── activations.pt   # { "q_proj": [N_tokens, D], ... } 或 { "gate_proj": ... }
    ├── activations_truncated.pt  # 同上，供 ARHQ 直接使用
    └── weights.pt       # { "q_proj": [D_out, D], ... } 或 { "gate_proj": ... }

用法:
  # 处理 sample 0~31，所有层
  python claude/hadazca_calib.py --sample_range 0-31

  # 处理 sample 32~63，只处理第 0 层
  python claude/hadazca_calib.py --sample_range 32-63 --layers 0

  # 处理 sample 0~127（默认），第 0~3 层
  python claude/hadazca_calib.py --sample_range 0-127 --layers 0-3
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MODEL_PATH = os.path.expanduser("/home/wangyifeng/work/models/Qwen3-4B-Thinking-2507")
DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(__file__), "eval_result")
DEFAULT_CALIB_DIR  = "/home/wangyifeng/work/data/calib"
ATTN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_TARGET_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_TARGET_MODULES = ATTN_TARGET_MODULES + FFN_TARGET_MODULES


def parse_args():
    parser = argparse.ArgumentParser(
        description="逐层收集目标 linear 的激活值和权重，供 HadaZCA / ARHQ 使用"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--precision", type=str, default="vllm")
    parser.add_argument("--dataset", type=str, default="MATH-500")
    parser.add_argument("--repeat_index", type=int, default=0)
    parser.add_argument("--sample_range", type=str, default="0-127",
                        help="样本索引范围，如 '0-31' 或 '32-63'")
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--max_layer_tokens", type=int, default=-1,
                        help="每层每个 linear 最多收集多少 token；<=0 表示不限制")
    parser.add_argument("--calib_dir", type=str, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layers", type=str, default="",
                        help="要处理的层，如 '0' 或 '0-3' 或 '0,2,5'，空=全部")
    parser.add_argument("--module_set", type=str, default="attn",
                        choices=["attn", "ffn", "all", "custom"],
                        help="目标模块预设: attn=q/k/v/o, ffn=gate/up/down, all=attention+ffn, custom=使用 --target_modules")
    parser.add_argument("--target_modules", type=str,
                        default=",".join(ATTN_TARGET_MODULES),
                        help="目标模块，逗号分隔；当 --module_set=custom 时使用")
    parser.add_argument("--save_tag", type=str, default="",
                        help="保存目录后缀，如 'ffn'，会写入 {precision}_{dataset}_{save_tag}")
    return parser.parse_args()


def parse_range(range_str):
    """解析范围字符串如 '0-31'，返回 (start, end)，end 为闭区间。"""
    parts = range_str.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start
    return start, end


def parse_layer_range(layers_str, num_layers):
    """解析层范围字符串，返回层索引列表。"""
    if not layers_str:
        return list(range(num_layers))
    if "-" in layers_str and "," not in layers_str:
        parts = layers_str.split("-")
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(x.strip()) for x in layers_str.split(",")]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建（与 calib.py 一致）
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_ids(tokenizer, question: str) -> list:
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt_text, add_special_tokens=False)["input_ids"]


def load_sample(fp: str, repeat_index: int):
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)
    question = data.get("question", "")
    results = data.get("results", [])
    output_ids = None
    for r in results:
        if r.get("repeat_index") == repeat_index:
            output_ids = r.get("output_ids")
            break
    if output_ids is None and results:
        output_ids = results[0].get("output_ids")
    if not question or not output_ids:
        return None
    return question, output_ids


# ─────────────────────────────────────────────────────────────────────────────
# Hook 注册
# ─────────────────────────────────────────────────────────────────────────────

def register_hooks_for_layer(model, layer_idx, target_modules, max_layer_tokens: int = -1):
    """对指定 transformer layer 的目标模块注册 hook。"""
    # Keep the trailing dot so layer 1 does not accidentally match 10-19.
    prefix = f"model.layers.{layer_idx}."
    activation_store = {}
    token_counts = {mod_name: 0 for mod_name in target_modules}
    hooks = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        matched = False
        for mod_name in target_modules:
            if name.endswith(mod_name):
                matched = True
                break
        if not matched:
            continue

        short_name = name.split(".")[-1]  # e.g. "q_proj"

        def make_hook(sname):
            def hook_fn(mod, inp, out):
                x = inp[0].detach().float()
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])  # (seq_len, D)
                if max_layer_tokens > 0:
                    remaining = max_layer_tokens - token_counts[sname]
                    if remaining <= 0:
                        return
                    if x.shape[0] > remaining:
                        x = x[:remaining]
                if x.numel() == 0:
                    return
                activation_store.setdefault(sname, []).append(x.cpu())
                token_counts[sname] += x.shape[0]
            return hook_fn

        h = module.register_forward_hook(make_hook(short_name))
        hooks.append(h)

    return hooks, activation_store, token_counts


def extract_weights_for_layer(model, layer_idx, target_modules):
    """提取指定 transformer layer 的目标模块权重。"""
    # Keep the trailing dot so layer 2 does not accidentally match 20-29.
    prefix = f"model.layers.{layer_idx}."
    weights = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        for mod_name in target_modules:
            if name.endswith(mod_name):
                weights[mod_name] = module.weight.detach().float().cpu()
                break
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.module_set == "attn":
        target_modules = list(ATTN_TARGET_MODULES)
    elif args.module_set == "ffn":
        target_modules = list(FFN_TARGET_MODULES)
    elif args.module_set == "all":
        target_modules = list(ALL_TARGET_MODULES)
    else:
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    # 解析 sample range
    sample_start, sample_end = parse_range(args.sample_range)
    print(f"样本范围: [{sample_start}, {sample_end}]")
    print(f"目标模块: {target_modules}")

    # 1. 找 JSON 文件
    pattern = os.path.join(args.result_dir, args.precision, args.dataset, "*.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        print(f"[ERROR] 未找到 JSON 文件: {pattern}")
        sys.exit(1)

    # 按 sample_range 切片
    if sample_end + 1 > len(json_files):
        print(f"[WARNING] sample_end={sample_end} 超过文件数 {len(json_files)}，截断")
        sample_end = len(json_files) - 1
    json_files = json_files[sample_start:sample_end + 1]
    print(f"使用 {len(json_files)} 个 JSON 文件 (idx {sample_start}~{sample_end})")

    # 2. 加载 tokenizer + 模型
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  模型内存: {model.get_memory_footprint() / 1e9:.2f} GB")
    
    # 3. 预构建所有 full_ids（只做一次 tokenize）
    print("\n构建 token 序列...")
    all_full_ids = []
    for fp in json_files:
        sample = load_sample(fp, args.repeat_index)
        if sample is None:
            continue
        question, output_ids = sample
        prompt_ids = build_prompt_ids(tokenizer, question)
        full_ids = prompt_ids + list(output_ids)
        if len(full_ids) > args.max_seq_len:
            full_ids = full_ids[:args.max_seq_len]
        all_full_ids.append(full_ids)
    print(f"  有效样本: {len(all_full_ids)}, 总 token: {sum(len(s) for s in all_full_ids):,}")

    # 4. 确定层范围
    num_layers = model.config.num_hidden_layers
    layer_indices = parse_layer_range(args.layers, num_layers)
    print(f"\n将处理 {len(layer_indices)} 层: {layer_indices}")

    calib_dir = os.path.expanduser(args.calib_dir)
    dataset_dir = f"{args.precision}_{args.dataset}"
    if args.save_tag:
        dataset_dir = f"{dataset_dir}_{args.save_tag}"
    # 保存路径包含 samples_{start}
    base_dir = os.path.join(calib_dir, dataset_dir, f"samples_{sample_start:04d}")
    device = next(model.parameters()).device

    # 5. 逐层处理
    for layer_idx in layer_indices:
        print(f"\n{'='*60}")
        print(f"处理 layer {layer_idx}")
        print(f"{'='*60}")

        # 注册 hook
        hooks, activation_store, token_counts = register_hooks_for_layer(
            model, layer_idx, target_modules, args.max_layer_tokens
        )
        print(f"  注册了 {len(hooks)} 个 hook")

        if len(hooks) == 0:
            print(f"  [SKIP] layer {layer_idx} 未找到目标模块")
            continue

        # Forward pass
        for i, full_ids in enumerate(all_full_ids):
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_tensor)
            if args.max_layer_tokens > 0:
                done = all(token_counts.get(mod_name, 0) >= args.max_layer_tokens
                           for mod_name in target_modules)
                if done:
                    print(f"  已达到每个 linear 约 {args.max_layer_tokens} token，提前结束该层")
                    break
            if (i + 1) % 20 == 0 or i == len(all_full_ids) - 1:
                counts_str = ", ".join(
                    f"{mod_name}={token_counts.get(mod_name, 0)}"
                    for mod_name in target_modules
                )
                print(f"  forward: {i+1}/{len(all_full_ids)}  ({counts_str})")

        # 移除 hook
        for h in hooks:
            h.remove()

        # 合并激活
        activations = {}
        for mod_name, tensor_list in activation_store.items():
            if tensor_list:
                activations[mod_name] = torch.cat(tensor_list, dim=0)
                print(f"  {mod_name} 激活: {tuple(activations[mod_name].shape)}")
        activations_truncated = {
            mod_name: acts
            for mod_name, acts in activations.items()
        }

        # 提取权重
        weights = extract_weights_for_layer(model, layer_idx, target_modules)
        for mod_name, w in weights.items():
            print(f"  {mod_name} 权重: {tuple(w.shape)}")

        # 保存
        save_dir = os.path.join(base_dir, f"layer_{layer_idx}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(activations, os.path.join(save_dir, "activations.pt"))
        torch.save(activations_truncated, os.path.join(save_dir, "activations_truncated.pt"))
        torch.save(weights, os.path.join(save_dir, "weights.pt"))
        print(f"  已保存到 {save_dir}")
        if token_counts:
            print("  token 统计: " + ", ".join(
                f"{mod_name}={token_counts.get(mod_name, 0)}"
                for mod_name in target_modules
            ))

        # 释放内存
        del token_counts, activation_store, activations, activations_truncated, weights
        torch.cuda.empty_cache()

    # 6. 保存 meta
    meta = {
        "model_path": args.model_path,
        "precision": args.precision,
        "dataset": args.dataset,
        "sample_range": [sample_start, sample_end],
        "num_samples": len(all_full_ids),
        "total_tokens": sum(len(s) for s in all_full_ids),
        "max_seq_len": args.max_seq_len,
        "max_layer_tokens": args.max_layer_tokens,
        "layers": layer_indices,
        "module_set": args.module_set,
        "target_modules": target_modules,
        "save_tag": args.save_tag,
    }
    meta_path = os.path.join(base_dir, "hadazca_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nmeta 保存到: {meta_path}")
    print("校准数据收集完成。")


if __name__ == "__main__":
    main()
