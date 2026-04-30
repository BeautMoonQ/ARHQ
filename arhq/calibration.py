"""Build calibration tensors for ARHQ/SVDQuant decomposition.

The calibration input is a directory of generation result JSON files, where
each file contains:

* ``question``
* ``results[*].output_ids``

For each requested transformer layer, this script replays prompt + generated
tokens through the model, hooks selected linear modules, and saves the linear
input activations plus current model weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Iterable

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_MODULES = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {
    "attn": ATTN_MODULES,
    "ffn": FFN_MODULES,
    "all": ATTN_MODULES + FFN_MODULES,
}


def parse_int_set(spec: str, upper: int | None = None) -> list[int]:
    if not spec:
        if upper is None:
            raise ValueError("empty range requires upper")
        return list(range(upper))
    vals: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            vals.extend(range(int(start), int(end) + 1))
        else:
            vals.append(int(part))
    return vals


def select_modules(module_set: str, target_modules: str) -> list[str]:
    if module_set == "custom":
        modules = [m.strip() for m in target_modules.split(",") if m.strip()]
        if not modules:
            raise ValueError("--target_modules is empty")
        return modules
    return list(MODULE_SETS[module_set])


def build_prompt_ids(tokenizer, question: str) -> list[int]:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, add_special_tokens=False)["input_ids"]


def load_generation_ids(path: str, repeat_index: int) -> tuple[str, list[int]] | None:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    question = payload.get("question") or ""
    results = payload.get("results") or []
    output_ids = None
    for item in results:
        if item.get("repeat_index") == repeat_index:
            output_ids = item.get("output_ids")
            break
    if output_ids is None and results:
        output_ids = results[0].get("output_ids")
    if not question or not output_ids:
        return None
    return question, list(output_ids)


def register_layer_hooks(model, layer_idx: int, module_names: Iterable[str],
                         max_tokens: int):
    prefix = f"model.layers.{layer_idx}."
    target = set(module_names)
    acts: dict[str, list[torch.Tensor]] = {}
    counts = {name: 0 for name in target}
    hooks = []

    def make_hook(short_name: str):
        def hook_fn(_module, inputs, _output):
            x = inputs[0].detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            if max_tokens > 0:
                remain = max_tokens - counts[short_name]
                if remain <= 0:
                    return
                x = x[:remain]
            if x.numel() == 0:
                return
            acts.setdefault(short_name, []).append(x.cpu())
            counts[short_name] += x.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name not in target:
            continue
        hooks.append(module.register_forward_hook(make_hook(short_name)))

    return hooks, acts, counts


def extract_layer_weights(model, layer_idx: int, module_names: Iterable[str]):
    prefix = f"model.layers.{layer_idx}."
    target = set(module_names)
    weights = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith(prefix):
            continue
        short_name = name.rsplit(".", 1)[-1]
        if short_name in target:
            weights[short_name] = module.weight.detach().float().cpu()
    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build layer-wise calibration activations and weights."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--result_dir", required=True,
                        help="Directory containing eval result JSON files.")
    parser.add_argument("--output_dir", required=True,
                        help="Calibration output directory.")
    parser.add_argument("--repeat_index", type=int, default=0)
    parser.add_argument("--sample_range", default="0-127",
                        help="JSON file indices, e.g. 0-127 or 0,3,5.")
    parser.add_argument("--layers", default="",
                        help="Layer indices. Empty means all layers.")
    parser.add_argument("--module_set", default="attn",
                        choices=["attn", "ffn", "all", "custom"])
    parser.add_argument("--target_modules", default=",".join(ATTN_MODULES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--max_seq_len", type=int, default=32768)
    parser.add_argument("--max_layer_tokens", type=int, default=30000,
                        help="Stop replaying a layer after every hooked linear reaches this many tokens.")
    return parser.parse_args()


def main():
    args = parse_args()
    module_names = select_modules(args.module_set, args.target_modules)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    json_files = sorted(glob.glob(os.path.join(args.result_dir, "*.json")))
    if not json_files:
        raise FileNotFoundError(f"no JSON files found in {args.result_dir}")
    sample_indices = parse_int_set(args.sample_range)
    json_files = [json_files[i] for i in sample_indices if i < len(json_files)]
    if not json_files:
        raise ValueError("sample range selects no JSON files")

    print(f"Model:       {args.model_path}")
    print(f"Results:     {args.result_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Samples:     {len(json_files)}")
    print(f"Module set:  {args.module_set} {module_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    full_sequences = []
    for path in json_files:
        sample = load_generation_ids(path, args.repeat_index)
        if sample is None:
            continue
        question, output_ids = sample
        ids = build_prompt_ids(tokenizer, question) + output_ids
        full_sequences.append(ids[:args.max_seq_len])
    if not full_sequences:
        raise ValueError("no valid calibration sequences loaded")

    num_layers = model.config.num_hidden_layers
    layers = parse_int_set(args.layers, upper=num_layers)
    os.makedirs(args.output_dir, exist_ok=True)

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}")
        hooks, act_store, counts = register_layer_hooks(
            model, layer_idx, module_names, args.max_layer_tokens
        )
        if not hooks:
            print("  no hooks registered; skipped")
            continue

        for seq_idx, ids in enumerate(full_sequences):
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                model(input_ids)
            if args.max_layer_tokens > 0 and all(
                counts[name] >= args.max_layer_tokens for name in module_names
            ):
                print(f"  reached max_layer_tokens={args.max_layer_tokens}")
                break
            if (seq_idx + 1) % 20 == 0 or seq_idx + 1 == len(full_sequences):
                stats = ", ".join(f"{k}={counts.get(k, 0)}" for k in module_names)
                print(f"  replay {seq_idx + 1}/{len(full_sequences)}: {stats}")

        for hook in hooks:
            hook.remove()

        activations = {
            name: torch.cat(tensors, dim=0)
            for name, tensors in act_store.items()
            if tensors
        }
        weights = extract_layer_weights(model, layer_idx, module_names)
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        torch.save(activations, os.path.join(layer_dir, "activations.pt"))
        torch.save(activations, os.path.join(layer_dir, "activations_truncated.pt"))
        torch.save(weights, os.path.join(layer_dir, "weights.pt"))
        for name in module_names:
            a_shape = tuple(activations[name].shape) if name in activations else None
            w_shape = tuple(weights[name].shape) if name in weights else None
            print(f"  {name}: act={a_shape}, weight={w_shape}")
        torch.cuda.empty_cache()

    meta = {
        "model_path": args.model_path,
        "result_dir": args.result_dir,
        "repeat_index": args.repeat_index,
        "sample_range": args.sample_range,
        "num_sequences": len(full_sequences),
        "layers": layers,
        "module_set": args.module_set,
        "target_modules": module_names,
        "max_seq_len": args.max_seq_len,
        "max_layer_tokens": args.max_layer_tokens,
    }
    with open(os.path.join(args.output_dir, "calibration_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
