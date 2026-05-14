#!/usr/bin/env python3
"""Check whether saved inference decomposition matches current model weights.

For raw setting, we check:
    W_orig ?= W_res + B_r @ A_fac^T

For smoothing setting, inference actually uses transformed weights, so we check:
    W_orig * scale ?= W_res + B_r @ A_fac^T

This script reads only the needed projection weights from safetensors shards
instead of loading the whole model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Iterable

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: safetensors. Run this script in an environment "
        "with safetensors installed, e.g. `conda run -n llmc python ...`."
    ) from exc

from arhq.quant import nvfp4_quantize


DEFAULT_MODEL_PATH = os.path.expanduser("~/work/models/Qwen3-4B-Thinking-2507")
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
    "layer_results",
)
ATTN_PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_PROJ_TYPES = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {
    "attn": ATTN_PROJ_TYPES,
    "ffn": FFN_PROJ_TYPES,
    "all": ATTN_PROJ_TYPES + FFN_PROJ_TYPES,
}
NUM_LAYERS = 36


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return list(range(NUM_LAYERS))
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",") if x]


def parse_projs(spec: str) -> list[str]:
    if not spec:
        return ATTN_PROJ_TYPES
    projs = [x.strip() for x in spec.split(",") if x.strip()]
    valid = MODULE_SETS["all"]
    invalid = [x for x in projs if x not in valid]
    if invalid:
        raise ValueError(f"Invalid projections: {invalid}, allowed: {valid}")
    return projs


def projs_from_module_set(module_set: str) -> list[str]:
    if module_set not in MODULE_SETS:
        raise ValueError(f"Invalid module_set={module_set}, allowed={sorted(MODULE_SETS)}")
    return MODULE_SETS[module_set]


def load_weight_map(model_path: str) -> dict[str, str]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["weight_map"]


def load_tensor_from_shard(model_path: str, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard_name = weight_map[key]
    shard_path = os.path.join(model_path, shard_name)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def decomp_path(results_dir: str, layer_idx: int, proj: str, method: str,
                setting: str, rank: int) -> str:
    fname = f"{proj}_{method}_{setting}_rank{rank}.pt"
    return os.path.join(results_dir, f"layer_{layer_idx}", fname)


def target_weight(W_orig: torch.Tensor, decomp: dict, setting: str) -> torch.Tensor:
    if setting == "smoothing":
        if "scale" not in decomp:
            raise ValueError("Smoothing check requested but saved decomposition has no scale")
        return W_orig * decomp["scale"].float()
    return W_orig


def format_metric(x: float) -> str:
    return f"{x:.6e}"


def iter_checks(model_path: str, results_dir: str, method: str, setting: str,
                rank: int, layers: Iterable[int], projs: Iterable[str]):
    weight_map = load_weight_map(model_path)
    for layer_idx in layers:
        for proj in projs:
            if proj in ATTN_PROJ_TYPES:
                key = f"model.layers.{layer_idx}.self_attn.{proj}.weight"
            else:
                key = f"model.layers.{layer_idx}.mlp.{proj}.weight"
            path = decomp_path(results_dir, layer_idx, proj, method, setting, rank)
            if not os.path.exists(path):
                yield {
                    "layer": layer_idx,
                    "proj": proj,
                    "path": path,
                    "missing": True,
                }
                continue

            W_orig = load_tensor_from_shard(model_path, weight_map, key).float()
            decomp = torch.load(path, map_location="cpu", weights_only=True)

            W_res = decomp["W_res"].float()
            B_r = decomp["B_r"].float()
            A_fac = decomp["A_fac"].float()
            W_target = target_weight(W_orig, decomp, setting)

            W_recon = W_res + B_r @ A_fac.T
            diff = W_target - W_recon

            W_infer = nvfp4_quantize(W_res) + B_r @ A_fac.T
            diff_infer = W_target - W_infer

            denom = W_target.norm().clamp(min=1e-12)
            yield {
                "layer": layer_idx,
                "proj": proj,
                "path": path,
                "missing": False,
                "shape": tuple(W_orig.shape),
                "max_abs_err": diff.abs().max().item(),
                "mean_abs_err": diff.abs().mean().item(),
                "rel_err": (diff.norm() / denom).item(),
                "infer_max_abs_err": diff_infer.abs().max().item(),
                "infer_rel_err": (diff_infer.norm() / denom).item(),
            }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether saved decomposition matches current model weights."
    )
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--method", default="arhq",
                        choices=["arhq", "r_only", "svdquant"])
    parser.add_argument("--setting", default="raw",
                        choices=["raw", "smoothing"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--layers", default="", help="e.g. '0-35' or '0,2,5'")
    parser.add_argument("--module_set", default="attn", choices=["attn", "ffn", "all"])
    parser.add_argument("--projs", default="", help="comma separated projections")
    parser.add_argument("--topk", type=int, default=10,
                        help="Print top-k worst cases by reconstruction error")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    projs = parse_projs(args.projs) if args.projs else projs_from_module_set(args.module_set)

    print("Checking decomposition consistency")
    print(f"  model_path:  {args.model_path}")
    print(f"  results_dir: {args.results_dir}")
    print(f"  method:      {args.method}")
    print(f"  setting:     {args.setting}")
    print(f"  rank:        {args.rank}")
    print(f"  layers:      {layers}")
    print(f"  module_set:  {args.module_set}")
    print(f"  projs:       {projs}")

    rows = list(iter_checks(
        args.model_path, args.results_dir, args.method, args.setting,
        args.rank, layers, projs
    ))

    missing = [r for r in rows if r["missing"]]
    valid = [r for r in rows if not r["missing"]]

    print(f"\nFound {len(valid)} decomposition files, missing {len(missing)}")
    if missing:
        for row in missing[: args.topk]:
            print(f"  MISSING  L{row['layer']:02d} {row['proj']}: {row['path']}")

    if not valid:
        return

    worst_recon = sorted(valid, key=lambda r: r["rel_err"], reverse=True)[: args.topk]
    worst_infer = sorted(valid, key=lambda r: r["infer_rel_err"], reverse=True)[: args.topk]

    max_abs = max(r["max_abs_err"] for r in valid)
    mean_rel = sum(r["rel_err"] for r in valid) / len(valid)
    max_rel = max(r["rel_err"] for r in valid)

    max_infer_abs = max(r["infer_max_abs_err"] for r in valid)
    mean_infer_rel = sum(r["infer_rel_err"] for r in valid) / len(valid)
    max_infer_rel = max(r["infer_rel_err"] for r in valid)

    print("\nSaved decomposition consistency:")
    print(f"  max_abs_err:  {format_metric(max_abs)}")
    print(f"  mean_rel_err: {format_metric(mean_rel)}")
    print(f"  max_rel_err:  {format_metric(max_rel)}")

    print("\nApproximation actually used at inference (quantized W_res):")
    print(f"  max_abs_err:  {format_metric(max_infer_abs)}")
    print(f"  mean_rel_err: {format_metric(mean_infer_rel)}")
    print(f"  max_rel_err:  {format_metric(max_infer_rel)}")

    print("\nWorst saved-decomp cases:")
    for row in worst_recon:
        print(
            f"  L{row['layer']:02d} {row['proj']:7s} "
            f"shape={row['shape']} "
            f"max_abs={format_metric(row['max_abs_err'])} "
            f"rel={format_metric(row['rel_err'])}"
        )

    print("\nWorst inference cases:")
    for row in worst_infer:
        print(
            f"  L{row['layer']:02d} {row['proj']:7s} "
            f"shape={row['shape']} "
            f"max_abs={format_metric(row['infer_max_abs_err'])} "
            f"rel={format_metric(row['infer_rel_err'])}"
        )


if __name__ == "__main__":
    main()
