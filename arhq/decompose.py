"""Extract ARHQ/SVDQuant low-rank factors from calibration tensors."""

from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from arhq.lowrank import evaluate_single
from arhq.transforms import search_best_alpha

ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_MODULES = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {
    "attn": ATTN_MODULES,
    "ffn": FFN_MODULES,
    "all": ATTN_MODULES + FFN_MODULES,
}
DEFAULT_CONFIGS = ["arhq:raw", "arhq:smoothing", "svdquant:smoothing"]


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return list(range(36))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def parse_configs(spec: str) -> list[tuple[str, str]]:
    items = DEFAULT_CONFIGS if not spec else [x.strip() for x in spec.split(",")]
    configs = []
    for item in items:
        method, setting = item.split(":", 1)
        method = method.strip()
        setting = setting.strip()
        if method not in ("arhq", "r_only", "svdquant"):
            raise ValueError(f"unknown method: {method}")
        if setting not in ("raw", "smoothing"):
            raise ValueError(f"unknown setting: {setting}")
        configs.append((method, setting))
    return configs


def select_modules(module_set: str, projs: str) -> list[str]:
    if projs:
        return [p.strip() for p in projs.split(",") if p.strip()]
    return list(MODULE_SETS[module_set])


def load_projection(calib_dir: str, layer_idx: int, proj: str,
                    eval_tokens: int, device: str):
    layer_dir = os.path.join(calib_dir, f"layer_{layer_idx}")
    activations = torch.load(
        os.path.join(layer_dir, "activations_truncated.pt"),
        map_location="cpu",
        weights_only=True,
    )
    weights = torch.load(
        os.path.join(layer_dir, "weights.pt"),
        map_location="cpu",
        weights_only=True,
    )
    if proj not in activations:
        raise KeyError(f"{proj} missing from {layer_dir}/activations_truncated.pt")
    if proj not in weights:
        raise KeyError(f"{proj} missing from {layer_dir}/weights.pt")
    acts = activations[proj].float()
    if acts.shape[0] <= eval_tokens:
        raise ValueError(
            f"L{layer_idx} {proj}: activation tokens {acts.shape[0]} <= eval_tokens {eval_tokens}"
        )
    return (
        acts[-eval_tokens:].to(device),
        weights[proj].float().to(device),
        acts[:-eval_tokens].to(device),
    )


def decomp_path(output_dir: str, layer_idx: int, proj: str,
                method: str, setting: str, rank: int) -> str:
    return os.path.join(
        output_dir,
        f"layer_{layer_idx}",
        f"{proj}_{method}_{setting}_rank{rank}.pt",
    )


def save_decomp(path: str, result: dict, proj: str, method: str, setting: str,
                rank: int, module_set: str, alpha, scale):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "B_r": result["B_r"],
        "A_fac": result["A_fac"],
        "W_res": result["W_res"],
        "rank": rank,
        "method": method,
        "setting": setting,
        "proj": proj,
        "module_set": module_set,
        "metric": "activation_residual" if method in ("arhq", "r_only") else "plain_svd",
        "beta": result.get("beta"),
    }
    if scale is not None:
        payload["scale"] = scale.half().cpu()
        payload["alpha"] = alpha
    torch.save(payload, path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ARHQ/SVDQuant LoRA factors from calibration data."
    )
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--output_dir", default="results/layer_results")
    parser.add_argument("--summary_dir", default="results/summary")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="0-35")
    parser.add_argument("--module_set", default="attn",
                        choices=["attn", "ffn", "all"])
    parser.add_argument("--projs", default="",
                        help="Optional comma-separated projections overriding --module_set.")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                        help="Comma-separated method:setting pairs.")
    parser.add_argument("--eval_tokens", type=int, default=2048)
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    projs = select_modules(args.module_set, args.projs)
    configs = parse_configs(args.configs)
    os.makedirs(args.summary_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(args.summary_dir, f"decompose_rank{args.rank}{suffix}.csv")

    print(f"Calibration: {args.calib_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Layers:      {layers}")
    print(f"Projections: {projs}")
    print(f"Configs:     {configs}")

    rows = []
    t0 = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer", "proj", "rank", "method", "setting",
            "snr_baseline_db", "snr_method_db", "snr_improvement_db", "alpha",
        ])

        for layer_idx in layers:
            print(f"\nLayer {layer_idx}")
            for proj in projs:
                A_eval, W, A_calib = load_projection(
                    args.calib_dir, layer_idx, proj, args.eval_tokens, args.device
                )
                print(
                    f"  {proj}: A_calib={tuple(A_calib.shape)} "
                    f"A_eval={tuple(A_eval.shape)} W={tuple(W.shape)}"
                )
                best_alpha, best_scale = search_best_alpha(A_calib, W)

                for method, setting in configs:
                    path = decomp_path(
                        args.output_dir, layer_idx, proj, method, setting, args.rank
                    )
                    if args.skip_existing and os.path.exists(path):
                        print(f"    skip existing {path}")
                        continue
                    scale = best_scale if setting == "smoothing" else None
                    result = evaluate_single(
                        A_eval, W, A_calib, args.rank, method, scale=scale
                    )
                    save_decomp(
                        path, result, proj, method, setting, args.rank,
                        args.module_set, best_alpha if scale is not None else None,
                        scale,
                    )
                    writer.writerow([
                        layer_idx, proj, args.rank, method, setting,
                        result["snr_baseline_db"], result["snr_method_db"],
                        result["snr_improvement_db"],
                        best_alpha if scale is not None else "",
                    ])
                    f.flush()
                    rows.append(result)
                    print(
                        f"    {method}:{setting} snr={result['snr_method_db']:.4f} "
                        f"imp={result['snr_improvement_db']:+.4f}"
                    )

                del A_eval, W, A_calib, best_scale
                torch.cuda.empty_cache()

    print(f"\nSaved summary: {csv_path}")
    print(f"Saved factors: {args.output_dir}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
