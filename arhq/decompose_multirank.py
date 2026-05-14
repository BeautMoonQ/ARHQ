"""Decompose all 36 layers for multiple ranks in a single pass per (layer, proj).

For each (layer, proj, method, setting):
  1. Load activations & weight, compute SmoothQuant scale (if smoothing).
  2. Run the full-rank decomposition once (eigendecomp + SVD).
  3. Truncate to each target rank in fp32 and save independent .pt files.

This is mathematically equivalent to running the existing single-rank
decompose script once per rank, but ~4x faster (no repeated eigendecomp/SVD)
and avoids fp16 accumulation error from post-hoc extraction.

Usage:
  conda run -n llmc python -m arhq.decompose_multirank \\
      --calib_dir ~/work/data/calib/vllm_MATH-500/samples_0000 \\
      --output_dir results/layer_results \\
      --device cuda:0 --layers 0-8 --module_set all \\
      --ranks 16 32 64 128 \\
      --configs arhq:raw arhq:smoothing svdquant:smoothing
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Iterable

import torch

from arhq.lowrank import decompose_multirank, compute_snr
from arhq.quant import nvfp4_quantize
from arhq.transforms import search_best_alpha

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {"attn": ATTN, "ffn": FFN, "all": ATTN + FFN}
DEFAULT_CONFIGS = ["arhq:raw", "arhq:smoothing", "svdquant:smoothing"]


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return list(range(36))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def parse_configs(specs: Iterable[str]) -> list[tuple[str, str]]:
    items = list(specs) if specs else DEFAULT_CONFIGS
    out = []
    for item in items:
        method, setting = item.split(":", 1)
        if method not in ("arhq", "r_only", "svdquant"):
            raise ValueError(f"unknown method: {method}")
        if setting not in ("raw", "smoothing"):
            raise ValueError(f"unknown setting: {setting}")
        out.append((method, setting))
    return out


def load_projection(calib_dir: str, layer_idx: int, proj: str,
                    eval_tokens: int, device: str):
    layer_dir = os.path.join(calib_dir, f"layer_{layer_idx}")
    activations = torch.load(
        os.path.join(layer_dir, "activations_truncated.pt"),
        map_location="cpu", weights_only=True,
    )
    weights = torch.load(
        os.path.join(layer_dir, "weights.pt"),
        map_location="cpu", weights_only=True,
    )
    if proj not in activations or proj not in weights:
        return None
    acts = activations[proj].float()
    if acts.shape[0] <= eval_tokens:
        raise ValueError(
            f"L{layer_idx} {proj}: tokens {acts.shape[0]} <= eval_tokens {eval_tokens}"
        )
    return (
        acts[-eval_tokens:].to(device),
        weights[proj].float().to(device),
        acts[:-eval_tokens].to(device),
    )


def decomp_path(output_dir: str, layer_idx: int, proj: str,
                method: str, setting: str, rank: int) -> str:
    return os.path.join(
        output_dir, f"layer_{layer_idx}",
        f"{proj}_{method}_{setting}_rank{rank}.pt",
    )


def save_decomp(path: str, dec: dict, *, proj: str, method: str, setting: str,
                rank: int, module_set: str, alpha, scale,
                snr_baseline_db: float, snr_method_db: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "B_r": dec["B_r"].half().cpu(),
        "A_fac": dec["A_fac"].half().cpu(),
        "W_res": dec["W_res"].half().cpu(),
        "rank": rank,
        "method": method,
        "setting": setting,
        "proj": proj,
        "module_set": module_set,
        "metric": dec.get("metric",
                          "activation_residual" if method in ("arhq", "r_only")
                          else "plain_svd"),
        "snr_baseline_db": round(snr_baseline_db, 4),
        "snr_method_db": round(snr_method_db, 4),
    }
    if scale is not None:
        payload["scale"] = scale.half().cpu()
        payload["alpha"] = alpha
    torch.save(payload, path)


@torch.no_grad()
def evaluate_snr(A_eval, W_orig, A_calib, scale, decomp_fp32):
    """Compute baseline + method SNR for one decomp (fp32 in/out)."""
    Y_true = A_eval @ W_orig.T
    if scale is not None:
        A_e = A_eval / scale
        Wt = W_orig * scale
    else:
        A_e = A_eval
        Wt = W_orig
    Y_base = nvfp4_quantize(A_e) @ nvfp4_quantize(Wt).T
    snr_base = compute_snr(Y_true, Y_base)
    B_r = decomp_fp32["B_r"]
    A_fac = decomp_fp32["A_fac"]
    W_res = decomp_fp32["W_res"]
    Y_hat = nvfp4_quantize(A_e) @ nvfp4_quantize(W_res).T + (A_e @ A_fac) @ B_r.T
    snr_method = compute_snr(Y_true, Y_hat)
    return snr_base, snr_method


@torch.no_grad()
def run(args):
    layers = parse_layers(args.layers)
    projs = MODULE_SETS[args.module_set]
    configs = parse_configs(args.configs)
    ranks = sorted(set(args.ranks))

    os.makedirs(args.summary_dir, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(args.summary_dir,
                            f"decompose_multirank{suffix}.csv")

    print(f"Calib:   {args.calib_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Layers:  {layers}")
    print(f"Projs:   {projs}")
    print(f"Configs: {configs}")
    print(f"Ranks:   {ranks}")

    rows = []
    t0 = time.time()
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "layer", "proj", "rank", "method", "setting",
        "snr_baseline_db", "snr_method_db", "snr_improvement_db", "alpha",
    ])

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}")
        layer_t0 = time.time()
        for proj in projs:
            loaded = load_projection(
                args.calib_dir, layer_idx, proj, args.eval_tokens, args.device
            )
            if loaded is None:
                print(f"  {proj}: SKIP (not present)")
                continue
            A_eval, W_orig, A_calib = loaded

            # Best alpha once per (layer, proj); reused across configs+ranks
            best_alpha, best_scale = search_best_alpha(A_calib, W_orig)

            for method, setting in configs:
                # Skip group if all (rank) files already exist
                if args.skip_existing:
                    if all(os.path.exists(decomp_path(
                            args.output_dir, layer_idx, proj, method, setting, r))
                           for r in ranks):
                        print(f"  {proj} {method}:{setting}: SKIP (all ranks exist)")
                        continue

                scale = best_scale if setting == "smoothing" else None
                if scale is not None:
                    A_c = A_calib / scale
                    Wt = W_orig * scale
                else:
                    A_c = A_calib
                    Wt = W_orig

                # Single decomposition for all ranks
                decomps = decompose_multirank(method, Wt, A_c, ranks)

                for r in ranks:
                    dec_r = decomps[r]
                    snr_base, snr_method = evaluate_snr(
                        A_eval, W_orig, A_calib, scale, dec_r
                    )
                    path = decomp_path(args.output_dir, layer_idx, proj,
                                       method, setting, r)
                    save_decomp(
                        path, dec_r,
                        proj=proj, method=method, setting=setting,
                        rank=r, module_set=args.module_set,
                        alpha=best_alpha if scale is not None else None,
                        scale=scale,
                        snr_baseline_db=snr_base,
                        snr_method_db=snr_method,
                    )
                    writer.writerow([
                        layer_idx, proj, r, method, setting,
                        round(snr_base, 4), round(snr_method, 4),
                        round(snr_method - snr_base, 4),
                        best_alpha if scale is not None else "",
                    ])
                    csv_f.flush()
                    rows.append((layer_idx, proj, r, method, setting,
                                 snr_method))

                tail = rows[-len(ranks):]
                snrs_str = " ".join(f"r{rec[2]}={rec[-1]:.2f}" for rec in tail)
                print(f"  {proj:11s} {method}:{setting:9s}  {snrs_str}")

                del decomps
            del A_eval, W_orig, A_calib, best_scale
            torch.cuda.empty_cache()
        print(f"  layer {layer_idx} done in {time.time() - layer_t0:.1f}s")

    csv_f.close()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    print(f"Summary CSV: {csv_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib_dir", default=os.path.expanduser(
        "~/work/data/calib/vllm_MATH-500/samples_0000"))
    ap.add_argument("--output_dir", default="results/layer_results")
    ap.add_argument("--summary_dir", default="results/summary")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--layers", default="0-35")
    ap.add_argument("--module_set", default="all", choices=["attn", "ffn", "all"])
    ap.add_argument("--ranks", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--eval_tokens", type=int, default=2048)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip_existing", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
