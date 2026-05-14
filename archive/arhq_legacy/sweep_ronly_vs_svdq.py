"""Legacy script: sweep all 36 layers, smooth R-only vs smooth SVDQuant.

Use ``python -m arhq.decompose --configs arhq:smoothing,svdquant:smoothing``
for the minimal codebase flow.

For each (layer, proj, rank):
  - Computes best SmoothQuant alpha
  - Runs smooth + SVDQuant  and  smooth + R_only
  - Saves decomposition params (B_r, A_fac, W_res, scale, alpha) for both
  - Reports absolute SNR (dB) for comparison

Saved .pt files contain everything needed for simulated nvfp4+lora inference:
  B_r:     [D_out, rank]  low-rank left factor
  A_fac:   [D, rank]      low-rank right factor (activation side)
  W_res:   [D_out, D]     quantized residual weight
  scale:   [D]            SmoothQuant channel scale
  alpha:   float          SmoothQuant alpha
  rank:    int
  method:  str
"""

import argparse
import csv
import os
import time

import torch

from arhq.quant import nvfp4_quantize
from arhq.lowrank import (svdquant_decompose, r_only_decompose,
                          compute_snr)
from arhq.transforms import search_best_alpha

SAMPLES_DIR = os.path.expanduser("~/work/data/calib/vllm_MATH-500/samples_0000")
EVAL_TOKENS = 2048
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
RANKS = [32, 64, 128, 256]
METHODS = ["svdquant", "r_only"]
NUM_LAYERS = 36
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_proj_to_gpu(layer_idx: int, proj: str, device: str):
    """Load one projection's data to GPU, returns (A_calib, A_eval, W)."""
    layer_dir = os.path.join(SAMPLES_DIR, f"layer_{layer_idx}")
    activations = torch.load(
        os.path.join(layer_dir, "activations_truncated.pt"),
        map_location="cpu", weights_only=True,
    )
    weights = torch.load(
        os.path.join(layer_dir, "weights.pt"),
        map_location="cpu", weights_only=True,
    )
    A_all = activations[proj].float()
    W = weights[proj].float()
    A_calib = A_all[:-EVAL_TOKENS].to(device)
    A_eval = A_all[-EVAL_TOKENS:].to(device)
    W = W.to(device)
    return A_calib, A_eval, W


@torch.no_grad()
def evaluate_and_save(A_eval, W, A_calib, rank, method, scale, alpha,
                      layer_idx, proj, save_dir):
    """Evaluate one config and save decomposition params.

    Returns dict with SNR metrics.
    """
    Y_true = A_eval @ W.T

    # Apply smoothing
    A_e = A_eval / scale
    A_c = A_calib / scale
    Wt = W * scale

    # Baseline (no low-rank)
    Y_base = nvfp4_quantize(A_e) @ nvfp4_quantize(Wt).T
    snr_base = compute_snr(Y_true, Y_base)

    # Decompose
    if method == "svdquant":
        decomp = svdquant_decompose(Wt, rank)
    elif method == "r_only":
        decomp = r_only_decompose(Wt, A_c, rank)
    else:
        raise ValueError(f"Unknown method: {method}")

    B_r = decomp["B_r"]
    A_fac = decomp["A_fac"]
    W_res = decomp["W_res"]

    # Reconstruct: Y ≈ Q(A') @ Q(W_res)^T + (A' @ A_fac) @ B_r^T
    Y_hat = nvfp4_quantize(A_e) @ nvfp4_quantize(W_res).T + (A_e @ A_fac) @ B_r.T
    snr_method = compute_snr(Y_true, Y_hat)

    # Save decomposition params (half precision for storage)
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{proj}_{method}_smoothing_rank{rank}.pt"
    save_data = {
        "B_r": B_r.half().cpu(),
        "A_fac": A_fac.half().cpu(),
        "W_res": W_res.half().cpu(),
        "scale": scale.half().cpu(),
        "alpha": alpha,
        "rank": rank,
        "method": method,
        "layer": layer_idx,
        "proj": proj,
        "snr_baseline_db": round(snr_base, 4),
        "snr_method_db": round(snr_method, 4),
    }
    torch.save(save_data, os.path.join(save_dir, fname))

    return {
        "snr_baseline_db": round(snr_base, 4),
        "snr_method_db": round(snr_method, 4),
        "snr_improvement_db": round(snr_method - snr_base, 4),
    }


def run_sweep(layers, device, tag=""):
    os.makedirs(os.path.join(RESULTS_DIR, "summary"), exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    csv_path = os.path.join(RESULTS_DIR, "summary", f"ronly_vs_svdq_all_layers{suffix}.csv")

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["layer", "proj", "rank", "method",
                     "snr_baseline_db", "snr_method_db", "snr_improvement_db",
                     "alpha"])

    all_results = []
    t0 = time.time()

    for layer_idx in layers:
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}")
        print(f"{'='*70}")
        layer_t0 = time.time()

        for proj in PROJ_TYPES:
            A_calib, A_eval, W = load_proj_to_gpu(layer_idx, proj, device)

            # Compute best smoothing scale once per (layer, proj)
            best_alpha, best_scale = search_best_alpha(A_calib, W)

            save_dir = os.path.join(RESULTS_DIR, "layer_results",
                                    f"layer_{layer_idx}")

            for rank in RANKS:
                if rank > min(W.shape[0], W.shape[1]):
                    continue

                snrs = {}
                for method in METHODS:
                    try:
                        result = evaluate_and_save(
                            A_eval, W, A_calib, rank, method,
                            best_scale, best_alpha, layer_idx, proj, save_dir
                        )
                    except Exception as e:
                        print(f"  ERROR: L{layer_idx} {proj} {method} r={rank}: {e}")
                        continue

                    snrs[method] = result["snr_method_db"]
                    row = {
                        "layer": layer_idx, "proj": proj, "rank": rank,
                        "method": method, **result, "alpha": best_alpha,
                    }
                    all_results.append(row)
                    writer.writerow([
                        layer_idx, proj, rank, method,
                        result["snr_baseline_db"], result["snr_method_db"],
                        result["snr_improvement_db"], best_alpha,
                    ])
                    csv_file.flush()

                # Print comparison
                if len(snrs) == 2:
                    diff = snrs["r_only"] - snrs["svdquant"]
                    marker = "✓" if diff > 0 else "✗"
                    print(f"  L{layer_idx:02d} {proj:7s} r={rank:3d}  "
                          f"svdq={snrs['svdquant']:6.2f}  "
                          f"r_only={snrs['r_only']:6.2f}  "
                          f"Δ={diff:+.2f} dB {marker}  "
                          f"α={best_alpha:.2f}")

            del A_calib, A_eval, W
            torch.cuda.empty_cache()

        layer_elapsed = time.time() - layer_t0
        print(f"  Layer {layer_idx} done in {layer_elapsed:.1f}s")

    csv_file.close()
    elapsed = time.time() - t0

    # ========== Summary ==========
    print(f"\n{'='*70}")
    print(f"FULL SWEEP COMPLETE: {len(all_results)} experiments in {elapsed:.1f}s")
    print(f"{'='*70}")

    print_summary(all_results)
    print(f"\nCSV saved to: {csv_path}")
    print(f"Decomposition params saved to: {os.path.join(RESULTS_DIR, 'layer_results')}/")


def print_summary(results):
    """Print comprehensive R_only vs SVDQuant comparison."""

    # 1. Average SNR by rank
    print(f"\n--- Average Absolute SNR (dB) by rank ---")
    for rank in RANKS:
        svdq = [r["snr_method_db"] for r in results
                if r["rank"] == rank and r["method"] == "svdquant"]
        ronly = [r["snr_method_db"] for r in results
                 if r["rank"] == rank and r["method"] == "r_only"]
        if svdq and ronly:
            s_avg = sum(svdq) / len(svdq)
            r_avg = sum(ronly) / len(ronly)
            diff = r_avg - s_avg
            print(f"  rank={rank:3d}  SVDQuant={s_avg:.2f}  R_only={r_avg:.2f}  "
                  f"Δ={diff:+.2f} dB")

    # 2. Per-proj breakdown for rank=128
    print(f"\n--- Per-proj breakdown (rank=128) ---")
    for proj in PROJ_TYPES:
        svdq = [r["snr_method_db"] for r in results
                if r["rank"] == 128 and r["method"] == "svdquant" and r["proj"] == proj]
        ronly = [r["snr_method_db"] for r in results
                 if r["rank"] == 128 and r["method"] == "r_only" and r["proj"] == proj]
        if svdq and ronly:
            s_avg = sum(svdq) / len(svdq)
            r_avg = sum(ronly) / len(ronly)
            diff = r_avg - s_avg
            print(f"  {proj:7s}  SVDQuant={s_avg:.2f}  R_only={r_avg:.2f}  "
                  f"Δ={diff:+.2f} dB")

    # 3. Win rate
    print(f"\n--- Win rate (R_only > SVDQuant) ---")
    for rank in RANKS:
        wins, total = 0, 0
        for r in results:
            if r["rank"] != rank or r["method"] != "svdquant":
                continue
            total += 1
            # Find matching r_only
            match = [x for x in results
                     if x["layer"] == r["layer"] and x["proj"] == r["proj"]
                     and x["rank"] == rank and x["method"] == "r_only"]
            if match and match[0]["snr_method_db"] > r["snr_method_db"]:
                wins += 1
        if total:
            pct = 100 * wins / total
            print(f"  rank={rank:3d}  {wins}/{total} ({pct:.0f}%)")

    # 4. Per-layer average
    print(f"\n--- Per-layer average Δ(R_only - SVDQuant) dB, rank=128 ---")
    layer_set = sorted(set(r["layer"] for r in results))
    for layer_idx in layer_set:
        svdq = [r["snr_method_db"] for r in results
                if r["layer"] == layer_idx and r["rank"] == 128
                and r["method"] == "svdquant"]
        ronly = [r["snr_method_db"] for r in results
                 if r["layer"] == layer_idx and r["rank"] == 128
                 and r["method"] == "r_only"]
        if svdq and ronly:
            diff = sum(ronly) / len(ronly) - sum(svdq) / len(svdq)
            bar = "█" * max(0, int(diff * 5)) if diff > 0 else "░" * max(0, int(-diff * 5))
            print(f"  Layer {layer_idx:2d}: Δ={diff:+.2f} dB {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep all layers: smooth+R_only vs smooth+SVDQuant")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="",
                        help="Layer range, e.g. '0-17' or '0,2,5'. Empty=all 36")
    parser.add_argument("--tag", default="",
                        help="Output CSV tag to avoid file collision between GPUs")
    args = parser.parse_args()

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            parts = args.layers.split("-")
            layers = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(NUM_LAYERS))

    print(f"Layers: {layers}")
    print(f"Device: {args.device}")
    print(f"Ranks:  {RANKS}")
    print(f"Methods: {METHODS}")
    print(f"Total experiments: {len(layers) * len(PROJ_TYPES) * len(RANKS) * len(METHODS)}")
    print(f"Saving params to: {os.path.join(RESULTS_DIR, 'layer_results')}/")
    print()

    run_sweep(layers, args.device, tag=args.tag)


if __name__ == "__main__":
    main()
