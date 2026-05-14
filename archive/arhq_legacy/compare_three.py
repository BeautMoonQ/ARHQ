"""Legacy script: compare three configurations head-to-head.

Use ``python -m arhq.decompose --configs ...`` for the minimal codebase flow.

Original behavior:
  1. raw + ARHQ         (no transfer, ARHQ decomposition)
  2. smoothing + ARHQ   (SmoothQuant transfer + ARHQ)
  3. smoothing + SVDQuant (SmoothQuant transfer + SVDQuant, ≈ paper config)

Reports absolute SNR (not improvement over baseline) for fair cross-setting comparison.
"""

import argparse
import csv
import os
import time

import torch

from arhq.quant import nvfp4_quantize, nvfp4_quantize_2d
from arhq.lowrank import svdquant_decompose, arhq_decompose, compute_snr
from arhq.transforms import search_best_alpha

SAMPLES_DIR = os.path.expanduser("~/work/data/calib/vllm_MATH-500/samples_0000")
EVAL_TOKENS = 2048
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
RANKS = [32, 64, 128, 256]
NUM_LAYERS = 36
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

CONFIGS = [
    ("raw_arhq",       None,        "arhq"),
    ("smooth_svdq",    "smoothing", "svdquant"),
    ("smooth_arhq",    "smoothing", "arhq"),
]


def load_layer_data(layer_idx: int, device: str):
    layer_dir = os.path.join(SAMPLES_DIR, f"layer_{layer_idx}")
    activations = torch.load(
        os.path.join(layer_dir, "activations_truncated.pt"),
        map_location="cpu", weights_only=True,
    )
    weights = torch.load(
        os.path.join(layer_dir, "weights.pt"),
        map_location="cpu", weights_only=True,
    )
    return activations, weights


def load_proj_data(activations, weights, proj, device):
    """Load one proj to GPU, returns (A_calib, A_eval, W)."""
    A_all = activations[proj].float()
    W = weights[proj].float()
    A_calib = A_all[:-EVAL_TOKENS].to(device)
    A_eval = A_all[-EVAL_TOKENS:].to(device)
    W = W.to(device)
    return A_calib, A_eval, W


@torch.no_grad()
def evaluate_config(A_eval, W, A_calib, rank, setting, method, scale,
                    w_quant_fn=None):
    """Returns absolute SNR in dB for one config."""
    if w_quant_fn is None:
        w_quant_fn = nvfp4_quantize
    Y_true = A_eval @ W.T

    if scale is not None:
        A_e = A_eval / scale
        A_c = A_calib / scale
        Wt = W * scale
    else:
        A_e = A_eval
        A_c = A_calib
        Wt = W

    # Decompose
    if method == "svdquant":
        decomp = svdquant_decompose(Wt, rank)
    else:
        decomp = arhq_decompose(Wt, A_c, rank)

    W_res = decomp["W_res"]
    B_r = decomp["B_r"]
    A_fac = decomp["A_fac"]

    Y_hat = nvfp4_quantize(A_e) @ w_quant_fn(W_res).T + (A_e @ A_fac) @ B_r.T
    return compute_snr(Y_true, Y_hat)


def run(layers, device, weight_quant_2d=False):
    w_quant_fn = nvfp4_quantize_2d if weight_quant_2d else nvfp4_quantize
    w_tag = "2d" if weight_quant_2d else "1d"
    print(f"Weight quantization: {w_tag} block")

    os.makedirs(os.path.join(RESULTS_DIR, "summary"), exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "summary", f"three_way_compare_{w_tag}.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["layer", "proj", "rank", "raw_arhq", "smooth_svdq",
                     "smooth_arhq", "alpha"])

    all_rows = []
    t0 = time.time()

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")
        activations, weights = load_layer_data(layer_idx, device)

        for proj in PROJ_TYPES:
            A_calib, A_eval, W = load_proj_data(activations, weights, proj, device)

            # Compute smoothing scale once
            best_alpha, best_scale = search_best_alpha(A_calib, W)

            # Also compute no-lowrank baselines for reference
            Y_true = A_eval @ W.T
            snr_base_raw = compute_snr(
                Y_true, nvfp4_quantize(A_eval) @ w_quant_fn(W).T)
            A_s = A_eval / best_scale
            W_s = W * best_scale
            snr_base_smooth = compute_snr(
                Y_true, nvfp4_quantize(A_s) @ w_quant_fn(W_s).T)

            print(f"  {proj:7s}  base_raw={snr_base_raw:.2f}  "
                  f"base_smooth={snr_base_smooth:.2f}  alpha={best_alpha:.2f}")

            for rank in RANKS:
                if rank > min(W.shape[0], W.shape[1]):
                    continue

                snr_raw_arhq = evaluate_config(
                    A_eval, W, A_calib, rank, None, "arhq", None, w_quant_fn)
                snr_smooth_svdq = evaluate_config(
                    A_eval, W, A_calib, rank, "smoothing", "svdquant", best_scale, w_quant_fn)
                snr_smooth_arhq = evaluate_config(
                    A_eval, W, A_calib, rank, "smoothing", "arhq", best_scale, w_quant_fn)

                # Who wins?
                best_val = max(snr_raw_arhq, snr_smooth_svdq, snr_smooth_arhq)
                markers = {
                    snr_raw_arhq: "raw_arhq",
                    snr_smooth_svdq: "smooth_svdq",
                    snr_smooth_arhq: "smooth_arhq",
                }
                winner = markers[best_val]

                print(f"    r={rank:3d}  raw_arhq={snr_raw_arhq:6.2f}  "
                      f"sm_svdq={snr_smooth_svdq:6.2f}  "
                      f"sm_arhq={snr_smooth_arhq:6.2f}  "
                      f"best={winner}")

                row = {
                    "layer": layer_idx, "proj": proj, "rank": rank,
                    "raw_arhq": round(snr_raw_arhq, 4),
                    "smooth_svdq": round(snr_smooth_svdq, 4),
                    "smooth_arhq": round(snr_smooth_arhq, 4),
                    "alpha": best_alpha,
                }
                all_rows.append(row)
                writer.writerow([
                    layer_idx, proj, rank,
                    round(snr_raw_arhq, 4), round(snr_smooth_svdq, 4),
                    round(snr_smooth_arhq, 4), best_alpha,
                ])
                csv_file.flush()

            del A_calib, A_eval, W
            torch.cuda.empty_cache()
        del activations, weights
        torch.cuda.empty_cache()

    csv_file.close()
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary ({len(all_rows)} experiments in {elapsed:.1f}s)")
    print(f"{'='*70}")

    print(f"\nAverage absolute SNR (dB) across all layers:")
    for rank in RANKS:
        ra = [r["raw_arhq"] for r in all_rows if r["rank"] == rank]
        ss = [r["smooth_svdq"] for r in all_rows if r["rank"] == rank]
        sa = [r["smooth_arhq"] for r in all_rows if r["rank"] == rank]
        if ra:
            print(f"  rank={rank:3d}  raw_arhq={sum(ra)/len(ra):.2f}  "
                  f"smooth_svdq={sum(ss)/len(ss):.2f}  "
                  f"smooth_arhq={sum(sa)/len(sa):.2f}")

    print(f"\nPer-proj average absolute SNR (dB), rank=128:")
    for proj in PROJ_TYPES:
        ra = [r["raw_arhq"] for r in all_rows if r["rank"] == 128 and r["proj"] == proj]
        ss = [r["smooth_svdq"] for r in all_rows if r["rank"] == 128 and r["proj"] == proj]
        sa = [r["smooth_arhq"] for r in all_rows if r["rank"] == 128 and r["proj"] == proj]
        if ra:
            print(f"  {proj:7s}  raw_arhq={sum(ra)/len(ra):.2f}  "
                  f"smooth_svdq={sum(ss)/len(ss):.2f}  "
                  f"smooth_arhq={sum(sa)/len(sa):.2f}")

    # Win count
    print(f"\nWin count (best absolute SNR):")
    for rank in RANKS:
        wins = {"raw_arhq": 0, "smooth_svdq": 0, "smooth_arhq": 0}
        for r in all_rows:
            if r["rank"] != rank:
                continue
            best_key = max(["raw_arhq", "smooth_svdq", "smooth_arhq"],
                           key=lambda k: r[k])
            wins[best_key] += 1
        total = sum(wins.values())
        print(f"  rank={rank:3d}  raw_arhq={wins['raw_arhq']}/{total}  "
              f"smooth_svdq={wins['smooth_svdq']}/{total}  "
              f"smooth_arhq={wins['smooth_arhq']}/{total}")

    print(f"\nSaved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="")
    parser.add_argument("--weight_quant_2d", action="store_true",
                        help="Use 16x16 2D block weight quantization (B100 TE style)")
    args = parser.parse_args()

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            parts = args.layers.split("-")
            layers = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(NUM_LAYERS))

    run(layers, args.device, weight_quant_2d=args.weight_quant_2d)


if __name__ == "__main__":
    main()
