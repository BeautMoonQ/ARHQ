"""Legacy script: sweep evaluation across layers/projections/ranks.

Use ``python -m arhq.decompose`` for the minimal codebase flow.
"""

import argparse
import csv
import json
import os
import time

import torch

from arhq.lowrank import evaluate_single
from arhq.transforms import search_best_alpha

SAMPLES_DIR = os.path.expanduser("~/work/data/calib/vllm_MATH-500/samples_0000")
EVAL_TOKENS = 2048
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
RANKS = [32, 64, 128, 256]
METHODS = ["svdquant", "arhq"]
SETTINGS = ["raw", "smoothing"]
NUM_LAYERS = 36
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_layer_data(layer_idx: int, device: str):
    """Load full layer data once (all projs)."""
    layer_dir = os.path.join(SAMPLES_DIR, f"layer_{layer_idx}")
    activations = torch.load(
        os.path.join(layer_dir, "activations_truncated.pt"),
        map_location="cpu", weights_only=True,
    )
    weights = torch.load(
        os.path.join(layer_dir, "weights.pt"),
        map_location="cpu", weights_only=True,
    )
    data = {}
    for proj in PROJ_TYPES:
        A_all = activations[proj].float()
        W = weights[proj].float()
        data[proj] = {
            "A_calib": A_all[:-EVAL_TOKENS].to(device),
            "A_eval": A_all[-EVAL_TOKENS:].to(device),
            "W": W.to(device),
        }
    return data


def run_sweep(layers: list, device: str, save_params: bool = True,
              output_tag: str = ""):
    os.makedirs(os.path.join(RESULTS_DIR, "summary"), exist_ok=True)
    tag = f"_{output_tag}" if output_tag else ""
    csv_path = os.path.join(RESULTS_DIR, "summary", f"sweep_results{tag}.csv")

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["layer", "proj", "rank", "method", "setting",
                     "snr_baseline_db", "snr_method_db", "snr_improvement_db",
                     "beta", "alpha"])

    all_results = []
    t0 = time.time()

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        layer_data = load_layer_data(layer_idx, device)

        for proj in PROJ_TYPES:
            d = layer_data[proj]
            A_eval, W, A_calib = d["A_eval"], d["W"], d["A_calib"]

            # Precompute smoothing scale once per (layer, proj)
            best_alpha, best_scale = search_best_alpha(A_calib, W)

            for setting in SETTINGS:
                scale = best_scale if setting == "smoothing" else None

                for rank in RANKS:
                    if rank > min(W.shape[0], W.shape[1]):
                        continue
                    for method in METHODS:
                        try:
                            result = evaluate_single(
                                A_eval, W, A_calib, rank, method, scale
                            )
                        except Exception as e:
                            print(f"  ERROR: L{layer_idx} {proj} {method} {setting} r={rank}: {e}")
                            continue

                        row = {
                            "layer": layer_idx, "proj": proj, "rank": rank,
                            "method": method, "setting": setting,
                            **{k: v for k, v in result.items()
                               if k not in ("B_r", "A_fac", "W_res")},
                            "alpha": best_alpha if setting == "smoothing" else None,
                        }
                        all_results.append(row)
                        writer.writerow([
                            layer_idx, proj, rank, method, setting,
                            result["snr_baseline_db"], result["snr_method_db"],
                            result["snr_improvement_db"], result.get("beta"),
                            best_alpha if setting == "smoothing" else "",
                        ])
                        csv_file.flush()

                        imp = result["snr_improvement_db"]
                        marker = "+" if imp > 0 else ""
                        alpha_str = f" a={best_alpha:.2f}" if setting == "smoothing" else ""
                        print(f"  L{layer_idx:02d} {proj:7s} {setting:9s} {method:9s} r={rank:3d} "
                              f"base={result['snr_baseline_db']:6.2f} method={result['snr_method_db']:6.2f} "
                              f"imp={marker}{imp:.2f} dB{alpha_str}")

                        # Save decomposition params
                        if save_params:
                            save_dir = os.path.join(RESULTS_DIR, "layer_results",
                                                    f"layer_{layer_idx}")
                            os.makedirs(save_dir, exist_ok=True)
                            fname = f"{proj}_{method}_{setting}_rank{rank}.pt"
                            save_data = {
                                "B_r": result["B_r"],
                                "A_fac": result["A_fac"],
                                "W_res": result["W_res"],
                                "rank": rank,
                                "method": method,
                                "setting": setting,
                                "beta": result.get("beta"),
                            }
                            if scale is not None:
                                save_data["scale"] = scale.half().cpu()
                                save_data["alpha"] = best_alpha
                            torch.save(save_data, os.path.join(save_dir, fname))

            del A_eval, W, A_calib
        del layer_data
        torch.cuda.empty_cache()

    csv_file.close()
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(all_results)} experiments in {elapsed:.1f}s")
    print(f"Results saved to {csv_path}")

    # Print comparison summary
    print_summary(all_results)


def print_summary(results: list):
    """Print ARHQ vs SVDQuant comparison."""
    print(f"\n{'='*60}")
    print("ARHQ vs SVDQuant Summary (SNR improvement over baseline, dB)")
    print(f"{'='*60}")

    for setting in SETTINGS:
        print(f"\n--- Setting: {setting} ---")
        for rank in RANKS:
            svd_imps, arhq_imps = [], []
            for r in results:
                if r["setting"] != setting or r["rank"] != rank:
                    continue
                if r["method"] == "svdquant":
                    svd_imps.append(r["snr_improvement_db"])
                elif r["method"] == "arhq":
                    arhq_imps.append(r["snr_improvement_db"])
            if svd_imps and arhq_imps:
                svd_avg = sum(svd_imps) / len(svd_imps)
                arhq_avg = sum(arhq_imps) / len(arhq_imps)
                diff = arhq_avg - svd_avg
                print(f"  rank={rank:3d}  SVDQuant avg={svd_avg:+.2f}  ARHQ avg={arhq_avg:+.2f}  "
                      f"ARHQ-SVD={diff:+.2f} dB")

    # Per-proj breakdown for rank=128
    print(f"\nPer-proj breakdown (rank=128):")
    for setting in SETTINGS:
        print(f"  --- {setting} ---")
        for proj in PROJ_TYPES:
            svd = [r["snr_improvement_db"] for r in results
                   if r["setting"] == setting and r["rank"] == 128
                   and r["method"] == "svdquant" and r["proj"] == proj]
            arhq = [r["snr_improvement_db"] for r in results
                    if r["setting"] == setting and r["rank"] == 128
                    and r["method"] == "arhq" and r["proj"] == proj]
            if svd and arhq:
                s = sum(svd) / len(svd)
                a = sum(arhq) / len(arhq)
                print(f"    {proj:7s}  SVD={s:+.2f}  ARHQ={a:+.2f}  diff={a-s:+.2f} dB")

    # Win rate
    print(f"\nWin rate (ARHQ > SVDQuant):")
    for setting in SETTINGS:
        for rank in RANKS:
            wins, total = 0, 0
            for r in results:
                if r["setting"] != setting or r["rank"] != rank or r["method"] != "svdquant":
                    continue
                total += 1
                # Find matching ARHQ result
                arhq_match = [x for x in results
                              if x["layer"] == r["layer"] and x["proj"] == r["proj"]
                              and x["setting"] == setting and x["rank"] == rank
                              and x["method"] == "arhq"]
                if arhq_match and arhq_match[0]["snr_improvement_db"] > r["snr_improvement_db"]:
                    wins += 1
            if total:
                print(f"  {setting:9s} rank={rank:3d}  {wins}/{total} ({100*wins//total}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--layers", default="",
                        help="Layer range, e.g. '0-17' or '0,2,5'. Empty=all")
    parser.add_argument("--no-save-params", action="store_true")
    parser.add_argument("--tag", default="", help="Output file tag")
    args = parser.parse_args()

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            parts = args.layers.split("-")
            layers = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(NUM_LAYERS))

    run_sweep(layers, args.device, save_params=not args.no_save_params,
              output_tag=args.tag)


if __name__ == "__main__":
    main()
