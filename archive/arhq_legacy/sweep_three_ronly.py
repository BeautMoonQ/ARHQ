"""Legacy script: sweep three attention configs on one calibration set.

Use ``python -m arhq.decompose --configs arhq:raw,arhq:smoothing,svdquant:smoothing``
for the minimal codebase flow.

Original behavior:
1. raw + r_only
2. smoothing + r_only
3. smoothing + svdquant

Saves summary CSV plus decomposition params for later inference.
"""

import argparse
import csv
import os
import time

import torch

from arhq.lowrank import evaluate_single
from arhq.transforms import search_best_alpha

EVAL_TOKENS = 2048
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
NUM_LAYERS = 36
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

CONFIGS = [
    ("r_only", "raw"),
    ("r_only", "smoothing"),
    ("svdquant", "smoothing"),
]


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return list(range(NUM_LAYERS))
    if "-" in spec and "," not in spec:
        start, end = spec.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in spec.split(",") if x]


def load_layer_data(samples_dir: str, layer_idx: int, device: str):
    layer_dir = os.path.join(samples_dir, f"layer_{layer_idx}")
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


def save_decomp(result: dict, save_dir: str, proj: str, method: str,
                setting: str, rank: int, alpha, scale):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{proj}_{method}_{setting}_rank{rank}.pt")
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
        save_data["alpha"] = alpha
    torch.save(save_data, path)


def run(samples_dir: str, layers: list[int], device: str, rank: int, tag: str = ""):
    os.makedirs(os.path.join(RESULTS_DIR, "summary"), exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    csv_path = os.path.join(
        RESULTS_DIR, "summary", f"three_ronly_rank{rank}{suffix}.csv"
    )
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow([
        "layer", "proj", "rank", "method", "setting",
        "snr_baseline_db", "snr_method_db", "snr_improvement_db", "beta", "alpha",
    ])

    all_results = []
    t0 = time.time()

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        layer_data = load_layer_data(samples_dir, layer_idx, device)

        for proj in PROJ_TYPES:
            d = layer_data[proj]
            A_eval, W, A_calib = d["A_eval"], d["W"], d["A_calib"]
            best_alpha, best_scale = search_best_alpha(A_calib, W)

            config_scores = {}
            for method, setting in CONFIGS:
                scale = best_scale if setting == "smoothing" else None
                try:
                    result = evaluate_single(A_eval, W, A_calib, rank, method, scale)
                except Exception as e:
                    print(f"  ERROR: L{layer_idx} {proj} {method} {setting} r={rank}: {e}")
                    continue

                config_scores[(method, setting)] = result["snr_method_db"]
                row = {
                    "layer": layer_idx,
                    "proj": proj,
                    "rank": rank,
                    "method": method,
                    "setting": setting,
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

                save_dir = os.path.join(RESULTS_DIR, "layer_results", f"layer_{layer_idx}")
                save_decomp(result, save_dir, proj, method, setting, rank, best_alpha, scale)

            if len(config_scores) == 3:
                raw_r = config_scores[("r_only", "raw")]
                sm_r = config_scores[("r_only", "smoothing")]
                sm_s = config_scores[("svdquant", "smoothing")]
                print(
                    f"  {proj:7s} r={rank:3d}  "
                    f"raw_r={raw_r:6.2f}  sm_r={sm_r:6.2f}  sm_svdq={sm_s:6.2f}  "
                    f"alpha={best_alpha:.2f}"
                )

            del A_eval, W, A_calib
            torch.cuda.empty_cache()

        del layer_data
        torch.cuda.empty_cache()

    csv_file.close()
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Sweep complete: {len(all_results)} experiments in {elapsed:.1f}s")
    print(f"Results saved to {csv_path}")
    print(f"Params saved to {os.path.join(RESULTS_DIR, 'layer_results')}/")

    for label, method, setting in [
        ("raw_r_only", "r_only", "raw"),
        ("smooth_r_only", "r_only", "smoothing"),
        ("smooth_svdq", "svdquant", "smoothing"),
    ]:
        vals = [r["snr_method_db"] for r in all_results
                if r["method"] == method and r["setting"] == setting]
        if vals:
            print(f"  {label:14s}: avg_abs_snr = {sum(vals)/len(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run raw+r_only, smooth+r_only, smooth+svdquant on one calibration set."
    )
    parser.add_argument("--samples_dir", required=True,
                        help="Calibration samples dir, e.g. .../vllm_ZebraLogic/samples_0000")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", default="",
                        help="Layer range, e.g. '0-35' or '0,2,5'")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--tag", default="", help="Output tag to avoid collisions")
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    print(f"Samples: {args.samples_dir}")
    print(f"Layers:  {layers}")
    print(f"Device:  {args.device}")
    print(f"Rank:    {args.rank}")
    print()

    run(args.samples_dir, layers, args.device, args.rank, tag=args.tag)


if __name__ == "__main__":
    main()
