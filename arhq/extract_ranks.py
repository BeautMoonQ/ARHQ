"""Extract smaller ranks from a saved rank=R decomposition.

SVD singular vectors are nested: rank-r result is always the first r columns
of the rank-R result (R >= r). So we can derive any r < R without re-running.

  B_r      = B_R[:, :r]
  A_fac_r  = A_fac_R[:, :r]
  W_res_r  = W_res_R + B_R[:, r:] @ A_fac_R[:, r:].T   (add tail back)

Usage:
  # Extract ranks [32, 64] from all saved rank=128 files
  python -m arhq.extract_ranks --source_rank 128 --target_ranks 32 64 \
      --method r_only

  # All methods, all layers
  python -m arhq.extract_ranks --source_rank 128 --target_ranks 32 64 \
      --method r_only svdquant
"""

import argparse
import os
import torch

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "layer_results")
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
NUM_LAYERS = 36


def extract_rank(source_path: str, target_rank: int, target_path: str):
    """Derive a smaller rank decomposition from a larger one."""
    data = torch.load(source_path, map_location="cpu", weights_only=True)

    B_R = data["B_r"].float()      # [D_out, R]
    A_R = data["A_fac"].float()    # [D_in, R]
    W_R = data["W_res"].float()    # [D_out, D_in]
    R = B_R.shape[1]

    assert target_rank < R, f"target_rank={target_rank} must be < source_rank={R}"

    B_r = B_R[:, :target_rank]
    A_r = A_R[:, :target_rank]
    # Add back the tail components (rank r..R) to the residual
    W_r = W_R + B_R[:, target_rank:] @ A_R[:, target_rank:].T

    save_data = {
        **{k: v for k, v in data.items() if k not in ("B_r", "A_fac", "W_res", "rank")},
        "B_r": B_r.half(),
        "A_fac": A_r.half(),
        "W_res": W_r.half(),
        "rank": target_rank,
        "derived_from_rank": R,
    }
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.save(save_data, target_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_rank", type=int, default=128)
    parser.add_argument("--target_ranks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--method", type=str, nargs="+",
                        default=["r_only", "svdquant"])
    parser.add_argument("--setting", type=str, default="smoothing")
    parser.add_argument("--layers", type=str, default="",
                        help="e.g. '0-35' or '0,5,10'. Empty = all 36")
    args = parser.parse_args()

    if args.layers:
        if "-" in args.layers and "," not in args.layers:
            a, b = args.layers.split("-")
            layers = list(range(int(a), int(b) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    else:
        layers = list(range(NUM_LAYERS))

    total = 0
    errors = 0
    for method in args.method:
        for layer_idx in layers:
            layer_dir = os.path.join(RESULTS_DIR, f"layer_{layer_idx}")
            for proj in PROJ_TYPES:
                src_fname = f"{proj}_{method}_{args.setting}_rank{args.source_rank}.pt"
                src_path = os.path.join(layer_dir, src_fname)
                if not os.path.exists(src_path):
                    print(f"  MISSING: {src_path}")
                    errors += 1
                    continue

                for r in args.target_ranks:
                    if r >= args.source_rank:
                        print(f"  SKIP: target_rank={r} >= source_rank={args.source_rank}")
                        continue
                    dst_fname = f"{proj}_{method}_{args.setting}_rank{r}.pt"
                    dst_path = os.path.join(layer_dir, dst_fname)

                    extract_rank(src_path, r, dst_path)
                    total += 1

    print(f"\nDone: extracted {total} files, {errors} missing sources")


if __name__ == "__main__":
    main()
