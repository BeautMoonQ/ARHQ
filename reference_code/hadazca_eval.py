"""
hadazca_eval.py
---------------
对 hadazca_calib.py 收集的每层 qkvo 激活+权重，执行多种变换+量化方案的对比评估。

支持用不同的 sample 分别做 ZCA 计算和误差评估：
  --zca_samples 0-31     用 sample 0~31 的激活计算 ZCA 矩阵
  --eval_samples 32-63   用 sample 32~63 的激活评估量化误差

方案列表:
  1. Baseline:          无变换，直接 NVFP4 量化
  2. WHT only:          全维度 WHT（block-diagonal）+ NVFP4
  3. WHT grouped:       指定 group_size 的分组 WHT + NVFP4
  4. ZCA only:          全维度 ZCA（原始空间）+ NVFP4
  5. ZCA grouped:       分组 ZCA（原始空间）+ NVFP4
  6. WHT + ZCA grouped: WHT 后分组 ZCA + NVFP4
  7. WHT + ZCA full:    WHT 后全维度 ZCA + NVFP4

用法:
  # 用同一批数据做 ZCA 和评估
  python claude/hadazca_eval.py --zca_samples 0-31 --eval_samples 0-31 --layers 0

  # 用不同数据做 ZCA 和评估（推荐）
  python claude/hadazca_eval.py --zca_samples 0-31 --eval_samples 32-63 --layers 0

  # 搜索最优 p
  python claude/hadazca_eval.py --zca_samples 0-31 --eval_samples 32-63 --layers 0 --search_p
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_CALIB_DIR = "/home/wangyifeng/work/data/calib"


def parse_args():
    parser = argparse.ArgumentParser(
        description="HadaZCA: WHT + ZCA + NVFP4 量化评估（多方案对比）"
    )
    parser.add_argument("--calib_dir", type=str, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--precision", type=str, default="vllm")
    parser.add_argument("--dataset", type=str, default="MATH-500")
    parser.add_argument("--zca_samples", type=str, default="0-31",
                        help="用于计算 ZCA 的样本范围，如 '0-31'（对应 samples_0000 目录）")
    parser.add_argument("--eval_samples", type=str, default="0-31",
                        help="用于评估误差的样本范围，如 '32-63'（对应 samples_0032 目录）")
    parser.add_argument("--layers", type=str, default="",
                        help="要处理的层，如 '0' 或 '0-3' 或 '0,2,5'，空=全部已收集层")
    parser.add_argument("--p", type=float, default=0.2,
                        help="ZCA 白化强度参数")
    parser.add_argument("--n_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=16,
                        help="量化分组大小")
    parser.add_argument("--zca_group_size", type=int, default=128,
                        help="ZCA 分组大小，-1 表示全维度 ZCA")
    parser.add_argument("--search_p", action="store_true",
                        help="对每层三分搜索最优 p")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def parse_range(range_str):
    """解析范围字符串如 '0-31'，返回 (start, end)。"""
    parts = range_str.split("-")
    start = int(parts[0])
    end = int(parts[1]) if len(parts) > 1 else start
    return start, end


# ─────────────────────────────────────────────────────────────────────────────
# Hadamard
# ─────────────────────────────────────────────────────────────────────────────

def get_hadamard(n, device="cpu"):
    """递归构造 Walsh-Hadamard 矩阵（n 必须是 2 的幂）。"""
    if n == 1:
        return torch.tensor([[1.0]], device=device)
    h = get_hadamard(n // 2, device)
    return torch.cat([torch.cat([h, h], dim=1),
                      torch.cat([h, -h], dim=1)], dim=0)


def get_block_hadamard(dim, device="cpu", block_size=None):
    """
    构造 block-diagonal Hadamard 矩阵。
    block_size=None 时自动取 dim 的最大 2 次幂因子。
    返回: (H, H_inv, block_size)
    """
    if block_size is None:
        block_size = 1
        while block_size * 2 <= dim and dim % (block_size * 2) == 0:
            block_size *= 2

    assert dim % block_size == 0, f"dim={dim} 不能被 block_size={block_size} 整除"
    num_blocks = dim // block_size

    h_small = get_hadamard(block_size, device=device)
    H = torch.block_diag(*[h_small for _ in range(num_blocks)])
    H_inv = torch.block_diag(*[h_small / block_size for _ in range(num_blocks)])
    return H, H_inv, block_size


# ─────────────────────────────────────────────────────────────────────────────
# ZCA（支持分组）
# ─────────────────────────────────────────────────────────────────────────────

def get_zca_matrix(X, p=0.2, epsilon=1e-8):
    """
    全维度 ZCA 白化矩阵。
    X: [N, D]（已中心化的数据）
    返回: (P_zca [D,D], P_inv [D,D])
    """
    N = X.shape[0]
    Sigma = X.T @ X / (N - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    eigenvalues = torch.clamp(eigenvalues, min=0)

    inv_sqrt = torch.pow(eigenvalues + epsilon, -p)
    P_zca = eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.T

    sqrt_val = torch.pow(eigenvalues + epsilon, p)
    P_inv = eigenvectors @ torch.diag(sqrt_val) @ eigenvectors.T

    return P_zca, P_inv


def get_grouped_zca_matrix(X, zca_group_size, p=0.2, epsilon=1e-8):
    """
    分组 ZCA 白化矩阵。将 D 维分成 D/zca_group_size 组，每组独立计算 ZCA。
    X: [N, D]（已中心化的数据）
    返回: (P_zca [D,D] block-diagonal, P_inv [D,D] block-diagonal)
    """
    N, D = X.shape
    assert D % zca_group_size == 0, f"D={D} 不能被 zca_group_size={zca_group_size} 整除"
    num_groups = D // zca_group_size

    zca_blocks = []
    inv_blocks = []

    for g in range(num_groups):
        start = g * zca_group_size
        end = start + zca_group_size
        X_g = X[:, start:end]

        Sigma_g = X_g.T @ X_g / (N - 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_g)
        eigenvalues = torch.clamp(eigenvalues, min=0)

        inv_sqrt = torch.pow(eigenvalues + epsilon, -p)
        P_g = eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.T
        zca_blocks.append(P_g)

        sqrt_val = torch.pow(eigenvalues + epsilon, p)
        P_inv_g = eigenvectors @ torch.diag(sqrt_val) @ eigenvectors.T
        inv_blocks.append(P_inv_g)

    P_zca = torch.block_diag(*zca_blocks)
    P_inv = torch.block_diag(*inv_blocks)
    return P_zca, P_inv


# ─────────────────────────────────────────────────────────────────────────────
# 量化
# ─────────────────────────────────────────────────────────────────────────────

def pseudo_quantize_group(x, n_bits=4, group_size=128):
    """对称分组量化（模拟 NVFP4）。"""
    orig_shape = x.shape

    if x.dim() == 2:
        if x.shape[1] % group_size != 0:
            pad_size = group_size - x.shape[1] % group_size
            x = torch.nn.functional.pad(x, (0, pad_size))
            orig_shape = x.shape
        x_reshaped = x.view(x.shape[0], -1, group_size)
    else:
        if x.shape[-1] % group_size != 0:
            pad_size = group_size - x.shape[-1] % group_size
            x = torch.nn.functional.pad(x, (0, pad_size))
            orig_shape = x.shape
        x_reshaped = x.view(*x.shape[:-1], -1, group_size)

    xmax = x_reshaped.abs().amax(dim=-1, keepdim=True)
    qmax = 2 ** (n_bits - 1) - 1
    scale = xmax / qmax
    scale = scale.clamp(min=1e-5)

    x_int = (x_reshaped / scale).round().clamp(-qmax, qmax)
    x_dequant = (x_int * scale).view(orig_shape)

    if x_dequant.dim() == 2:
        x_dequant = x_dequant[:, :orig_shape[1]]
    else:
        x_dequant = x_dequant[..., :orig_shape[-1]]

    return x_dequant


# ─────────────────────────────────────────────────────────────────────────────
# 量化误差计算辅助
# ─────────────────────────────────────────────────────────────────────────────

def compute_quant_error(Y_ref, A, W, mean, T, T_inv, n_bits, group_size):
    """
    通用的变换+量化+误差计算。
    T, T_inv: 变换矩阵
    mean: 激活均值（ZCA 数据的均值，用于 bias correction）
    A: 评估用激活（未中心化）
    """
    act_centered = A - mean
    A_t = act_centered @ T
    W_t = W @ T_inv.T

    A_q = pseudo_quantize_group(A_t, n_bits=n_bits, group_size=group_size)
    W_q = pseudo_quantize_group(W_t, n_bits=n_bits, group_size=group_size)

    bias = mean @ W.T
    Y_q = A_q @ W_q.T + bias

    diff = (Y_ref - Y_q).abs()
    mse = (Y_ref - Y_q).pow(2).mean().item()
    mean_abs = diff.mean().item()
    abs_max = diff.max().item()
    return mse, mean_abs, abs_max


# ─────────────────────────────────────────────────────────────────────────────
# 单层多方案评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_layer_module(zca_act, eval_act, weight, p, n_bits, group_size,
                          zca_group_size, device, hadamard_cache={},
                          save_dir=None, mod_name=None):
    """
    对单个模块执行 7 种方案的对比评估。
    zca_act:  用于计算 ZCA 的激活 [N_zca, D]
    eval_act: 用于评估误差的激活 [N_eval, D]
    weight:   权重 [D_out, D]
    返回: dict { scheme_name: {"mse": ..., "mean_abs": ...} }
    """
    D = eval_act.shape[-1]
    zca_act = zca_act.to(device).float()
    eval_act = eval_act.to(device).float()
    weight = weight.to(device).float()

    # 参考输出（用 eval 数据）
    Y_ref = eval_act @ weight.T

    # ZCA 数据的均值（变换矩阵基于 ZCA 数据计算）
    zca_mean = zca_act.mean(dim=0) * 0.0
    zca_centered = zca_act - zca_mean

    # print('zca_mean', zca_mean[:10], zca_mean.mean().item())
    # exit()
    # zca_centered = zca_act

    results = {}

    # ── 1. Baseline: 无变换直接量化 ──
    A_q = pseudo_quantize_group(eval_act, n_bits=n_bits, group_size=group_size)
    W_q = pseudo_quantize_group(weight, n_bits=n_bits, group_size=group_size)
    Y_raw = A_q @ W_q.T
    diff_raw = (Y_ref - Y_raw).abs()
    results["1_baseline"] = {
        "mse": (Y_ref - Y_raw).pow(2).mean().item(),
        "mean_abs": diff_raw.mean().item(),
        "abs_max": diff_raw.max().item(),
    }

    # ── 2. WHT only（自动 block size）──
    if D not in hadamard_cache:
        H, H_inv, bs = get_block_hadamard(D, device=device)
        hadamard_cache[D] = (H, H_inv, bs)
    H_full, H_full_inv, bs_full = hadamard_cache[D]
    mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                  H_full, H_full_inv, n_bits, group_size)
    results["2_wht_only"] = {"mse": mse, "mean_abs": ma, "abs_max": am, "info": f"block={bs_full}"}

    # ── 3. WHT grouped（group_size 大小的 block）──
    wht_gs = group_size
    if D % wht_gs == 0 and (wht_gs & (wht_gs - 1) == 0):
        H_g, H_g_inv, _ = get_block_hadamard(D, device=device, block_size=wht_gs)
        mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                      H_g, H_g_inv, n_bits, group_size)
        results["3_wht_grouped"] = {"mse": mse, "mean_abs": ma, "abs_max": am, "info": f"block={wht_gs}"}
    else:
        results["3_wht_grouped"] = {"mse": float("nan"), "mean_abs": float("nan"), "abs_max": float("nan"),
                                     "info": f"skip: group_size={wht_gs} not power of 2 or not divisor"}

    # ── 4. ZCA only（全维度，原始空间）──
    Z, Z_inv = get_zca_matrix(zca_centered, p=p)
    Z = Z.to(device)
    Z_inv = Z_inv.to(device)
    mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                  Z, Z_inv, n_bits, group_size)
    results["4_zca_only"] = {"mse": mse, "mean_abs": ma, "abs_max": am}

    # ── 5. ZCA grouped（分组，原始空间）──
    effective_zca_gs = zca_group_size if zca_group_size > 0 else D
    if D % effective_zca_gs == 0 and effective_zca_gs < D:
        Z_g, Z_g_inv = get_grouped_zca_matrix(zca_centered, effective_zca_gs, p=p)
        Z_g = Z_g.to(device)
        Z_g_inv = Z_g_inv.to(device)
        mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                      Z_g, Z_g_inv, n_bits, group_size)
        results["5_zca_grouped"] = {"mse": mse, "mean_abs": ma, "abs_max": am, "info": f"zca_gs={effective_zca_gs}"}
    else:
        results["5_zca_grouped"] = {"mse": float("nan"), "mean_abs": float("nan"), "abs_max": float("nan"),
                                     "info": f"skip or same as full (zca_gs={effective_zca_gs})"}

    # ── 6. WHT + ZCA grouped（WHT 后分组 ZCA）──
    zca_had = zca_centered @ H_full

    # zca_had = zca_act @ H_full
    # zca_mean = zca_had.mean(dim=0) * 0.0
    # zca_had = zca_had - zca_mean
    # print('zca_mean', zca_mean[:10], zca_mean.mean().item())
    # exit()

    if D % effective_zca_gs == 0 and effective_zca_gs < D:
        Z_hg, Z_hg_inv = get_grouped_zca_matrix(zca_had, effective_zca_gs, p=p)
        Z_hg = Z_hg.to(device)
        Z_hg_inv = Z_hg_inv.to(device)
        T = H_full @ Z_hg
        T_inv = Z_hg_inv @ H_full_inv
        mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                      T, T_inv, n_bits, group_size)
        results["6_wht+zca_grouped"] = {"mse": mse, "mean_abs": ma, "abs_max": am, "info": f"zca_gs={effective_zca_gs}"}
    else:
        results["6_wht+zca_grouped"] = {"mse": float("nan"), "mean_abs": float("nan"), "abs_max": float("nan"),
                                         "info": f"skip (zca_gs={effective_zca_gs})"}

    # ── 6b/6c/6d. WHT + ZCA grouped（额外 group size 对比：16, 64, 256）──
    for extra_gs in [16, 32, 64, 256]:
        key = f"6_wht+zca_gs{extra_gs}"
        if extra_gs == effective_zca_gs:
            # 已经在方案 6 中计算过，跳过
            continue
        if D % extra_gs == 0 and extra_gs < D:
            Z_ex, Z_ex_inv = get_grouped_zca_matrix(zca_had, extra_gs, p=p)
            Z_ex = Z_ex.to(device)
            Z_ex_inv = Z_ex_inv.to(device)

            # 保存 gs16 和 gs32 的 ZCA 矩阵
            # if extra_gs in (16, 32) and save_dir is not None and mod_name is not None:
            if extra_gs in (16, 64, 256) and save_dir is not None and mod_name is not None:
                save_name = f"zca_gs{extra_gs}_{mod_name}.pt"
                torch.save({
                    "Z": Z_ex.cpu(),
                    "Z_inv": Z_ex_inv.cpu(),
                    "H": H_full.cpu(),
                    "H_inv": H_full_inv.cpu(),
                    "mean": zca_mean.cpu(),
                    "p": p,
                    "zca_group_size": extra_gs,
                }, os.path.join(save_dir, save_name))
                print(f"    已保存 {save_dir} {save_name}")
            T_ex = H_full @ Z_ex
            T_ex_inv = Z_ex_inv @ H_full_inv
            mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                          T_ex, T_ex_inv, n_bits, group_size)
            results[key] = {"mse": mse, "mean_abs": ma, "abs_max": am, "info": f"zca_gs={extra_gs}"}
        else:
            results[key] = {"mse": float("nan"), "mean_abs": float("nan"), "abs_max": float("nan"),
                            "info": f"skip (D={D} % {extra_gs} != 0)"}

    # ── 7. WHT + ZCA full（WHT 后全维度 ZCA）──
    Z_hf, Z_hf_inv = get_zca_matrix(zca_had, p=p)
    Z_hf = Z_hf.to(device)
    Z_hf_inv = Z_hf_inv.to(device)
    T = H_full @ Z_hf
    T_inv = Z_hf_inv @ H_full_inv
    mse, ma, am = compute_quant_error(Y_ref, eval_act, weight, zca_mean,
                                  T, T_inv, n_bits, group_size)
    results["7_wht+zca_full"] = {"mse": mse, "mean_abs": ma, "abs_max": am}

    return results


def search_best_p(zca_act, eval_act, weight, n_bits, group_size, zca_group_size,
                  device, scheme="7_wht+zca_full"):
    """三分搜索最优 p。"""
    low, high = 0.0, 1.0

    def get_mse(p_val):
        result = evaluate_layer_module(zca_act, eval_act, weight, p_val,
                                       n_bits, group_size, zca_group_size, device)
        return result[scheme]["mse"]

    while high - low > 0.01:
        m1 = low + (high - low) / 3
        m2 = high - (high - low) / 3
        e1 = get_mse(m1)
        e2 = get_mse(m2)
        if e1 < e2:
            high = m2
        else:
            low = m1

    return (low + high) / 2


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def samples_dir_name(sample_start):
    """从 sample range 的起始 idx 得到目录名。"""
    return f"samples_{sample_start:04d}"


def find_collected_layers(samples_path):
    """扫描指定 samples 目录下已收集的层。"""
    layers = []
    if not os.path.isdir(samples_path):
        return layers
    for entry in os.listdir(samples_path):
        if entry.startswith("layer_"):
            try:
                idx = int(entry.split("_")[1])
                if os.path.exists(os.path.join(samples_path, entry, "activations.pt")):
                    layers.append(idx)
            except ValueError:
                pass
    return sorted(layers)


def load_layer_data(samples_path, layer_idx):
    """加载指定层的激活和权重。"""
    layer_dir = os.path.join(samples_path, f"layer_{layer_idx}")
    activations = torch.load(os.path.join(layer_dir, "activations.pt"), map_location="cpu")
    weights = torch.load(os.path.join(layer_dir, "weights.pt"), map_location="cpu")
    return activations, weights


def parse_layer_range(layers_str, available):
    if not layers_str:
        return available
    if "-" in layers_str and "," not in layers_str:
        parts = layers_str.split("-")
        return [i for i in range(int(parts[0]), int(parts[1]) + 1) if i in available]
    requested = [int(x.strip()) for x in layers_str.split(",")]
    return [i for i in requested if i in available]


SCHEME_NAMES = {
    "1_baseline":         "Baseline (no transform)",
    "2_wht_only":         "WHT only",
    "3_wht_grouped":      "WHT grouped",
    "4_zca_only":         "ZCA full",
    "5_zca_grouped":      "ZCA grouped",
    "6_wht+zca_grouped":  "WHT + ZCA grouped",
    "6_wht+zca_gs16":     "WHT + ZCA gs16",
    "6_wht+zca_gs64":     "WHT + ZCA gs64",
    "6_wht+zca_gs256":    "WHT + ZCA gs256",
    "7_wht+zca_full":     "WHT + ZCA full",
}


@torch.no_grad()
def main():
    args = parse_args()
    device = args.device

    base_dir = os.path.join(
        os.path.expanduser(args.calib_dir),
        f"{args.precision}_{args.dataset}"
    )

    # 解析 sample ranges
    zca_start, zca_end = parse_range(args.zca_samples)
    eval_start, eval_end = parse_range(args.eval_samples)
    same_samples = (zca_start == eval_start)

    zca_samples_path = os.path.join(base_dir, samples_dir_name(zca_start))
    eval_samples_path = os.path.join(base_dir, samples_dir_name(eval_start))

    print(f"ZCA 数据:  {zca_samples_path}  (samples {zca_start}-{zca_end})")
    print(f"Eval 数据: {eval_samples_path}  (samples {eval_start}-{eval_end})")

    # 检查目录
    for path, label in [(zca_samples_path, "ZCA"), (eval_samples_path, "Eval")]:
        if not os.path.isdir(path):
            print(f"[ERROR] {label} 数据目录不存在: {path}")
            sys.exit(1)

    # 找可用层（取两个目录的交集）
    zca_layers = set(find_collected_layers(zca_samples_path))
    eval_layers = set(find_collected_layers(eval_samples_path))
    available_layers = sorted(zca_layers & eval_layers)

    if not available_layers:
        print(f"[ERROR] ZCA 和 Eval 目录没有共同的层数据")
        sys.exit(1)

    layer_indices = parse_layer_range(args.layers, available_layers)
    print(f"可用层: {available_layers}")
    print(f"将评估: {layer_indices}")
    print(f"参数: p={args.p}, n_bits={args.n_bits}, group_size={args.group_size}, "
          f"zca_group_size={args.zca_group_size}")
    if args.search_p:
        print("模式: 三分搜索最优 p")

    all_results = {}

    for layer_idx in layer_indices:
        zca_activations, zca_weights = load_layer_data(zca_samples_path, layer_idx)
        if same_samples:
            eval_activations, eval_weights = zca_activations, zca_weights
        else:
            eval_activations, eval_weights = load_layer_data(eval_samples_path, layer_idx)

        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}")
        print(f"{'='*70}")

        layer_results = {}

        for mod_name in eval_activations:
            if mod_name not in eval_weights:
                print(f"  [SKIP] {mod_name}: 无对应权重")
                continue
            if mod_name not in zca_activations:
                print(f"  [SKIP] {mod_name}: ZCA 数据中无此模块")
                continue

            zca_act = zca_activations[mod_name]
            eval_act = eval_activations[mod_name]
            w = eval_weights[mod_name]
            print(f"\n  {mod_name}: zca_act {tuple(zca_act.shape)}, "
                  f"eval_act {tuple(eval_act.shape)}, weight {tuple(w.shape)}")

            if args.search_p:
                best_p = search_best_p(zca_act, eval_act, w, args.n_bits,
                                       args.group_size, args.zca_group_size, device)
                print(f"  搜索到最优 p = {best_p:.4f}")
                p = best_p
            else:
                p = args.p

            layer_save_dir = os.path.join(base_dir, f"layer_{layer_idx}")
            os.makedirs(layer_save_dir, exist_ok=True)
            result = evaluate_layer_module(
                zca_act, eval_act, w, p, args.n_bits, args.group_size,
                args.zca_group_size, device,
                save_dir=layer_save_dir, mod_name=mod_name
            )
            layer_results[mod_name] = {"p": p, "schemes": result}

            # 打印对比表
            baseline_mse = result["1_baseline"]["mse"]
            baseline_am = result["1_baseline"]["abs_max"]
            print(f"  {'Scheme':<25} {'MSE':>12} {'MSE_R':>8} {'AbsMax':>12} {'AM_R':>8}")
            print(f"  {'-'*68}")
            for scheme_key in sorted(result.keys()):
                r = result[scheme_key]
                mse_val = r["mse"]
                am_val = r["abs_max"]
                mse_r = mse_val / max(baseline_mse, 1e-30)
                am_r = am_val / max(baseline_am, 1e-30)
                info = r.get("info", "")
                name = SCHEME_NAMES.get(scheme_key, scheme_key)
                if info:
                    name = f"{name} ({info})"
                print(f"  {name:<25} {mse_val:>12.4e} {mse_r:>8.4f} {am_val:>12.4e} {am_r:>8.4f}")

        all_results[f"layer_{layer_idx}"] = layer_results
        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*70}")
    print("汇总（MSE ratio vs Baseline）")
    print(f"{'='*70}")

    for layer_key, mods in all_results.items():
        for mod_name, data in mods.items():
            schemes = data["schemes"]
            baseline_mse = schemes["1_baseline"]["mse"]
            baseline_am = schemes["1_baseline"]["abs_max"]
            best_scheme = min(
                ((k, v["mse"]) for k, v in schemes.items() if k != "1_baseline"),
                key=lambda x: x[1]
            )
            print(f"  {layer_key}.{mod_name} (p={data['p']:.3f}):")
            for sk in sorted(schemes.keys()):
                if sk == "1_baseline":
                    continue
                mse_r = schemes[sk]["mse"] / max(baseline_mse, 1e-30)
                am_r = schemes[sk]["abs_max"] / max(baseline_am, 1e-30)
                marker = " <-- best" if sk == best_scheme[0] else ""
                print(f"    {SCHEME_NAMES.get(sk, sk):<25} mse_r={mse_r:.4f}  am_r={am_r:.4f}{marker}")

    # 保存结果
    result_path = os.path.join(base_dir, "hadazca_results.json")
    serializable = {
        "config": {
            "zca_samples": args.zca_samples,
            "eval_samples": args.eval_samples,
            "p": args.p,
            "n_bits": args.n_bits,
            "group_size": args.group_size,
            "zca_group_size": args.zca_group_size,
        },
        "results": {},
    }
    for k, v in all_results.items():
        serializable["results"][k] = {}
        for mk, mv in v.items():
            serializable["results"][k][mk] = {
                "p": mv["p"],
                "schemes": {
                    sk: {kk: float(vv) if isinstance(vv, (int, float)) else str(vv)
                         for kk, vv in sv.items()}
                    for sk, sv in mv["schemes"].items()
                }
            }
    with open(result_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n结果保存到: {result_path}")


if __name__ == "__main__":
    main()
