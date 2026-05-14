"""
FP4 Rotation Quantization Research — prepare.py
================================================
固定的评估标准和数据加载。Agent 不可修改此文件。

数据来源: hadazca_calib.py 收集的逐层 qkvo 激活值和权重
  samples_0000: sample 0-15, ~97K tokens
  samples_0016: sample 16-31, ~110K tokens

每层数据格式:
  activations.pt → {"q_proj": [N_tokens, D_in], ...}
  weights.pt     → {"q_proj": [D_out, D_in], ...}

评估逻辑:
  对每层每个 proj，计算 Y_true = A @ W.T，然后对比：
    - baseline: nvfp4_quantize(A) @ nvfp4_quantize(W).T
    - rotated:  nvfp4_quantize((A-mean) @ T) @ nvfp4_quantize(W @ T_inv.T).T + bias

主指标: avg_snr_improvement_db (越高越好)
"""

import argparse
import os
import sys
import torch

# ---------------------------------------------------------------------------
# 固定常量
# ---------------------------------------------------------------------------
TIME_BUDGET = 600

# Weight 量化模式: "nvfp4" | "none" | "gptq"
WEIGHT_QUANT_MODE = "nvfp4"

CALIB_BASE = os.path.expanduser("~/work/data/calib/vllm_MATH-500")
SAMPLES_DIR = os.path.join(CALIB_BASE, "samples_0000")

NUM_LAYERS = 36
EVAL_LAYERS = [2]   # 当前研究目标层
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# activations_truncated.pt 共 10000 tokens
# 评估集：最后 2048 tokens；校准集：其余全部
EVAL_TOKENS = 2048

# Qwen3-4B 维度
HIDDEN_DIM = 2560
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
# q_proj: [2560, 2560], k_proj: [640, 2560], v_proj: [640, 2560], o_proj: [2560, 2560]


# ---------------------------------------------------------------------------
# NVFP4 量化 (E2M1, per-block FP8 E4M3 scale)
# ---------------------------------------------------------------------------
def _fp8_e4m3_sim(x: torch.Tensor) -> torch.Tensor:
    """模拟 FP8 E4M3 scale 的精度限制。"""
    log2_x = torch.log2(x.abs().clamp(min=1e-45))
    exponent = log2_x.floor()
    mantissa_scale = 2.0 ** (exponent - 3)
    return (torch.round(x / mantissa_scale) * mantissa_scale).clamp(min=1e-12)


def nvfp4_quantize(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVFP4 (E2M1) 量化模拟，per-block FP8 E4M3 scale。"""
    original_shape = x.shape
    last_dim = x.shape[-1]
    pad_size = (block_size - last_dim % block_size) % block_size
    if pad_size > 0:
        x = torch.nn.functional.pad(x, (0, pad_size))

    x_flat = x.reshape(-1, block_size)
    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 6.0
    scale = _fp8_e4m3_sim(scale)

    x_scaled = x_flat / scale
    fp4_pos = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                           device=x.device, dtype=x.dtype)
    all_vals = torch.cat([-fp4_pos[1:].flip(0), fp4_pos])

    x_clamped = x_scaled.clamp(-6.0, 6.0)
    diffs = (x_clamped.unsqueeze(-1) - all_vals).abs()
    x_quant = all_vals[diffs.argmin(dim=-1)]

    x_dequant = (x_quant * scale).reshape(x.shape)
    if pad_size > 0:
        x_dequant = x_dequant[..., :last_dim]
    return x_dequant.reshape(original_shape)


# ---------------------------------------------------------------------------
# GPTQ 量化
# ---------------------------------------------------------------------------
def gptq_quantize(W: torch.Tensor, H: torch.Tensor,
                  block_size: int = 16) -> torch.Tensor:
    """
    GPTQ weight 量化（量化到 NVFP4 grid，使用 Hessian 做误差补偿）。

    W: [D_out, D_in]  权重矩阵
    H: [D_in, D_in]   Hessian = X.T @ X（无需归一化）

    按 block_size 列一组做 GPTQ，每组内逐列量化并补偿误差到同组剩余列。
    """
    W_q = W.clone()
    D_out, D_in = W.shape

    # Hessian 对角加 dampening
    damp = 0.01 * H.diag().mean()
    H = H + damp * torch.eye(D_in, device=H.device, dtype=H.dtype)

    for col_start in range(0, D_in, block_size):
        col_end = min(col_start + block_size, D_in)
        bs = col_end - col_start

        # 该 block 的 Hessian 子矩阵的逆
        H_block = H[col_start:col_end, col_start:col_end]
        H_block_inv = torch.linalg.inv(H_block)

        W_block = W_q[:, col_start:col_end].clone()
        Q_block = torch.zeros_like(W_block)
        Err = torch.zeros_like(W_block)

        for j in range(bs):
            w_col = W_block[:, j]
            h_jj = H_block_inv[j, j]

            # 量化该列（用 nvfp4 的 round-to-nearest）
            q_col = nvfp4_quantize(w_col.unsqueeze(0)).squeeze(0)
            Q_block[:, j] = q_col
            err = (w_col - q_col) / h_jj

            # 补偿误差到该 block 内剩余列
            if j + 1 < bs:
                W_block[:, j + 1:] -= err.unsqueeze(1) * H_block_inv[j, j + 1:].unsqueeze(0)
            Err[:, j] = err

        W_q[:, col_start:col_end] = Q_block

        # 补偿误差到后续所有列
        if col_end < D_in:
            W_q[:, col_end:] -= Err @ H[col_start:col_end, col_end:]

    return W_q


# ---------------------------------------------------------------------------
# Weight 量化调度
# ---------------------------------------------------------------------------
def quantize_weight(W: torch.Tensor, A_calib: torch.Tensor = None) -> torch.Tensor:
    """根据 WEIGHT_QUANT_MODE 对权重进行量化。"""
    if WEIGHT_QUANT_MODE == "none":
        return W
    elif WEIGHT_QUANT_MODE == "gptq":
        if A_calib is None:
            raise ValueError("GPTQ mode requires calibration activations")
        H = A_calib.T @ A_calib
        return gptq_quantize(W, H)
    else:  # "nvfp4"
        return nvfp4_quantize(W)


def parse_prepare_args():
    """从 sys.argv 中提取 prepare 相关参数（不影响 optimize.py 的参数）。"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--weight_quant", type=str, default="nvfp4",
                        choices=["nvfp4", "none", "gptq"],
                        help="Weight quantization mode")
    args, _ = parser.parse_known_args()
    return args


# 启动时解析命令行参数
_prepare_args = parse_prepare_args()
WEIGHT_QUANT_MODE = _prepare_args.weight_quant


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def _load_layer_raw(layer_idx: int) -> tuple:
    """加载指定层数据（不缓存，避免多层时 OOM）。"""
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


def load_calibration_data(layer_idx: int, proj_type: str) -> tuple:
    """返回校准集 (A [N-EVAL_TOKENS, D], W [D_out, D])。取除最后 EVAL_TOKENS 外的全部 token。"""
    activations, weights = _load_layer_raw(layer_idx)
    A = activations[proj_type].float()[:-EVAL_TOKENS]
    W = weights[proj_type].float()
    return A, W


def load_eval_data(layer_idx: int, proj_type: str) -> tuple:
    """返回评估集 (A [EVAL_TOKENS, D], W [D_out, D])。取最后 EVAL_TOKENS 个 token。"""
    activations, weights = _load_layer_raw(layer_idx)
    A = activations[proj_type].float()[-EVAL_TOKENS:]
    W = weights[proj_type].float()
    return A, W


# ---------------------------------------------------------------------------
# 评估函数（唯一的评估标准）
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_transform(get_transform_fn) -> dict:
    """
    评估变换方案的性能。

    Args:
        get_transform_fn: callable(layer_idx, proj_type, A_calib, W, device)
            -> (T, T_inv, mean) 或 None
            T: [D, D] 变换矩阵
            T_inv: [D, D] 逆变换矩阵，满足 T @ T_inv ≈ I
            mean: [D] 激活均值（用于 bias correction），可以为 zeros

    Returns:
        dict with 'avg_snr_improvement_db' (主指标，越高越好) 等
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    improvements = []
    per_module = {}

    for layer_idx in EVAL_LAYERS:
        for proj in PROJ_TYPES:
            A_eval, W = load_eval_data(layer_idx, proj)
            A_eval = A_eval.to(device)
            W = W.to(device)
            Y_true = A_eval @ W.T

            # Baseline: 无变换直接量化
            A_calib_base, _ = load_calibration_data(layer_idx, proj)
            A_calib_base = A_calib_base.to(device)
            A_q_base = nvfp4_quantize(A_eval)
            W_q_base = quantize_weight(W, A_calib_base)
            Y_base = A_q_base @ W_q_base.T
            err_base = (Y_base - Y_true).norm()
            snr_base = 20 * torch.log10(
                Y_true.norm() / err_base.clamp(min=1e-12)
            )

            # 有变换（A_calib_base 已加载，复用）
            A_calib = A_calib_base
            result = get_transform_fn(layer_idx, proj, A_calib, W, device)

            if result is not None:
                T, T_inv, mean = result
                T = T.to(device)
                T_inv = T_inv.to(device)
                mean = mean.to(device)

                A_centered = A_eval - mean
                A_t = A_centered @ T
                W_t = W @ T_inv.T

                A_q = nvfp4_quantize(A_t)
                # GPTQ 需要变换后的校准激活
                A_calib_t = (A_calib - mean) @ T
                W_q = quantize_weight(W_t, A_calib_t)

                bias = mean @ W.T
                Y_rot = A_q @ W_q.T + bias
            else:
                Y_rot = Y_base

            err_rot = (Y_rot - Y_true).norm()
            snr_rot = 20 * torch.log10(
                Y_true.norm() / err_rot.clamp(min=1e-12)
            )

            imp = (snr_rot - snr_base).item()
            improvements.append(imp)
            key = f"L{layer_idx}_{proj}"
            per_module[key] = {
                "snr_base": round(snr_base.item(), 2),
                "snr_rot": round(snr_rot.item(), 2),
                "improvement_db": round(imp, 4),
            }

            del A_eval, W, A_calib, Y_true, Y_base, Y_rot
            torch.cuda.empty_cache()

    avg_imp = sum(improvements) / len(improvements) if improvements else 0.0
    worst_imp = min(improvements) if improvements else 0.0

    return {
        "avg_snr_improvement_db": round(avg_imp, 4),
        "worst_case_improvement_db": round(worst_imp, 4),
        "n_eval_points": len(improvements),
        "per_module": per_module,
    }


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("prepare.py: 此文件提供数据加载和评估工具。")
    print(f"  SAMPLES_DIR:       {SAMPLES_DIR}")
    print(f"  EVAL_LAYERS:       {EVAL_LAYERS}")
    print(f"  EVAL_TOKENS:       {EVAL_TOKENS}")
    print(f"  CALIB_TOKENS:      全部 - {EVAL_TOKENS} (动态)")
    print(f"  TIME_BUDGET:       {TIME_BUDGET}s")
    print(f"  WEIGHT_QUANT_MODE: {WEIGHT_QUANT_MODE}")
