"""Transforms for difficulty transfer before quantization."""

import torch


def get_smoothquant_scaling(A_calib: torch.Tensor, W: torch.Tensor,
                            alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant-style per-channel scaling (as used in SVDQuant).

    lambda_i = max(|X[:,i]|)^alpha / max(|W[i,:]|)^(1-alpha)

    Convention: Y = X @ W.T where X is [N, D_in], W is [D_out, D_in].
    So X[:,i] is activation channel i, W[:,i] (= W's column i) is the
    corresponding weight input channel.

    Returns scale [D_in] such that:
        X_hat = X / scale       (per-channel divide)
        W_hat = W * scale       (per-channel multiply on input dim)
    """
    # activation per-channel max: [D_in]
    act_max = A_calib.abs().amax(dim=0).clamp(min=1e-12)
    # weight per-input-channel max: W is [D_out, D_in], max over D_out
    w_max = W.abs().amax(dim=0).clamp(min=1e-12)

    scale = act_max.pow(alpha) / w_max.pow(1 - alpha)
    return scale


def search_best_alpha(A_calib: torch.Tensor, W: torch.Tensor,
                      alphas=None, chunk_size: int = 2048) -> tuple:
    """Search for best alpha that minimizes output MSE after scaling + quantization.

    Processes activation in chunks to avoid OOM.
    Returns (best_alpha, best_scale).
    """
    from .quant import nvfp4_quantize

    if alphas is None:
        alphas = [i * 0.05 for i in range(21)]  # 0.0 to 1.0 step 0.05

    N = A_calib.shape[0]
    best_mse = float("inf")
    best_alpha = 0.5
    best_scale = None

    for alpha in alphas:
        scale = get_smoothquant_scaling(A_calib, W, alpha)
        W_s = W * scale
        W_s_q = nvfp4_quantize(W_s)

        total_se = 0.0
        total_n = 0
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            A_chunk = A_calib[start:end]
            Y_true_chunk = A_chunk @ W.T
            A_s_chunk = A_chunk / scale
            Y_hat_chunk = nvfp4_quantize(A_s_chunk) @ W_s_q.T
            total_se += (Y_true_chunk - Y_hat_chunk).pow(2).sum().item()
            total_n += Y_true_chunk.numel()

        mse = total_se / total_n
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_scale = scale

    return best_alpha, best_scale
