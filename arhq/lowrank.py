"""SVDQuant baseline and ARHQ low-rank decomposition + evaluation."""

import torch
from .quant import nvfp4_quantize


def svdquant_decompose(W: torch.Tensor, rank: int) -> dict:
    """SVDQuant baseline: truncated SVD of W.

    W: [D_out, D]
    Returns {B_r: [D_out, r], A_fac: [D, r], W_res: [D_out, D]}
    where W_sig = B_r @ A_fac.T
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    B_r = U[:, :rank] * S[:rank]        # [D_out, r]
    A_fac = Vh[:rank, :].T              # [D, r]
    W_sig = B_r @ A_fac.T
    W_res = W - W_sig
    return {"B_r": B_r, "A_fac": A_fac, "W_res": W_res}


def arhq_decompose(W: torch.Tensor, A_calib: torch.Tensor,
                   rank: int, epsilon: float = 1e-6,
                   w_quant_fn=None) -> dict:
    """ARHQ: Act-Residual Hessian weighted low-rank decomposition.

    W: [D_out, D]
    A_calib: [N, D] calibration activations (already transformed if using difficulty transfer)
    w_quant_fn: if provided, also incorporate weight quantization residual into H_res
    Returns {B_r, A_fac, W_res, beta_a, beta_w}
    """
    N, D = A_calib.shape
    D_out = W.shape[0]

    # Activation Hessian
    H = A_calib.T @ A_calib / N                    # [D, D]

    # Activation quantization residual
    E_a = A_calib - nvfp4_quantize(A_calib)         # [N, D]
    R_a = E_a.T @ E_a / N                           # [D, D]

    # Scale alignment for activation residual
    h_diag_mean = H.diag().mean()
    ra_diag_mean = R_a.diag().mean().clamp(min=1e-12)
    beta_a = (h_diag_mean / ra_diag_mean).item()

    # Act-Residual Hessian
    H_res = H + beta_a * R_a

    # Optionally add weight quantization residual
    beta_w = None
    if w_quant_fn is not None:
        E_w = W - w_quant_fn(W)                     # [D_out, D]
        R_w = E_w.T @ E_w / D_out                   # [D, D]
        rw_diag_mean = R_w.diag().mean().clamp(min=1e-12)
        beta_w = (h_diag_mean / rw_diag_mean).item()
        H_res = H_res + beta_w * R_w

    # Eigendecomposition
    eigenvalues, U_h = torch.linalg.eigh(H_res)
    eigenvalues = eigenvalues.clamp(min=epsilon)

    # H_res^{1/2} and H_res^{-1/2}
    sqrt_eig = eigenvalues.sqrt()
    inv_sqrt_eig = 1.0 / sqrt_eig

    H_res_sqrt = U_h @ torch.diag(sqrt_eig) @ U_h.T         # [D, D]
    H_res_inv_sqrt = U_h @ torch.diag(inv_sqrt_eig) @ U_h.T  # [D, D]

    # Weighted matrix
    M = W @ H_res_sqrt                              # [D_out, D]

    # SVD of M
    U_m, S_m, Vh_m = torch.linalg.svd(M, full_matrices=False)
    B_r = U_m[:, :rank] * S_m[:rank]                # [D_out, r]
    A_tilde = Vh_m[:rank, :].T                      # [D, r]

    # Map back to original coordinates
    A_fac = H_res_inv_sqrt @ A_tilde                # [D, r]
    W_sig = B_r @ A_fac.T
    W_res = W - W_sig

    return {"B_r": B_r, "A_fac": A_fac, "W_res": W_res,
            "beta_a": beta_a, "beta_w": beta_w}


def r_only_decompose(W: torch.Tensor, A_calib: torch.Tensor,
                     rank: int, epsilon: float = 1e-6) -> dict:
    """R-only decomposition: use only quantization residual covariance R.

    H_res = R_a = E^T E / N, where E = A - Q(A).
    No activation Hessian H, no beta hyperparameter.
    """
    N, D = A_calib.shape

    E_a = A_calib - nvfp4_quantize(A_calib)
    R_a = E_a.T @ E_a / N

    eigenvalues, U_h = torch.linalg.eigh(R_a)
    eigenvalues = eigenvalues.clamp(min=epsilon)

    sqrt_eig = eigenvalues.sqrt()
    inv_sqrt_eig = 1.0 / sqrt_eig

    R_sqrt = U_h @ torch.diag(sqrt_eig) @ U_h.T
    R_inv_sqrt = U_h @ torch.diag(inv_sqrt_eig) @ U_h.T

    M = W @ R_sqrt
    U_m, S_m, Vh_m = torch.linalg.svd(M, full_matrices=False)
    B_r = U_m[:, :rank] * S_m[:rank]
    A_tilde = Vh_m[:rank, :].T

    A_fac = R_inv_sqrt @ A_tilde
    W_sig = B_r @ A_fac.T
    W_res = W - W_sig

    return {"B_r": B_r, "A_fac": A_fac, "W_res": W_res}


def compute_snr(Y_true: torch.Tensor, Y_hat: torch.Tensor) -> float:
    """SNR in dB."""
    err = (Y_hat - Y_true).norm()
    sig = Y_true.norm()
    return (20 * torch.log10(sig / err.clamp(min=1e-12))).item()


@torch.no_grad()
def evaluate_single(A_eval: torch.Tensor, W: torch.Tensor,
                    A_calib: torch.Tensor, rank: int, method: str,
                    scale: torch.Tensor = None) -> dict:
    """Evaluate one (layer, proj, rank, method, setting) combination.

    Args:
        A_eval: [N_eval, D] eval activations (original space)
        W: [D_out, D] weight (original space)
        A_calib: [N_calib, D] calibration activations (original space)
        rank: low-rank dimension
        method: "svdquant" or "arhq"
        scale: None (raw setting) or [D] per-channel scale for smoothing
               A' = A / scale, W' = W * scale

    Returns dict with SNR metrics and decomposition params.
    """
    # Ground truth in original space
    Y_true = A_eval @ W.T

    # Apply smoothing if given
    if scale is not None:
        A_eval_t = A_eval / scale
        A_calib_t = A_calib / scale
        W_t = W * scale                  # [D_out, D] * [D] broadcasts on last dim
    else:
        A_eval_t = A_eval
        A_calib_t = A_calib
        W_t = W

    # Baseline: no low-rank, just quantize everything
    Y_baseline = nvfp4_quantize(A_eval_t) @ nvfp4_quantize(W_t).T
    snr_baseline = compute_snr(Y_true, Y_baseline)

    # Low-rank decomposition on (possibly smoothed) weight
    if method == "svdquant":
        decomp = svdquant_decompose(W_t, rank)
    elif method == "arhq":
        decomp = arhq_decompose(W_t, A_calib_t, rank)
    elif method == "r_only":
        decomp = r_only_decompose(W_t, A_calib_t, rank)
    else:
        raise ValueError(f"Unknown method: {method}")

    B_r = decomp["B_r"]
    A_fac = decomp["A_fac"]
    W_res = decomp["W_res"]

    # Deploy: Y ≈ A_t @ (A_fac @ B_r.T) + Q(A_t) @ Q(W_res).T
    # Low-rank branch in float, main branch quantized
    Y_main = nvfp4_quantize(A_eval_t) @ nvfp4_quantize(W_res).T
    Y_lr = (A_eval_t @ A_fac) @ B_r.T
    Y_hat = Y_main + Y_lr
    snr_method = compute_snr(Y_true, Y_hat)

    return {
        "snr_baseline_db": round(snr_baseline, 4),
        "snr_method_db": round(snr_method, 4),
        "snr_improvement_db": round(snr_method - snr_baseline, 4),
        "beta": decomp.get("beta_a", decomp.get("beta")),
        # Decomposition params for saving
        "B_r": B_r.half().cpu(),
        "A_fac": A_fac.half().cpu(),
        "W_res": W_res.half().cpu(),
    }
