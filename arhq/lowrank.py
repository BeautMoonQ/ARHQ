"""Low-rank decompositions and nvfp4 simulation utilities.

This module contains the two algorithms exposed by the minimal ARHQ repo:

* ARHQ: Activation Residual Hessian Quantization.  The low-rank branch is
  selected under the activation quantization residual metric
  ``R_x = (X - Q(X)).T @ (X - Q(X)) / N``.
* SVDQuant: a reproduction baseline that uses a plain truncated SVD of the
  weight matrix, optionally after SmoothQuant-style smoothing.
"""

import torch
from .quant import nvfp4_quantize, nvfp4_quantize_2d


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


def residual_hessian_decompose(W: torch.Tensor, A_calib: torch.Tensor,
                               rank: int, epsilon: float = 1e-6,
                               w_quant_fn=None) -> dict:
    """Legacy/general residual-Hessian decomposition using H + beta R.

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

    return {
        "B_r": B_r,
        "A_fac": A_fac,
        "W_res": W_res,
        "beta_a": beta_a,
        "beta_w": beta_w,
        "metric": "H_plus_beta_R",
    }


def arhq_decompose(W: torch.Tensor, A_calib: torch.Tensor,
                   rank: int, epsilon: float = 1e-6) -> dict:
    """ARHQ decomposition using only activation quantization residual Hessian.

    Objective:
        min_rank(L)<=r || E_x @ (W - L).T ||_F^2

    where E_x = X - Q(X).  Equivalently, solve a weighted low-rank
    approximation under G_x = E_x.T @ E_x / N.
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

    return {
        "B_r": B_r,
        "A_fac": A_fac,
        "W_res": W_res,
        "metric": "activation_residual",
    }


def r_only_decompose(W: torch.Tensor, A_calib: torch.Tensor,
                     rank: int, epsilon: float = 1e-6) -> dict:
    """Compatibility alias for ARHQ.

    Older experiment scripts used ``r_only`` for the current ARHQ method.
    New code should call :func:`arhq_decompose` or pass ``method=arhq``.
    """
    out = arhq_decompose(W, A_calib, rank, epsilon=epsilon)
    out["legacy_method"] = "r_only"
    return out


def _truncate_full_svd(B_full: torch.Tensor, A_full: torch.Tensor,
                       W: torch.Tensor, rank: int) -> dict:
    """Truncate full-rank SVD factors to `rank` and recompute W_res in fp32.

    B_full: [D_out, R_max] — left factor at max rank
    A_full: [D_in,  R_max] — right factor at max rank
    """
    B_r = B_full[:, :rank].contiguous()
    A_fac = A_full[:, :rank].contiguous()
    W_res = W - B_r @ A_fac.T
    return {"B_r": B_r, "A_fac": A_fac, "W_res": W_res}


def svdquant_decompose_multirank(W: torch.Tensor, ranks: list[int]) -> dict[int, dict]:
    """Compute SVDQuant decomposition once and produce all `ranks` independently.

    Each per-rank result is bit-identical (in fp32) to calling
    :func:`svdquant_decompose` with that rank, because SVD singular vectors are
    nested. W_res is recomputed in fp32 from the truncated factors, so there is
    no fp16 accumulation error.
    """
    max_rank = max(ranks)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    B_full = U[:, :max_rank] * S[:max_rank]   # [D_out, max_rank]
    A_full = Vh[:max_rank, :].T               # [D_in,  max_rank]
    out = {}
    for r in ranks:
        out[r] = _truncate_full_svd(B_full, A_full, W, r)
    return out


def _compute_R_chunked(A_calib: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Compute R = E^T E / N where E = A - Q(A), processing in chunks.

    Avoids materializing the full nvfp4 intermediate tensor (which expands
    the last dim by 15x and OOMs on large FFN activations).
    """
    N, D = A_calib.shape
    R = torch.zeros(D, D, dtype=A_calib.dtype, device=A_calib.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = A_calib[start:end]
        e = chunk - nvfp4_quantize(chunk)
        R += e.T @ e
    return R / N


def arhq_decompose_multirank(W: torch.Tensor, A_calib: torch.Tensor,
                             ranks: list[int],
                             epsilon: float = 1e-6) -> dict[int, dict]:
    """Compute ARHQ decomposition once and produce all `ranks` independently.

    Reuses the same R eigendecomposition and SVD of M = W @ R_sqrt for all
    ranks. Each per-rank result is bit-identical (in fp32) to calling
    :func:`arhq_decompose` with that rank.
    """
    max_rank = max(ranks)
    N, D = A_calib.shape

    R_a = _compute_R_chunked(A_calib)

    eigenvalues, U_h = torch.linalg.eigh(R_a)
    eigenvalues = eigenvalues.clamp(min=epsilon)
    sqrt_eig = eigenvalues.sqrt()
    inv_sqrt_eig = 1.0 / sqrt_eig
    R_sqrt = U_h @ torch.diag(sqrt_eig) @ U_h.T
    R_inv_sqrt = U_h @ torch.diag(inv_sqrt_eig) @ U_h.T

    M = W @ R_sqrt
    U_m, S_m, Vh_m = torch.linalg.svd(M, full_matrices=False)
    B_full = U_m[:, :max_rank] * S_m[:max_rank]      # [D_out, max_rank]
    A_tilde_full = Vh_m[:max_rank, :].T              # [D_in, max_rank]
    A_full = R_inv_sqrt @ A_tilde_full               # [D_in, max_rank]

    out = {}
    for r in ranks:
        d = _truncate_full_svd(B_full, A_full, W, r)
        d["metric"] = "activation_residual"
        out[r] = d
    return out


def decompose_multirank(method: str, W: torch.Tensor, A_calib: torch.Tensor,
                        ranks: list[int]) -> dict[int, dict]:
    """Dispatch multi-rank decomposition by method name."""
    if method == "svdquant":
        return svdquant_decompose_multirank(W, ranks)
    if method in ("arhq", "r_only"):
        return arhq_decompose_multirank(W, A_calib, ranks)
    raise ValueError(f"unknown method: {method}")


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
    Y_baseline = nvfp4_quantize(A_eval_t) @ nvfp4_quantize_2d(W_t).T
    snr_baseline = compute_snr(Y_true, Y_baseline)

    # Low-rank decomposition on (possibly smoothed) weight
    if method == "svdquant":
        decomp = svdquant_decompose(W_t, rank)
    elif method == "arhq":
        decomp = arhq_decompose(W_t, A_calib_t, rank)
    elif method == "arhq_full":
        decomp = residual_hessian_decompose(W_t, A_calib_t, rank)
    elif method == "r_only":
        decomp = r_only_decompose(W_t, A_calib_t, rank)
    else:
        raise ValueError(f"Unknown method: {method}")

    B_r = decomp["B_r"]
    A_fac = decomp["A_fac"]
    W_res = decomp["W_res"]

    # Deploy: Y ≈ A_t @ (A_fac @ B_r.T) + Q(A_t) @ Q(W_res).T
    # Low-rank branch in float, main branch quantized
    Y_main = nvfp4_quantize(A_eval_t) @ nvfp4_quantize_2d(W_res).T
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
