"""NVFP4 quantization simulation, extracted from reference_code/prepare.py."""

import torch


def _fp8_e4m3_sim(x: torch.Tensor) -> torch.Tensor:
    log2_x = torch.log2(x.abs().clamp(min=1e-45))
    exponent = log2_x.floor()
    mantissa_scale = 2.0 ** (exponent - 3)
    return (torch.round(x / mantissa_scale) * mantissa_scale).clamp(min=1e-12)


# FP4 E2M1 codebook (shared across functions)
_FP4_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _fp4_round(x_scaled: torch.Tensor) -> torch.Tensor:
    """Round to nearest FP4 E2M1 value. Input should be pre-scaled."""
    fp4_pos = torch.tensor(_FP4_POS, device=x_scaled.device, dtype=x_scaled.dtype)
    all_vals = torch.cat([-fp4_pos[1:].flip(0), fp4_pos])
    x_clamped = x_scaled.clamp(-6.0, 6.0)
    diffs = (x_clamped.unsqueeze(-1) - all_vals).abs()
    return all_vals[diffs.argmin(dim=-1)]


def nvfp4_quantize(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVFP4 (E2M1) quantization simulation, 1D per-block FP8 E4M3 scale."""
    original_shape = x.shape
    last_dim = x.shape[-1]
    pad_size = (block_size - last_dim % block_size) % block_size
    if pad_size > 0:
        x = torch.nn.functional.pad(x, (0, pad_size))

    x_flat = x.reshape(-1, block_size)
    amax = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / 6.0
    scale = _fp8_e4m3_sim(scale)

    x_quant = _fp4_round(x_flat / scale)
    x_dequant = (x_quant * scale).reshape(x.shape)
    if pad_size > 0:
        x_dequant = x_dequant[..., :last_dim]
    return x_dequant.reshape(original_shape)


def nvfp4_quantize_2d(W: torch.Tensor, block_h: int = 16, block_w: int = 16) -> torch.Tensor:
    """NVFP4 (E2M1) with 2D block scale (B100 Transformer Engine style).

    W: [D_out, D_in]. Each 16x16 tile shares one FP8 E4M3 scale.
    """
    D_out, D_in = W.shape
    pad_h = (block_h - D_out % block_h) % block_h
    pad_w = (block_w - D_in % block_w) % block_w
    if pad_h > 0 or pad_w > 0:
        W = torch.nn.functional.pad(W, (0, pad_w, 0, pad_h))

    H, Wi = W.shape
    # Reshape to [num_blocks_h, block_h, num_blocks_w, block_w]
    W_blocks = W.reshape(H // block_h, block_h, Wi // block_w, block_w)
    # amax per 2D block: [num_blocks_h, 1, num_blocks_w, 1]
    amax = W_blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale = amax / 6.0
    scale = _fp8_e4m3_sim(scale)

    W_quant = _fp4_round(W_blocks / scale)
    W_dequant = (W_quant * scale).reshape(H, Wi)

    return W_dequant[:D_out, :D_in]
