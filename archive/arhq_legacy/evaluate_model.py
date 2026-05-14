"""Legacy full-model quantized inference simulation.

Use ``python -m arhq.eval_quantized`` for the minimal codebase flow.

Usage:
  # Perplexity evaluation on wikitext2
  CUDA_VISIBLE_DEVICES=2 conda run -n llmc --no-capture-output python -m archive.arhq_legacy.evaluate_model \
      --method arhq --setting raw --rank 128 --device cuda:0

  # Compare all methods
  CUDA_VISIBLE_DEVICES=2 conda run -n llmc --no-capture-output python -m archive.arhq_legacy.evaluate_model \
      --method all --rank 128 --device cuda:0
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from arhq.quant import nvfp4_quantize

MODEL_PATH = os.path.expanduser("/home/wangyifeng/work/models/Qwen3-4B-Thinking-2507")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
NUM_LAYERS = 36


class QuantizedLowRankLinear(nn.Module):
    """Simulated quantized linear with low-rank compensation.

    Y = Q(A') @ Q(W_res)^T + (A' @ A_fac) @ B_r^T + bias
    where A' = (A - mean) @ T  if transform is provided, else A' = A
    """

    def __init__(self, W_res: torch.Tensor, B_r: torch.Tensor, A_fac: torch.Tensor,
                 T: torch.Tensor = None, T_inv: torch.Tensor = None,
                 mean: torch.Tensor = None, bias_W: torch.Tensor = None):
        super().__init__()
        # Store as parameters for device movement
        self.register_buffer("W_res", W_res)
        self.register_buffer("B_r", B_r)
        self.register_buffer("A_fac", A_fac)
        self.has_transform = T is not None
        if T is not None:
            self.register_buffer("T", T)
            self.register_buffer("mean", mean)
            self.register_buffer("bias_vec", bias_W)  # [D_out], precomputed mean @ W.T
        else:
            self.T = None
            self.mean = None
            self.bias_vec = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])

        if self.has_transform:
            x_t = (x - self.mean) @ self.T
        else:
            x_t = x

        # Main quantized branch
        y_main = nvfp4_quantize(x_t.float()).to(x_t.dtype) @ nvfp4_quantize(self.W_res.float()).to(self.W_res.dtype).T
        # Low-rank branch (float)
        y_lr = (x_t @ self.A_fac) @ self.B_r.T
        y = y_main + y_lr

        if self.has_transform:
            y = y + self.bias_vec

        if len(orig_shape) == 3:
            y = y.reshape(orig_shape[0], orig_shape[1], -1)
        return y


class QuantizedOnlyLinear(nn.Module):
    """Simulated quantized linear without low-rank (baseline)."""

    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        y = nvfp4_quantize(x.float()).to(x.dtype) @ nvfp4_quantize(self.W.float()).to(self.W.dtype).T
        if len(orig_shape) == 3:
            y = y.reshape(orig_shape[0], orig_shape[1], -1)
        return y


def load_decomposition(layer_idx: int, proj: str, method: str, setting: str,
                       rank: int) -> dict:
    """Load saved decomposition parameters."""
    path = os.path.join(RESULTS_DIR, "layer_results", f"layer_{layer_idx}",
                        f"{proj}_{method}_{setting}_rank{rank}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def replace_attention_projections(model, method: str, setting: str, rank: int,
                                  device: str):
    """Replace attention q/k/v/o_proj with quantized low-rank modules."""
    replaced = 0
    for layer_idx in range(NUM_LAYERS):
        layer = model.model.layers[layer_idx]
        for proj in PROJ_TYPES:
            orig_linear = getattr(layer.self_attn, proj)
            W_orig = orig_linear.weight.data.float()

            if method == "baseline":
                new_mod = QuantizedOnlyLinear(W_orig.half())
            else:
                decomp = load_decomposition(layer_idx, proj, method, setting, rank)
                if decomp is None:
                    print(f"  WARNING: no decomposition for L{layer_idx} {proj}, using baseline")
                    new_mod = QuantizedOnlyLinear(W_orig.half())
                else:
                    B_r = decomp["B_r"].float()
                    A_fac = decomp["A_fac"].float()
                    W_res = decomp["W_res"].float()

                    if "T" in decomp and decomp["T"] is not None:
                        T = decomp["T"].float()
                        mean = decomp["mean"].float()
                        bias_W = (mean @ W_orig.T).float()
                        new_mod = QuantizedLowRankLinear(
                            W_res.half(), B_r.half(), A_fac.half(),
                            T.half(), None, mean.half(), bias_W.half()
                        )
                    else:
                        new_mod = QuantizedLowRankLinear(
                            W_res.half(), B_r.half(), A_fac.half()
                        )

            new_mod = new_mod.to(device)
            setattr(layer.self_attn, proj, new_mod)
            replaced += 1

    print(f"Replaced {replaced} attention projections with {method} (rank={rank}, setting={setting})")


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, device: str, max_length: int = 2048,
                        stride: int = 512):
    """Evaluate perplexity on wikitext2 test set."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    seq_len = input_ids.size(1)
    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_chunk = input_ids[:, begin:end]

        outputs = model(input_chunk)
        logits = outputs.logits

        # Only compute loss on the new tokens (not the overlapping context)
        shift_logits = logits[:, -target_len:-1, :].contiguous()
        shift_labels = input_ids[:, prev_end + 1:end].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        nlls.append(loss.item() * target_len)

        prev_end = end
        if end == seq_len:
            break

    ppl = torch.exp(torch.tensor(sum(nlls) / prev_end))
    return ppl.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all",
                        help="'svdquant', 'arhq', 'baseline', or 'all'")
    parser.add_argument("--setting", default="raw", choices=["raw", "hadazca"])
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--original", action="store_true",
                        help="Also evaluate original (unquantized) model")
    args = parser.parse_args()

    methods = ["baseline", "svdquant", "arhq"] if args.method == "all" else [args.method]
    if args.original:
        methods = ["original"] + methods

    print(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating: {method} (rank={args.rank}, setting={args.setting})")
        print(f"{'='*60}")

        print(f"Loading model from {MODEL_PATH}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map=args.device, trust_remote_code=True,
        )
        model.eval()

        if method != "original":
            replace_attention_projections(model, method, args.setting, args.rank,
                                         args.device)

        t0 = time.time()
        ppl = evaluate_perplexity(model, tokenizer, args.device)
        elapsed = time.time() - t0
        results[method] = ppl
        print(f"  Perplexity: {ppl:.4f}  ({elapsed:.1f}s)")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for method, ppl in results.items():
        print(f"  {method:12s}: PPL = {ppl:.4f}")


if __name__ == "__main__":
    main()
