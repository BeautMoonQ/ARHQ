"""Export ARHQ/SVDQuant nvfp4+LoRA model artifacts.

The exported artifact is intentionally explicit rather than pretending to be a
plain Hugging Face checkpoint: each transformed linear projection stores a
2D-block-nvfp4-simulated residual weight plus the floating-point LoRA branch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

from arhq.quant import nvfp4_pack_2d, nvfp4_quantize_2d

try:
    from safetensors.torch import save_file
except Exception as exc:  # pragma: no cover - checked at runtime
    save_file = None
    _SAFETENSORS_IMPORT_ERROR = exc
else:
    _SAFETENSORS_IMPORT_ERROR = None


ATTN_PROJ_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
FFN_PROJ_TYPES = ["gate_proj", "up_proj", "down_proj"]
MODULE_SETS = {
    "attn": ATTN_PROJ_TYPES,
    "ffn": FFN_PROJ_TYPES,
    "all": ATTN_PROJ_TYPES + FFN_PROJ_TYPES,
}
NUM_LAYERS = 36
DEFAULT_EXPORTS = [
    "svdquant:smoothing:svdquant_smoothing_rank128_fp16:fp16",
    "svdquant:smoothing:svdquant_smoothing_rank128_packed4bit:packed4bit",
    "arhq:raw:arhq_raw_rank128_fp16:fp16",
    "arhq:raw:arhq_raw_rank128_packed4bit:packed4bit",
    "arhq:smoothing:arhq_smoothing_rank128_fp16:fp16",
    "arhq:smoothing:arhq_smoothing_rank128_packed4bit:packed4bit",
]


@dataclass(frozen=True)
class ExportSpec:
    method: str
    setting: str
    name: str
    precision: str = "both"


def parse_export_specs(spec: str) -> list[ExportSpec]:
    items = DEFAULT_EXPORTS if not spec else [x.strip() for x in spec.split(",")]
    out = []
    for item in items:
        parts = item.split(":")
        if len(parts) == 3:
            method, setting, name = parts
            precision = "both"
        elif len(parts) == 4:
            method, setting, name, precision = parts
        else:
            raise ValueError(
                f"export spec must be method:setting:name[:precision], got: {item}"
            )
        if method not in ("arhq", "r_only", "svdquant"):
            raise ValueError(f"unsupported method: {method}")
        if setting not in ("raw", "smoothing"):
            raise ValueError(f"unsupported setting: {setting}")
        if precision not in ("fp16", "packed4bit", "both"):
            raise ValueError(f"unsupported precision: {precision}")
        out.append(
            ExportSpec(
                method=method,
                setting=setting,
                name=name,
                precision=precision,
            )
        )
    return out


def parse_layers(spec: str) -> list[int]:
    if not spec:
        return list(range(NUM_LAYERS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return out


def get_projection_module(layer, proj: str):
    if proj in ATTN_PROJ_TYPES:
        return getattr(layer.self_attn, proj)
    if proj in FFN_PROJ_TYPES:
        return getattr(layer.mlp, proj)
    raise ValueError(f"unknown projection: {proj}")


def find_decomp_path(decomp_dir: str, layer_idx: int, proj: str,
                     method: str, setting: str, rank: int) -> tuple[str, str]:
    method_candidates = [method]
    if method == "arhq":
        # Older artifacts for the current ARHQ method were saved as r_only.
        method_candidates.append("r_only")
    elif method == "r_only":
        method_candidates.append("arhq")

    for method_name in method_candidates:
        path = os.path.join(
            decomp_dir,
            f"layer_{layer_idx}",
            f"{proj}_{method_name}_{setting}_rank{rank}.pt",
        )
        if os.path.exists(path):
            return path, method_name
    expected = os.path.join(
        decomp_dir, f"layer_{layer_idx}", f"{proj}_{method}_{setting}_rank{rank}.pt"
    )
    raise FileNotFoundError(expected)


def copy_model_metadata(model_path: str, export_dir: str) -> list[str]:
    copied = []
    for name in [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "LICENSE",
    ]:
        src = os.path.join(model_path, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(export_dir, name))
            copied.append(name)
    return copied


def write_format_docs(export_dir: str, manifest: dict):
    method = manifest["method"]
    setting = manifest["setting"]
    rank = manifest["rank"]
    module_set = manifest["module_set"]
    precision = manifest["artifact_precision"]
    num_layers = manifest["num_layers"]
    num_projections = manifest["num_projections"]
    weight_block = manifest["quantization"]["weight"]["block_shape"]
    activation_block = manifest["quantization"]["activation"]["block_shape"]
    has_smoothing = setting == "smoothing"
    readme = f"""# ARHQ NVFP4 + LoRA Artifact

This directory contains an exported `{method}:{setting}` artifact with rank `{rank}` for module set `{module_set}`.

It is not a plain Hugging Face checkpoint. The artifact stores the tensors needed by the ARHQ/SVDQuant simulated nvfp4 + LoRA runtime:

```text
Y = Qx(X_or_X_scaled) @ W_res_q.T + (X_or_X_scaled @ A_fac) @ B_r.T
```

Summary:

- Layers: `{num_layers}`
- Projections: `{num_projections}`
- Activation quantization: simulated NVFP4 E2M1, block shape `{activation_block}`
- Weight quantization: simulated NVFP4 E2M1, 2D block shape `{weight_block}`
- LoRA rank: `{rank}`
- Stored residual format: `{precision}`
- Smoothing: `{"enabled" if has_smoothing else "disabled"}`

See `FORMAT.md` for the full tensor layout and `manifest.json` for machine-readable metadata.
"""
    fmt = f"""# Artifact Format

## Directory Layout

```text
manifest.json
FORMAT.md
layers/
  layer_0.safetensors
  ...
  layer_35.safetensors
```

Model/tokenizer metadata copied from the base model may also be present, for example `config.json`, `tokenizer.json`, and `generation_config.json`.

## Shards

Each `layers/layer_<idx>.safetensors` file contains all exported projections for one transformer layer.

For each projection `<proj>`, the shard contains:

```text
<proj>.B_r       # [D_out, rank], fp16 LoRA left factor
<proj>.A_fac     # [D_in, rank], fp16 LoRA right factor
<proj>.scale     # [D_in], fp16 smoothing scale, only when setting=smoothing
```

If `artifact_precision=fp16`, each projection also contains:

```text
<proj>.W_res_q   # [D_out, D_in], fp16 dequantized simulated NVFP4 residual weight
```

If `artifact_precision=packed4bit`, each projection also contains:

```text
<proj>.W_res_packed       # [ceil(D_out_pad * D_in_pad / 2)], uint8 packed 4-bit FP4 code indices
<proj>.W_res_scale        # [ceil(D_out/16), ceil(D_in/16)], fp16 simulated FP8 E4M3 block scales
<proj>.W_res_orig_shape   # [2], int32 original [D_out, D_in]
<proj>.W_res_padded_shape # [2], int32 padded [D_out_pad, D_in_pad]
<proj>.W_res_block_shape  # [2], int32 block shape, normally [16, 16]
```

If `artifact_precision=both`, both groups are stored in the same shard.

Projection names follow the model module names, e.g. `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

## Runtime Formula

For `setting=raw`:

```text
Y = Qx(X) @ W_res_q.T + (X @ A_fac) @ B_r.T
```

For `setting=smoothing`:

```text
X_s = X / scale
Y = Qx(X_s) @ W_res_q.T + (X_s @ A_fac) @ B_r.T
```

`W_res_q` is already the dequantized simulated NVFP4 residual weight and should not be quantized again for this simulation format.

`W_res_packed` and `W_res_scale` store the packed 4-bit version of the same residual weight. Two 4-bit FP4 code indices are packed into each byte:

```text
low nibble  = first FP4 code index
high nibble = second FP4 code index
```

The FP4 codebook index order is:

```text
[-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
```

## Quantization Layout

Activations use simulated NVFP4 E2M1 with 1D block scaling:

```text
activation block shape = {activation_block}
```

Weights use simulated NVFP4 E2M1 with Transformer-Engine-style 2D block scaling:

```text
weight block shape = {weight_block}
scale dtype = simulated FP8 E4M3
```

The artifact stores both forms:

- `artifact_precision=fp16`: stores `W_res_q`.
- `artifact_precision=packed4bit`: stores `W_res_packed` + `W_res_scale` and shape metadata.
- `artifact_precision=both`: stores both representations in one shard.

## Manifest

`manifest.json` contains:

- base model path
- method, setting, rank, module set
- quantization metadata
- source decomposition file for each projection
- tensor names and tensor shapes
- source method counts, useful because older ARHQ decomposition files may be stored under the legacy `r_only` method name
"""
    with open(os.path.join(export_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    with open(os.path.join(export_dir, "FORMAT.md"), "w", encoding="utf-8") as f:
        f.write(fmt)


def build_runtime_tensors(W_orig: torch.Tensor, decomp: dict, setting: str):
    W_orig = W_orig.float()
    B_r = decomp["B_r"].float()
    A_fac = decomp["A_fac"].float()
    scale: Optional[torch.Tensor] = None
    if setting == "smoothing":
        if "scale" not in decomp:
            raise ValueError("smoothing decomposition is missing `scale`")
        scale = decomp["scale"].float()
        W_target = W_orig * scale
    else:
        W_target = W_orig

    W_res = W_target - B_r @ A_fac.T
    packed = nvfp4_pack_2d(W_res)
    W_res_q = nvfp4_quantize_2d(W_res).half().cpu().contiguous()
    return (
        W_res_q,
        packed,
        B_r.half().cpu().contiguous(),
        A_fac.half().cpu().contiguous(),
        scale.half().cpu().contiguous() if scale is not None else None,
    )


def export_one(model, model_path: str, decomp_dir: str, output_root: str,
               spec: ExportSpec, rank: int, module_set: str, layers: list[int]):
    if save_file is None:
        raise RuntimeError(f"safetensors is required: {_SAFETENSORS_IMPORT_ERROR}")

    proj_types = MODULE_SETS[module_set]
    store_fp16 = spec.precision in ("fp16", "both")
    store_packed = spec.precision in ("packed4bit", "both")
    export_dir = os.path.join(output_root, spec.name)
    os.makedirs(export_dir, exist_ok=True)
    os.makedirs(os.path.join(export_dir, "layers"), exist_ok=True)
    copied_files = copy_model_metadata(model_path, export_dir)

    entries = []
    fallback_methods: dict[str, int] = {}

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        layer_tensors = {}
        layer_entries = []
        for proj in proj_types:
            decomp_path, stored_method = find_decomp_path(
                decomp_dir, layer_idx, proj, spec.method, spec.setting, rank
            )
            fallback_methods[stored_method] = fallback_methods.get(stored_method, 0) + 1
            decomp = torch.load(decomp_path, map_location="cpu", weights_only=True)
            W_orig = get_projection_module(layer, proj).weight.detach().cpu()
            W_res_q, packed, B_r, A_fac, scale = build_runtime_tensors(
                W_orig, decomp, spec.setting
            )
            prefix = proj
            tensor_names = {}
            tensor_shapes = {}
            if store_fp16:
                layer_tensors[f"{prefix}.W_res_q"] = W_res_q
                tensor_names["W_res_q"] = f"{prefix}.W_res_q"
                tensor_shapes["W_res_q"] = list(W_res_q.shape)
            if store_packed:
                layer_tensors[f"{prefix}.W_res_packed"] = packed["packed"]
                layer_tensors[f"{prefix}.W_res_scale"] = packed["scale"]
                layer_tensors[f"{prefix}.W_res_orig_shape"] = packed["orig_shape"]
                layer_tensors[f"{prefix}.W_res_padded_shape"] = packed["padded_shape"]
                layer_tensors[f"{prefix}.W_res_block_shape"] = packed["block_shape"]
                tensor_names.update({
                    "W_res_packed": f"{prefix}.W_res_packed",
                    "W_res_scale": f"{prefix}.W_res_scale",
                    "W_res_orig_shape": f"{prefix}.W_res_orig_shape",
                    "W_res_padded_shape": f"{prefix}.W_res_padded_shape",
                    "W_res_block_shape": f"{prefix}.W_res_block_shape",
                })
                tensor_shapes.update({
                    "W_res_packed": list(packed["packed"].shape),
                    "W_res_scale": list(packed["scale"].shape),
                })
            layer_tensors[f"{prefix}.B_r"] = B_r
            layer_tensors[f"{prefix}.A_fac"] = A_fac
            tensor_names.update({
                "B_r": f"{prefix}.B_r",
                "A_fac": f"{prefix}.A_fac",
            })
            tensor_shapes.update({
                "B_r": list(B_r.shape),
                "A_fac": list(A_fac.shape),
            })
            if scale is not None:
                layer_tensors[f"{prefix}.scale"] = scale
                tensor_names["scale"] = f"{prefix}.scale"
                tensor_shapes["scale"] = list(scale.shape)
            layer_entries.append({
                "projection": proj,
                "stored_method": stored_method,
                "source_file": os.path.relpath(decomp_path, export_dir),
                "tensors": tensor_names,
                "shapes": tensor_shapes,
            })

        shard_name = f"layers/layer_{layer_idx}.safetensors"
        save_file(
            layer_tensors,
            os.path.join(export_dir, shard_name),
            metadata={
                "format": "arhq_nvfp4_lora",
                "layer": str(layer_idx),
                "method": spec.method,
                "setting": spec.setting,
                "rank": str(rank),
                "artifact_precision": spec.precision,
            },
        )
        for entry in layer_entries:
            entry["shard"] = shard_name
            entry["layer"] = layer_idx
            entries.append(entry)
        print(f"  {spec.name}: exported layer {layer_idx}")

    manifest = {
        "format": "arhq_nvfp4_lora",
        "format_version": 1,
        "base_model_path": os.path.abspath(os.path.expanduser(model_path)),
        "method": spec.method,
        "setting": spec.setting,
        "artifact_precision": spec.precision,
        "rank": rank,
        "module_set": module_set,
        "layers": layers,
        "num_layers": len(layers),
        "num_projections": len(entries),
        "quantization": {
            "activation": {
                "name": "nvfp4_e2m1_simulated",
                "block_shape": [16],
            },
            "weight": {
                "name": "nvfp4_e2m1_simulated_dequantized",
                "packed_name": "nvfp4_e2m1_simulated_packed_uint8_codes",
                "block_shape": [16, 16],
                "scale_dtype": "fp8_e4m3_simulated",
                "stored_scale_dtype": "float16",
                "packed_codes_per_byte": 2,
            },
        },
        "runtime_formula": "Y = Qx(X_or_X_scaled) @ W_res_q.T + (X_or_X_scaled @ A_fac) @ B_r.T",
        "metadata_files_copied": copied_files,
        "stored_method_counts": fallback_methods,
        "entries": entries,
    }
    with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_format_docs(export_dir, manifest)
    print(f"  wrote {os.path.join(export_dir, 'manifest.json')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ARHQ/SVDQuant nvfp4+LoRA model artifacts."
    )
    parser.add_argument("--model_path", default=os.path.expanduser(
        "~/work/models/Qwen3-4B-Thinking-2507"
    ))
    parser.add_argument("--decomp_dir", default="results/layer_results")
    parser.add_argument("--output_dir", default="models")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--module_set", default="all", choices=["attn", "ffn", "all"])
    parser.add_argument("--layers", default="0-35")
    parser.add_argument("--exports", default=",".join(DEFAULT_EXPORTS),
                        help="Comma-separated method:setting:name specs.")
    return parser.parse_args()


def main():
    args = parse_args()
    specs = parse_export_specs(args.exports)
    layers = parse_layers(args.layers)

    print(f"Base model: {args.model_path}")
    print(f"Decomp dir: {args.decomp_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Exports:    {specs}")
    print("Loading base model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()

    for spec in specs:
        print(f"\nExporting {spec.name} ({spec.method}:{spec.setting})")
        export_one(
            model=model,
            model_path=args.model_path,
            decomp_dir=args.decomp_dir,
            output_root=args.output_dir,
            spec=spec,
            rank=args.rank,
            module_set=args.module_set,
            layers=layers,
        )


if __name__ == "__main__":
    main()
