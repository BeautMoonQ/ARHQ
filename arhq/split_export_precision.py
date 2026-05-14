"""Split mixed ARHQ export artifacts into fp16 and packed-4bit directories."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy

from safetensors.torch import load_file, save_file

from arhq.export_model import write_format_docs


FP16_KEYS = {"W_res_q"}
PACKED_KEYS = {
    "W_res_packed",
    "W_res_scale",
    "W_res_orig_shape",
    "W_res_padded_shape",
    "W_res_block_shape",
}


def wanted_tensor(field: str, precision: str) -> bool:
    if field in {"B_r", "A_fac", "scale"}:
        return True
    if precision == "fp16":
        return field in FP16_KEYS
    if precision == "packed4bit":
        return field in PACKED_KEYS
    raise ValueError(f"unsupported precision: {precision}")


def rewrite_manifest(src_manifest: dict, dst_name: str, precision: str) -> dict:
    manifest = deepcopy(src_manifest)
    manifest["artifact_precision"] = precision
    manifest["entries"] = []
    for entry in src_manifest["entries"]:
        tensors = {
            field: name
            for field, name in entry["tensors"].items()
            if wanted_tensor(field, precision)
        }
        shapes = {
            field: shape
            for field, shape in entry["shapes"].items()
            if wanted_tensor(field, precision)
        }
        new_entry = deepcopy(entry)
        new_entry["tensors"] = tensors
        new_entry["shapes"] = shapes
        manifest["entries"].append(new_entry)
    manifest["source_mixed_artifact"] = src_manifest.get("artifact_name", "")
    manifest["artifact_name"] = dst_name
    return manifest


def copy_metadata(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    skip = {"layers", "manifest.json", "README.md", "FORMAT.md"}
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)


def split_one(src_dir: str, dst_dir: str, precision: str, overwrite: bool):
    if os.path.exists(dst_dir):
        if not overwrite:
            raise FileExistsError(dst_dir)
        shutil.rmtree(dst_dir)
    copy_metadata(src_dir, dst_dir)
    os.makedirs(os.path.join(dst_dir, "layers"), exist_ok=True)

    with open(os.path.join(src_dir, "manifest.json"), encoding="utf-8") as f:
        src_manifest = json.load(f)
    manifest = rewrite_manifest(src_manifest, os.path.basename(dst_dir), precision)

    for layer_idx in manifest["layers"]:
        shard_name = f"layers/layer_{layer_idx}.safetensors"
        src_shard = os.path.join(src_dir, shard_name)
        dst_shard = os.path.join(dst_dir, shard_name)
        tensors = load_file(src_shard)
        names = set()
        for entry in manifest["entries"]:
            if entry["layer"] == layer_idx:
                names.update(entry["tensors"].values())
        save_file(
            {name: tensors[name] for name in sorted(names)},
            dst_shard,
            metadata={
                "format": "arhq_nvfp4_lora",
                "layer": str(layer_idx),
                "method": manifest["method"],
                "setting": manifest["setting"],
                "rank": str(manifest["rank"]),
                "artifact_precision": precision,
            },
        )
        print(f"  {os.path.basename(dst_dir)}: wrote layer {layer_idx}")

    with open(os.path.join(dst_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    write_format_docs(dst_dir, manifest)
    print(f"  wrote {os.path.join(dst_dir, 'manifest.json')}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split mixed ARHQ/SVDQuant exports into fp16 and packed4bit artifacts."
    )
    parser.add_argument("--models_dir", default="models")
    parser.add_argument(
        "--sources",
        default="svdquant_smoothing_rank128,arhq_raw_rank128,arhq_smoothing_rank128",
        help="Comma-separated mixed export directory names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in [x.strip() for x in args.sources.split(",") if x.strip()]:
        src_dir = os.path.join(args.models_dir, name)
        split_one(src_dir, os.path.join(args.models_dir, f"{name}_fp16"), "fp16", args.overwrite)
        split_one(
            src_dir,
            os.path.join(args.models_dir, f"{name}_packed4bit"),
            "packed4bit",
            args.overwrite,
        )


if __name__ == "__main__":
    main()
