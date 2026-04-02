"""
Tensor I/O module for safetensors files.
Handles reading, writing, and dtype conversion of model weights.
"""

import os
import json
import struct
import mmap
from typing import Optional, Generator
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file, load_file


# SDXL block structure for MBW (Merge Block Weighted)
SDXL_UNET_BLOCKS = {
    "input_blocks": [f"model.diffusion_model.input_blocks.{i}." for i in range(9)],
    "middle_block": ["model.diffusion_model.middle_block."],
    "output_blocks": [f"model.diffusion_model.output_blocks.{i}." for i in range(9)],
    "time_embed": ["model.diffusion_model.time_embed."],
    "label_emb": ["model.diffusion_model.label_emb."],
    "out": ["model.diffusion_model.out."],
}

# All 26 MBW positions for SDXL: IN00-IN08, MID, OUT00-OUT08, TE1, TE2, TIME, LABEL, OUT_FINAL, VAE
MBW_BLOCK_IDS = [
    "IN00", "IN01", "IN02", "IN03", "IN04", "IN05", "IN06", "IN07", "IN08",
    "MID",
    "OUT00", "OUT01", "OUT02", "OUT03", "OUT04", "OUT05", "OUT06", "OUT07", "OUT08",
    "TE1", "TE2", "TIME_EMBED", "LABEL_EMB", "OUT_FINAL", "VAE"
]

DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

DTYPE_NAMES = {
    torch.float16: "fp16",
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
}


def get_block_id_for_key(key: str) -> str:
    """Determine which MBW block a tensor key belongs to."""
    if key.startswith("first_stage_model."):
        return "VAE"
    if key.startswith("cond_stage_model.") or key.startswith("conditioner.embedders.0."):
        return "TE1"
    if key.startswith("conditioner.embedders.1."):
        return "TE2"
    if "time_embed" in key:
        return "TIME_EMBED"
    if "label_emb" in key:
        return "LABEL_EMB"
    if "middle_block" in key:
        return "MID"

    for i in range(9):
        prefix = f"model.diffusion_model.input_blocks.{i}."
        if key.startswith(prefix):
            return f"IN{i:02d}"

    for i in range(9):
        prefix = f"model.diffusion_model.output_blocks.{i}."
        if key.startswith(prefix):
            return f"OUT{i:02d}"

    if key.startswith("model.diffusion_model.out."):
        return "OUT_FINAL"

    # Default to MID for unknown diffusion model keys
    if key.startswith("model.diffusion_model."):
        return "MID"

    return "OTHER"


def scan_directory(directory: str, extensions: tuple = (".safetensors",)) -> list[dict]:
    """Scan a directory for model files and return their info."""
    results = []
    if not directory or not os.path.isdir(directory):
        return results

    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, directory)
                try:
                    size_bytes = os.path.getsize(full_path)
                    size_gb = size_bytes / (1024 ** 3)
                    results.append({
                        "name": fname,
                        "path": rel_path,
                        "full_path": full_path,
                        "size": f"{size_gb:.2f} GB" if size_gb >= 1 else f"{size_bytes / (1024**2):.0f} MB",
                        "size_bytes": size_bytes,
                    })
                except OSError:
                    continue
    return results


def read_metadata(filepath: str) -> dict:
    """Read metadata from a safetensors file without loading tensors."""
    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            metadata = header.get("__metadata__", {})
            # Count tensors and their dtypes
            tensor_count = 0
            dtypes = set()
            total_params = 0
            for key, value in header.items():
                if key != "__metadata__":
                    tensor_count += 1
                    dtypes.add(value.get("dtype", "unknown"))
                    shape = value.get("shape", [])
                    params = 1
                    for s in shape:
                        params *= s
                    total_params += params

            return {
                "metadata": metadata,
                "tensor_count": tensor_count,
                "dtypes": list(dtypes),
                "total_params": total_params,
                "file_size": os.path.getsize(filepath),
            }
    except Exception as e:
        return {"error": str(e)}


def get_tensor_keys(filepath: str) -> list[str]:
    """Get all tensor key names from a safetensors file."""
    with safe_open(filepath, framework="pt") as f:
        return list(f.keys())


def load_model_full(filepath: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load entire model into memory."""
    return load_file(filepath, device=device)


def load_tensor(filepath: str, key: str, device: str = "cpu") -> torch.Tensor:
    """Load a single tensor from a safetensors file (memory efficient)."""
    with safe_open(filepath, framework="pt", device=device) as f:
        return f.get_tensor(key)


def iterate_tensors(filepath: str, device: str = "cpu") -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over tensors one at a time (memory efficient)."""
    with safe_open(filepath, framework="pt", device=device) as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


def save_model(
    state_dict: dict[str, torch.Tensor],
    filepath: str,
    dtype: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
):
    """Save a model state dict to safetensors format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

    if dtype and dtype in DTYPE_MAP:
        target_dtype = DTYPE_MAP[dtype]
        converted = {}
        for key, tensor in state_dict.items():
            if tensor.is_floating_point():
                converted[key] = tensor.to(dtype=target_dtype)
            else:
                converted[key] = tensor
        state_dict = converted

    # Move all tensors to CPU for saving
    cpu_dict = {}
    for key, tensor in state_dict.items():
        cpu_dict[key] = tensor.cpu().contiguous()

    save_file(cpu_dict, filepath, metadata=metadata)


def save_model_streaming(
    filepath: str,
    keys: list[str],
    tensor_generator,
    dtype: Optional[str] = None,
    metadata: Optional[dict[str, str]] = None,
):
    """
    Save model by streaming tensors one at a time.
    Used in low-VRAM mode to avoid holding everything in memory.
    
    tensor_generator: callable(key) -> torch.Tensor
    """
    # We need to collect all tensors for safetensors format
    # But we process and convert them one at a time
    state_dict = {}
    target_dtype = DTYPE_MAP.get(dtype) if dtype else None

    for key in keys:
        tensor = tensor_generator(key)
        if target_dtype and tensor.is_floating_point():
            tensor = tensor.to(dtype=target_dtype)
        state_dict[key] = tensor.cpu().contiguous()

    save_file(state_dict, filepath, metadata=metadata)


def get_model_type_info(filepath: str) -> dict:
    """Analyze model file to determine its type and structure."""
    keys = get_tensor_keys(filepath)
    
    has_unet = any(k.startswith("model.diffusion_model.") for k in keys)
    has_vae = any(k.startswith("first_stage_model.") for k in keys)
    has_te1 = any(k.startswith("cond_stage_model.") or k.startswith("conditioner.embedders.0.") for k in keys)
    has_te2 = any(k.startswith("conditioner.embedders.1.") for k in keys)
    
    # Check if it's a LoRA
    is_lora = any("lora" in k.lower() for k in keys)
    
    # Check if it's a standalone VAE
    is_standalone_vae = has_vae and not has_unet
    
    model_type = "unknown"
    if is_lora:
        model_type = "lora"
    elif is_standalone_vae:
        model_type = "vae"
    elif has_unet and has_te2:
        model_type = "sdxl"
    elif has_unet and has_te1:
        model_type = "sd15"
    elif has_unet:
        model_type = "unet_only"
    
    block_counts = {}
    for key in keys:
        block_id = get_block_id_for_key(key)
        block_counts[block_id] = block_counts.get(block_id, 0) + 1
    
    return {
        "type": model_type,
        "has_unet": has_unet,
        "has_vae": has_vae,
        "has_te1": has_te1,
        "has_te2": has_te2,
        "is_lora": is_lora,
        "total_keys": len(keys),
        "block_counts": block_counts,
    }
