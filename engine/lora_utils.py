"""
LoRA utilities for parsing and applying LoRA weights to models.
Supports kohya-ss format (standard for SDXL LoRA files).
"""

import torch
from typing import Optional
from safetensors import safe_open


# LoRA key mapping patterns for SDXL
LORA_UNET_PREFIX = "lora_unet_"
LORA_TE1_PREFIX = "lora_te1_"
LORA_TE2_PREFIX = "lora_te2_"


def _convert_lora_key_to_model_key(lora_key: str) -> str:
    """Convert a LoRA key name to the corresponding model weight key."""
    # Remove lora suffix (.lora_down.weight, .lora_up.weight, .alpha)
    base_key = lora_key
    for suffix in [".lora_down.weight", ".lora_up.weight", ".alpha"]:
        if base_key.endswith(suffix):
            base_key = base_key[:-len(suffix)]
            break

    # Convert lora_unet_ prefix to model.diffusion_model.
    if base_key.startswith(LORA_UNET_PREFIX):
        model_key = base_key[len(LORA_UNET_PREFIX):]
        # Replace underscores that represent dots in the original key
        # LoRA keys use double underscores for dots in layer names
        model_key = model_key.replace("_", ".")
        # Fix specific patterns for SDXL UNet
        model_key = _fix_unet_key(model_key)
        model_key = "model.diffusion_model." + model_key
    elif base_key.startswith(LORA_TE1_PREFIX):
        model_key = base_key[len(LORA_TE1_PREFIX):]
        model_key = model_key.replace("_", ".")
        model_key = "cond_stage_model.transformer." + model_key
    elif base_key.startswith(LORA_TE2_PREFIX):
        model_key = base_key[len(LORA_TE2_PREFIX):]
        model_key = model_key.replace("_", ".")
        model_key = "conditioner.embedders.1.model." + model_key
    else:
        model_key = base_key

    return model_key


def _fix_unet_key(key: str) -> str:
    """Fix common key conversion issues for SDXL UNet."""
    import re
    
    # Fix block numbering: input.blocks.0 -> input_blocks.0
    key = re.sub(r'input\.blocks\.(\d+)', r'input_blocks.\1', key)
    key = re.sub(r'output\.blocks\.(\d+)', r'output_blocks.\1', key)
    key = re.sub(r'middle\.block\.(\d+)', r'middle_block.\1', key)
    key = re.sub(r'time\.embed', 'time_embed', key)
    key = re.sub(r'label\.emb', 'label_emb', key)
    
    # Fix transformer blocks
    key = key.replace("transformer.blocks", "transformer_blocks")
    
    # Fix projection names
    key = key.replace(".proj.in.", ".proj_in.")
    key = key.replace(".proj.out.", ".proj_out.")
    key = key.replace(".to.q.", ".to_q.")
    key = key.replace(".to.k.", ".to_k.")
    key = key.replace(".to.v.", ".to_v.")
    key = key.replace(".to.out.0.", ".to_out.0.")
    
    return key


def parse_lora_file(filepath: str) -> dict:
    """
    Parse a LoRA safetensors file and organize weights by target layer.
    
    Returns dict of:
    {
        model_key: {
            "up": tensor,
            "down": tensor,
            "alpha": float,
            "rank": int,
        }
    }
    """
    lora_weights = {}
    
    with safe_open(filepath, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # Group keys by their base name
        grouped = {}
        for key in keys:
            if key.endswith(".lora_down.weight"):
                base = key[:-len(".lora_down.weight")]
                if base not in grouped:
                    grouped[base] = {}
                grouped[base]["down"] = f.get_tensor(key)
            elif key.endswith(".lora_up.weight"):
                base = key[:-len(".lora_up.weight")]
                if base not in grouped:
                    grouped[base] = {}
                grouped[base]["up"] = f.get_tensor(key)
            elif key.endswith(".alpha"):
                base = key[:-len(".alpha")]
                if base not in grouped:
                    grouped[base] = {}
                grouped[base]["alpha_tensor"] = f.get_tensor(key)
        
        for lora_key, weights in grouped.items():
            if "up" not in weights or "down" not in weights:
                continue
            
            model_key = _convert_lora_key_to_model_key(lora_key + ".lora_down.weight")
            rank = weights["down"].shape[0]
            alpha = weights.get("alpha_tensor", torch.tensor(float(rank))).item()
            
            lora_weights[model_key] = {
                "up": weights["up"],
                "down": weights["down"],
                "alpha": alpha,
                "rank": rank,
            }
    
    return lora_weights


def compute_lora_delta(up: torch.Tensor, down: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    """Compute the weight delta for a single LoRA layer."""
    scale = alpha / rank
    
    # Handle different dimensionalities
    if down.dim() == 4:  # Conv2d
        # down: (rank, in_ch, kh, kw), up: (out_ch, rank, 1, 1)
        delta = torch.einsum("i j k l, m i -> m j k l", down.float(), up.float())
    elif down.dim() == 2:  # Linear
        # down: (rank, in_features), up: (out_features, rank)
        delta = up.float() @ down.float()
    else:
        # Fallback: try matrix multiply on first 2 dims
        delta = up.float() @ down.float()
    
    return delta * scale


def apply_lora_to_state_dict(
    state_dict: dict[str, torch.Tensor],
    lora_filepath: str,
    strength: float = 1.0,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Apply a LoRA file to a model state dict.
    Modifies weights in-place and returns the state dict.
    """
    lora_weights = parse_lora_file(lora_filepath)
    
    applied_count = 0
    skipped_keys = []
    
    for model_key, lora_data in lora_weights.items():
        # Try to find the matching key with .weight suffix
        target_key = model_key
        if target_key not in state_dict:
            target_key = model_key + ".weight"
        if target_key not in state_dict:
            # Try without the prefix transformations
            skipped_keys.append(model_key)
            continue
        
        delta = compute_lora_delta(
            lora_data["up"].to(device),
            lora_data["down"].to(device),
            lora_data["alpha"],
            lora_data["rank"],
        )
        
        target_tensor = state_dict[target_key].to(device).float()
        
        if delta.shape != target_tensor.shape:
            # Shape mismatch — try to reshape
            try:
                delta = delta.reshape(target_tensor.shape)
            except RuntimeError:
                skipped_keys.append(model_key)
                continue
        
        state_dict[target_key] = (target_tensor + delta * strength).to(state_dict[target_key].dtype)
        applied_count += 1
    
    return state_dict


def apply_lora_to_tensor(
    tensor: torch.Tensor,
    tensor_key: str,
    lora_weights: dict,
    strength: float = 1.0,
) -> torch.Tensor:
    """
    Apply LoRA to a single tensor (for streaming/low-VRAM mode).
    lora_weights should be pre-parsed via parse_lora_file().
    """
    # Check both with and without .weight suffix
    lora_key = tensor_key
    if lora_key not in lora_weights:
        lora_key = tensor_key.rstrip(".weight") if tensor_key.endswith(".weight") else tensor_key
    if lora_key not in lora_weights:
        return tensor
    
    lora_data = lora_weights[lora_key]
    device = tensor.device
    
    delta = compute_lora_delta(
        lora_data["up"].to(device),
        lora_data["down"].to(device),
        lora_data["alpha"],
        lora_data["rank"],
    )
    
    if delta.shape != tensor.shape:
        try:
            delta = delta.reshape(tensor.shape)
        except RuntimeError:
            return tensor
    
    return (tensor.float() + delta * strength).to(tensor.dtype)


def get_lora_info(filepath: str) -> dict:
    """Get information about a LoRA file without loading all weights."""
    info = {
        "path": filepath,
        "layers": 0,
        "unet_layers": 0,
        "te1_layers": 0,
        "te2_layers": 0,
        "ranks": set(),
    }
    
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.endswith(".lora_down.weight"):
                info["layers"] += 1
                rank = f.get_tensor(key).shape[0]
                info["ranks"].add(rank)
                
                if key.startswith(LORA_UNET_PREFIX):
                    info["unet_layers"] += 1
                elif key.startswith(LORA_TE1_PREFIX):
                    info["te1_layers"] += 1
                elif key.startswith(LORA_TE2_PREFIX):
                    info["te2_layers"] += 1
    
    info["ranks"] = sorted(list(info["ranks"]))
    return info
