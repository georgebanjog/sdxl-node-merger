"""
VAE utilities for extracting and replacing VAE components in SDXL models.
"""

import torch
from typing import Optional

# VAE key prefix in SDXL models
VAE_PREFIX = "first_stage_model."


def get_vae_keys(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Get all VAE-related keys from a model state dict."""
    return [k for k in state_dict.keys() if k.startswith(VAE_PREFIX)]


def get_non_vae_keys(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Get all non-VAE keys from a model state dict."""
    return [k for k in state_dict.keys() if not k.startswith(VAE_PREFIX)]


def extract_vae(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract VAE weights from a full model state dict."""
    return {k: v for k, v in state_dict.items() if k.startswith(VAE_PREFIX)}


def replace_vae(
    model_state_dict: dict[str, torch.Tensor],
    vae_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Replace VAE weights in a model state dict.
    The VAE state dict can either have 'first_stage_model.' prefix or not.
    """
    result = {}
    
    # Copy all non-VAE keys from model
    for key, value in model_state_dict.items():
        if not key.startswith(VAE_PREFIX):
            result[key] = value
    
    # Check if VAE dict has the prefix
    has_prefix = any(k.startswith(VAE_PREFIX) for k in vae_state_dict.keys())
    
    if has_prefix:
        # VAE dict already has the correct prefix
        for key, value in vae_state_dict.items():
            if key.startswith(VAE_PREFIX):
                result[key] = value
    else:
        # Add prefix to VAE keys
        for key, value in vae_state_dict.items():
            result[VAE_PREFIX + key] = value
    
    return result


def replace_vae_streaming(
    model_keys: list[str],
    model_path: str,
    vae_path: str,
    load_tensor_func,
) -> callable:
    """
    Create a tensor generator that replaces VAE on-the-fly (for streaming mode).
    
    Returns a function(key) -> tensor that loads from model or VAE as appropriate.
    """
    from safetensors import safe_open
    
    # Determine VAE key format
    with safe_open(vae_path, framework="pt") as f:
        vae_keys = list(f.keys())
    
    has_prefix = any(k.startswith(VAE_PREFIX) for k in vae_keys)
    
    def get_tensor(key: str) -> torch.Tensor:
        if key.startswith(VAE_PREFIX):
            # This is a VAE key — load from VAE file
            vae_key = key if has_prefix else key[len(VAE_PREFIX):]
            if vae_key in vae_keys or key in vae_keys:
                lookup_key = key if key in vae_keys else vae_key
                with safe_open(vae_path, framework="pt") as f:
                    return f.get_tensor(lookup_key)
        
        # Non-VAE key or VAE key not in VAE file — load from model
        return load_tensor_func(model_path, key)
    
    return get_tensor
