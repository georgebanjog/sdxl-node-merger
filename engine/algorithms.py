"""
Merge algorithms for SDXL model merging.
Each algorithm operates on tensors and returns the merged result.
"""

import torch
import math
from typing import Optional


# ─── Algorithm Registry ─────────────────────────────────────────────────────────

ALGORITHMS = {}


def register_algorithm(name: str, display_name: str, description: str, params: list, num_models: int = 2):
    """Decorator to register a merge algorithm."""
    def decorator(func):
        ALGORITHMS[name] = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "params": params,
            "num_models": num_models,
            "func": func,
        }
        return func
    return decorator


def get_algorithm_info() -> list[dict]:
    """Return info about all registered algorithms (without function references)."""
    result = []
    for name, info in ALGORITHMS.items():
        result.append({
            "name": info["name"],
            "display_name": info["display_name"],
            "description": info["description"],
            "params": info["params"],
            "num_models": info["num_models"],
        })
    return result


def merge_tensors(algorithm: str, tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    """Apply a merge algorithm to a list of tensors."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return ALGORITHMS[algorithm]["func"](tensors, params)


# ─── Parameter Definitions ───────────────────────────────────────────────────────

PARAM_ALPHA = {"name": "alpha", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "Alpha"}
PARAM_BETA = {"name": "beta", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "Beta"}
PARAM_DENSITY = {"name": "density", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "Density"}
PARAM_SEED = {"name": "seed", "type": "int", "min": 0, "max": 2**31, "default": 42, "label": "Seed"}
PARAM_DROP_RATE = {"name": "drop_rate", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "Drop Rate"}
PARAM_COSINE_THRESHOLD = {"name": "cosine_threshold", "type": "float", "min": 0.0, "max": 1.0, "default": 0.5, "step": 0.01, "label": "Cosine Threshold"}
PARAM_CLIP_DELTA = {"name": "clip_delta", "type": "bool", "default": False, "label": "Clip Deltas"}
PARAM_CLIP_VALUE = {"name": "clip_value", "type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "label": "Clip Value"}
PARAM_SMOOTH_SIGMA = {"name": "smooth_sigma", "type": "float", "min": 0.1, "max": 5.0, "default": 1.0, "step": 0.1, "label": "Smooth Sigma"}
PARAM_NORM_MEAN = {"name": "normalize_mean", "type": "bool", "default": True, "label": "Normalize Mean"}
PARAM_NORM_STD = {"name": "normalize_std", "type": "bool", "default": True, "label": "Normalize Std"}
PARAM_MAJORITY_METHOD = {"name": "majority_sign", "type": "select", "options": ["total", "frequency"], "default": "total", "label": "Majority Sign Method"}
PARAM_NORMALIZE = {"name": "normalize", "type": "bool", "default": False, "label": "Normalize"}


# ─── Algorithms ──────────────────────────────────────────────────────────────────

@register_algorithm(
    name="weighted_sum",
    display_name="Weighted Sum",
    description="Linear interpolation between two models: result = (1 - α) × A + α × B",
    params=[PARAM_ALPHA],
    num_models=2,
)
def weighted_sum(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    a, b = tensors[0], tensors[1]
    return torch.lerp(a.float(), b.float(), alpha).to(a.dtype)


@register_algorithm(
    name="add_difference",
    display_name="Add Difference",
    description="Add scaled difference: result = A + (B - C) × α",
    params=[PARAM_ALPHA],
    num_models=3,
)
def add_difference(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 1.0)
    a, b, c = tensors[0].float(), tensors[1].float(), tensors[2].float()
    return (a + (b - c) * alpha).to(tensors[0].dtype)


@register_algorithm(
    name="sum",
    display_name="Sum",
    description="Add model B scaled by alpha: result = A + B × α",
    params=[PARAM_ALPHA],
    num_models=2,
)
def sum_merge(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    a, b = tensors[0].float(), tensors[1].float()
    return (a + b * alpha).to(tensors[0].dtype)


@register_algorithm(
    name="tensor_sum",
    display_name="Tensor Sum",
    description="Weighted sum with optional normalization",
    params=[PARAM_ALPHA, PARAM_NORMALIZE],
    num_models=2,
)
def tensor_sum(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    normalize = params.get("normalize", False)
    a, b = tensors[0].float(), tensors[1].float()
    result = a * (1.0 - alpha) + b * alpha
    if normalize:
        norm_a = torch.norm(a)
        norm_result = torch.norm(result)
        if norm_result > 0:
            result = result * (norm_a / norm_result)
    return result.to(tensors[0].dtype)


@register_algorithm(
    name="ties",
    display_name="TIES Merge",
    description="TrIm, Elect Sign, and merge — preserves important weights",
    params=[PARAM_ALPHA, PARAM_DENSITY, PARAM_MAJORITY_METHOD],
    num_models=3,
)
def ties_merge(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    density = params.get("density", 0.5)
    majority_sign = params.get("majority_sign", "total")
    
    base, model_a, model_b = tensors[0].float(), tensors[1].float(), tensors[2].float()
    
    # Compute task vectors (deltas from base)
    delta_a = model_a - base
    delta_b = model_b - base
    
    # Step 1: Trim — keep only top-k% by magnitude
    deltas = [delta_a, delta_b]
    trimmed = []
    for delta in deltas:
        flat = delta.flatten().abs()
        if flat.numel() == 0:
            trimmed.append(delta)
            continue
        k = max(1, int(flat.numel() * density))
        threshold = torch.topk(flat, k).values[-1]
        mask = delta.abs() >= threshold
        trimmed.append(delta * mask.float())
    
    # Step 2: Elect sign
    stacked = torch.stack(trimmed, dim=0)
    if majority_sign == "total":
        sign_sum = stacked.sum(dim=0)
    else:  # frequency
        sign_sum = stacked.sign().sum(dim=0)
    
    elected_sign = sign_sum.sign()
    
    # Step 3: Disjoint merge — only keep values matching elected sign
    merged_delta = torch.zeros_like(base)
    count = torch.zeros_like(base)
    for t in trimmed:
        mask = (t.sign() == elected_sign).float()
        merged_delta += t * mask
        count += mask
    
    count = count.clamp(min=1)
    merged_delta /= count
    
    return (base + merged_delta * alpha).to(tensors[0].dtype)


@register_algorithm(
    name="dare",
    display_name="DARE Merge",
    description="Drop And REscale — randomly drops deltas with rescaling",
    params=[PARAM_ALPHA, PARAM_DROP_RATE, PARAM_SEED],
    num_models=3,
)
def dare_merge(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    drop_rate = params.get("drop_rate", 0.5)
    seed = params.get("seed", 42)
    
    base, model_a, model_b = tensors[0].float(), tensors[1].float(), tensors[2].float()
    
    # Compute deltas
    delta_a = model_a - base
    delta_b = model_b - base
    
    # Apply DARE to each delta
    generator = torch.Generator(device=base.device)
    
    dare_deltas = []
    for i, delta in enumerate([delta_a, delta_b]):
        generator.manual_seed(seed + i)
        # Create random mask (Bernoulli)
        mask = torch.bernoulli(
            torch.ones_like(delta) * (1.0 - drop_rate),
            generator=generator
        )
        # Rescale to preserve expected value
        rescale = 1.0 / (1.0 - drop_rate) if drop_rate < 1.0 else 0.0
        dare_deltas.append(delta * mask * rescale)
    
    # Merge the DARE deltas
    merged_delta = dare_deltas[0] * (1.0 - alpha) + dare_deltas[1] * alpha
    
    return (base + merged_delta).to(tensors[0].dtype)


@register_algorithm(
    name="cosine",
    display_name="Cosine Merge",
    description="Adaptive merging based on cosine similarity between tensors",
    params=[PARAM_ALPHA, PARAM_COSINE_THRESHOLD],
    num_models=2,
)
def cosine_merge(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    threshold = params.get("cosine_threshold", 0.5)
    
    a, b = tensors[0].float(), tensors[1].float()
    
    # Compute cosine similarity
    flat_a = a.flatten()
    flat_b = b.flatten()
    norm_a = torch.norm(flat_a)
    norm_b = torch.norm(flat_b)
    
    if norm_a > 0 and norm_b > 0:
        cos_sim = torch.dot(flat_a, flat_b) / (norm_a * norm_b)
        cos_sim = cos_sim.item()
    else:
        cos_sim = 1.0
    
    # Adaptive alpha based on similarity
    # High similarity → use original alpha
    # Low similarity → reduce blending to preserve features
    if cos_sim >= threshold:
        effective_alpha = alpha
    else:
        # Scale alpha down proportionally to similarity
        similarity_factor = max(0.0, cos_sim) / max(threshold, 1e-8)
        effective_alpha = alpha * similarity_factor
    
    return torch.lerp(a, b, effective_alpha).to(tensors[0].dtype)


@register_algorithm(
    name="train_difference",
    display_name="Train Difference",
    description="Extract training delta with optional clipping: result = A + clip(B - C) × α",
    params=[PARAM_ALPHA, PARAM_CLIP_DELTA, PARAM_CLIP_VALUE],
    num_models=3,
)
def train_difference(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 1.0)
    clip_delta = params.get("clip_delta", False)
    clip_value = params.get("clip_value", 1.0)
    
    a, b, c = tensors[0].float(), tensors[1].float(), tensors[2].float()
    delta = b - c
    
    if clip_delta:
        delta = torch.clamp(delta, -clip_value, clip_value)
    
    return (a + delta * alpha).to(tensors[0].dtype)


@register_algorithm(
    name="distribution",
    display_name="Distribution Merge",
    description="Align statistical distributions (mean and variance) between models",
    params=[PARAM_ALPHA, PARAM_NORM_MEAN, PARAM_NORM_STD],
    num_models=2,
)
def distribution_merge(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    normalize_mean = params.get("normalize_mean", True)
    normalize_std = params.get("normalize_std", True)
    
    a, b = tensors[0].float(), tensors[1].float()
    
    if a.numel() == 0 or a.dim() == 0:
        return torch.lerp(a, b, alpha).to(tensors[0].dtype)
    
    # Compute statistics
    mean_a, std_a = a.mean(), a.std()
    mean_b, std_b = b.mean(), b.std()
    
    # Normalize b to match a's distribution, then interpolate
    b_normalized = b.clone()
    
    if normalize_std and std_b > 1e-8:
        target_std = std_a * (1.0 - alpha) + std_b * alpha
        b_normalized = (b_normalized - mean_b) / std_b * target_std
        if normalize_mean:
            target_mean = mean_a * (1.0 - alpha) + mean_b * alpha
            b_normalized = b_normalized + target_mean
        else:
            b_normalized = b_normalized + mean_b
    elif normalize_mean:
        target_mean = mean_a * (1.0 - alpha) + mean_b * alpha
        b_normalized = b_normalized - mean_b + target_mean
    
    result = a * (1.0 - alpha) + b_normalized * alpha
    return result.to(tensors[0].dtype)


@register_algorithm(
    name="smoothed_add_difference",
    display_name="Smoothed Add Difference",
    description="Add Difference with Gaussian smoothing on deltas",
    params=[PARAM_ALPHA, PARAM_SMOOTH_SIGMA],
    num_models=3,
)
def smoothed_add_difference(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 1.0)
    sigma = params.get("smooth_sigma", 1.0)
    
    a, b, c = tensors[0].float(), tensors[1].float(), tensors[2].float()
    delta = b - c
    
    # Apply 1D Gaussian smoothing to the flattened delta
    original_shape = delta.shape
    flat = delta.flatten()
    
    if flat.numel() > 1 and sigma > 0:
        # Create Gaussian kernel
        kernel_size = max(3, int(sigma * 4) | 1)  # Ensure odd
        half = kernel_size // 2
        x = torch.arange(-half, half + 1, dtype=torch.float32, device=flat.device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        
        # Apply as 1D convolution
        flat_padded = torch.nn.functional.pad(flat.unsqueeze(0).unsqueeze(0), (half, half), mode='reflect')
        smoothed = torch.nn.functional.conv1d(flat_padded, kernel.unsqueeze(0).unsqueeze(0))
        delta = smoothed.squeeze().reshape(original_shape)
    
    return (a + delta * alpha).to(tensors[0].dtype)


@register_algorithm(
    name="multiply_difference",
    display_name="Multiply Difference",
    description="Scale model weights multiplicatively: result = A × (1 + (B/A - 1) × α)",
    params=[PARAM_ALPHA],
    num_models=2,
)
def multiply_difference(tensors: list[torch.Tensor], params: dict) -> torch.Tensor:
    alpha = params.get("alpha", 0.5)
    a, b = tensors[0].float(), tensors[1].float()
    
    # Avoid division by zero
    safe_a = torch.where(a.abs() > 1e-8, a, torch.ones_like(a) * 1e-8)
    ratio = b / safe_a
    result = a * (1.0 + (ratio - 1.0) * alpha)
    
    return result.to(tensors[0].dtype)
