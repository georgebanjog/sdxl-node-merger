"""
Merge Executor — executes a compiled graph plan.

Supports two modes:
1. Normal mode: loads models fully into memory
2. Low-VRAM mode: processes tensors one at a time (streaming)
"""

import os
import time
import torch
import gc
from typing import Optional, Callable

from . import tensor_io
from . import algorithms
from . import lora_utils
from . import vae_utils
from . import metadata as meta_utils


class MergeProgress:
    """Tracks and reports merge execution progress."""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.total_steps = 0
        self.current_step = 0
        self.current_operation = ""
        self.sub_progress = 0.0  # 0-1 within current step
        self.logs = []
        self.errors = []
        self.start_time = time.time()
    
    def set_total(self, total: int):
        self.total_steps = total
    
    def begin_step(self, step_num: int, operation: str):
        self.current_step = step_num
        self.current_operation = operation
        self.sub_progress = 0.0
        self._report()
    
    def update_sub(self, progress: float, detail: str = ""):
        self.sub_progress = progress
        if detail:
            self.current_operation = detail
        self._report()
    
    def log(self, message: str):
        self.logs.append(message)
        if self.callback:
            self.callback({
                "type": "log",
                "message": message,
            })
    
    def error(self, message: str):
        self.errors.append(message)
        if self.callback:
            self.callback({
                "type": "log",
                "message": f"ERROR: {message}",
            })
    
    def _report(self):
        if self.callback:
            elapsed = time.time() - self.start_time
            self.callback({
                "type": "progress",
                "total_steps": self.total_steps,
                "current_step": self.current_step,
                "operation": self.current_operation,
                "sub_progress": self.sub_progress,
                "overall_progress": (self.current_step + self.sub_progress) / max(self.total_steps, 1),
                "elapsed_seconds": elapsed,
            })


def execute_plan(
    steps: list,
    config: dict,
    low_vram: bool = False,
    device: str = "auto",
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    Execute a compiled merge plan.
    
    Args:
        steps: List of ExecutionStep objects from graph_compiler.compile_graph()
        config: Application config (directories, etc.)
        low_vram: If True, use streaming mode
        device: 'auto', 'cpu', or 'cuda'
        progress_callback: Function called with progress updates
    
    Returns:
        dict with execution results and statistics
    """
    progress = MergeProgress(progress_callback)
    progress.set_total(len(steps))
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    progress.log(f"Execution device: {device}")
    progress.log(f"Low VRAM mode: {low_vram}")
    progress.log(f"Total steps: {len(steps)}")
    
    # Runtime state: stores intermediate results (node_id -> data)
    state = {}
    
    # Track loaded LoRA data (for reuse)
    lora_cache = {}
    
    dirs = config.get("directories", {})
    checkpoint_dir = dirs.get("checkpoints", "")
    lora_dir = dirs.get("lora", "")
    vae_dir = dirs.get("vae", "")
    output_dir = dirs.get("output", "")
    
    results = {
        "success": True,
        "steps_completed": 0,
        "output_files": [],
        "errors": [],
    }
    
    try:
        for i, step in enumerate(steps):
            step_dict = step.to_dict() if hasattr(step, 'to_dict') else step
            step_type = step_dict["step_type"]
            node_id = step_dict["node_id"]
            params = step_dict["params"]
            
            progress.begin_step(i, f"Step {i+1}/{len(steps)}: {step_type}")
            
            if step_type == "load_checkpoint":
                _execute_load_checkpoint(
                    state, node_id, params, checkpoint_dir, device, low_vram, progress
                )
            
            elif step_type == "load_lora":
                _execute_load_lora(
                    state, lora_cache, node_id, params, lora_dir, progress
                )
            
            elif step_type == "load_vae":
                _execute_load_vae(
                    state, node_id, params, vae_dir, device, low_vram, progress
                )
            
            elif step_type == "merge":
                if low_vram:
                    _execute_merge_streaming(
                        state, node_id, params, device, progress
                    )
                else:
                    _execute_merge_full(
                        state, node_id, params, device, progress
                    )
            
            elif step_type == "apply_lora":
                _execute_apply_lora(
                    state, lora_cache, node_id, params, device, low_vram, progress
                )
            
            elif step_type == "replace_vae":
                _execute_replace_vae(
                    state, node_id, params, device, progress
                )
            
            elif step_type == "edit_metadata":
                _execute_edit_metadata(
                    state, node_id, params, progress
                )
            
            elif step_type == "save":
                output_path = _execute_save(
                    state, node_id, params, output_dir, progress
                )
                results["output_files"].append(output_path)
            
            results["steps_completed"] = i + 1
            progress.log(f"✓ Step {i+1} completed: {step_type}")
            
            # Clean up GPU memory
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
    
    except Exception as e:
        results["success"] = False
        results["errors"].append(str(e))
        progress.error(f"Execution failed: {str(e)}")
        raise
    
    finally:
        # Clean up all state
        state.clear()
        lora_cache.clear()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    progress.log("Execution completed successfully!")
    return results


# ─── Step Executors ──────────────────────────────────────────────────────────────

def _resolve_path(filename: str, directory: str) -> str:
    """Resolve a file path, checking both relative and absolute."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    
    full_path = os.path.join(directory, filename)
    if os.path.exists(full_path):
        return full_path
    
    raise FileNotFoundError(f"File not found: {filename} (looked in {directory})")


def _execute_load_checkpoint(state, node_id, params, checkpoint_dir, device, low_vram, progress):
    """Load a checkpoint file."""
    filepath = _resolve_path(params["file"], checkpoint_dir)
    progress.log(f"Loading checkpoint: {os.path.basename(filepath)}")
    
    if low_vram:
        # In low-VRAM mode, store just the path and keys
        keys = tensor_io.get_tensor_keys(filepath)
        state[node_id] = {
            "type": "checkpoint_ref",
            "path": filepath,
            "keys": keys,
            "metadata": meta_utils.read_safetensors_metadata(filepath),
        }
        progress.log(f"  Loaded reference ({len(keys)} tensors)")
    else:
        state_dict = tensor_io.load_model_full(filepath, device=device)
        md = meta_utils.read_safetensors_metadata(filepath)
        state[node_id] = {
            "type": "checkpoint",
            "state_dict": state_dict,
            "metadata": md,
        }
        progress.log(f"  Loaded {len(state_dict)} tensors to {device}")


def _execute_load_lora(state, lora_cache, node_id, params, lora_dir, progress):
    """Load a LoRA file."""
    filepath = _resolve_path(params["file"], lora_dir)
    progress.log(f"Loading LoRA: {os.path.basename(filepath)}")
    
    lora_data = lora_utils.parse_lora_file(filepath)
    lora_cache[node_id] = lora_data
    
    state[node_id] = {
        "type": "lora",
        "path": filepath,
        "strength": params.get("strength", 1.0),
        "data": lora_data,
    }
    progress.log(f"  Loaded {len(lora_data)} LoRA layers")


def _execute_load_vae(state, node_id, params, vae_dir, device, low_vram, progress):
    """Load a VAE file."""
    filepath = _resolve_path(params["file"], vae_dir)
    progress.log(f"Loading VAE: {os.path.basename(filepath)}")
    
    if low_vram:
        keys = tensor_io.get_tensor_keys(filepath)
        state[node_id] = {
            "type": "vae_ref",
            "path": filepath,
            "keys": keys,
        }
    else:
        state_dict = tensor_io.load_model_full(filepath, device=device)
        state[node_id] = {
            "type": "vae",
            "state_dict": state_dict,
        }
    progress.log(f"  VAE loaded")


def _execute_merge_full(state, node_id, params, device, progress):
    """Execute merge in normal (full memory) mode."""
    algorithm_name = params["algorithm"]
    algo_params = params["params"]
    model_inputs = params["model_inputs"]
    use_mbw = params.get("use_mbw", False)
    mbw_weights = params.get("mbw_weights", {})
    
    progress.log(f"Merging with algorithm: {algorithm_name}")
    
    # Gather input state dicts
    input_dicts = []
    input_names = sorted(model_inputs.keys())  # MODEL_A, MODEL_B, MODEL_C
    
    for input_name in input_names:
        src_id = model_inputs[input_name]
        src_state = state.get(src_id)
        if not src_state:
            raise ValueError(f"Source node {src_id} not found in state")
        
        if src_state["type"] == "checkpoint":
            input_dicts.append(src_state["state_dict"])
        elif src_state["type"] == "merged":
            input_dicts.append(src_state["state_dict"])
        else:
            raise ValueError(f"Cannot merge from node type: {src_state['type']}")
    
    if len(input_dicts) < 2:
        raise ValueError(f"Merge requires at least 2 models, got {len(input_dicts)}")
    
    # Get all unique keys
    all_keys = set()
    for d in input_dicts:
        all_keys.update(d.keys())
    all_keys = sorted(all_keys)
    
    progress.log(f"  Merging {len(all_keys)} tensors...")
    
    result = {}
    for ki, key in enumerate(all_keys):
        if ki % 100 == 0:
            progress.update_sub(ki / len(all_keys), f"Merging tensor {ki}/{len(all_keys)}")
        
        # Collect tensors from all inputs
        tensors = []
        for d in input_dicts:
            if key in d:
                tensors.append(d[key].to(device))
            else:
                # Key missing in this model — use zeros
                ref_tensor = next(dd[key] for dd in input_dicts if key in dd)
                tensors.append(torch.zeros_like(ref_tensor).to(device))
        
        # Determine effective parameters (MBW)
        effective_params = dict(algo_params)
        if use_mbw:
            block_id = tensor_io.get_block_id_for_key(key)
            if block_id in mbw_weights:
                effective_params["alpha"] = mbw_weights[block_id]
        
        # Apply merge algorithm
        try:
            result[key] = algorithms.merge_tensors(algorithm_name, tensors, effective_params)
        except Exception as e:
            # Fallback to simple interpolation on error
            progress.log(f"  ⚠ Algorithm error on {key}: {e}, using weighted_sum fallback")
            alpha = effective_params.get("alpha", 0.5)
            result[key] = torch.lerp(tensors[0].float(), tensors[1].float(), alpha).to(tensors[0].dtype)
    
    # Merge metadata from sources
    source_names = []
    for input_name in input_names:
        src_id = model_inputs[input_name]
        src_state = state.get(src_id, {})
        md = src_state.get("metadata", {})
        source_names.append(md.get("ss_model_name", os.path.basename(src_state.get("path", src_id))))
    
    merged_metadata = meta_utils.create_merge_metadata(
        algorithm=algorithm_name,
        source_models=source_names,
        params=algo_params,
        output_dtype="mixed",
    )
    
    state[node_id] = {
        "type": "merged",
        "state_dict": result,
        "metadata": merged_metadata,
    }
    progress.log(f"  Merge complete: {len(result)} tensors")


def _execute_merge_streaming(state, node_id, params, device, progress):
    """Execute merge in streaming (low-VRAM) mode."""
    algorithm_name = params["algorithm"]
    algo_params = params["params"]
    model_inputs = params["model_inputs"]
    use_mbw = params.get("use_mbw", False)
    mbw_weights = params.get("mbw_weights", {})
    
    progress.log(f"Streaming merge with algorithm: {algorithm_name}")
    
    # Gather input paths/refs
    input_refs = []
    input_names = sorted(model_inputs.keys())
    
    for input_name in input_names:
        src_id = model_inputs[input_name]
        src_state = state.get(src_id)
        if not src_state:
            raise ValueError(f"Source node {src_id} not found in state")
        input_refs.append(src_state)
    
    # Determine all keys
    all_keys = set()
    for ref in input_refs:
        if "keys" in ref:
            all_keys.update(ref["keys"])
        elif "state_dict" in ref:
            all_keys.update(ref["state_dict"].keys())
    all_keys = sorted(all_keys)
    
    progress.log(f"  Streaming merge of {len(all_keys)} tensors...")
    
    result = {}
    for ki, key in enumerate(all_keys):
        if ki % 50 == 0:
            progress.update_sub(ki / len(all_keys), f"Streaming tensor {ki}/{len(all_keys)}")
        
        tensors = []
        skip_key = False
        for ref in input_refs:
            if ref["type"] == "checkpoint_ref":
                if key in ref.get("keys", []):
                    t = tensor_io.load_tensor(ref["path"], key, device="cpu")
                    tensors.append(t.to(device))
                else:
                    # Key missing in this model — will fill with zeros below
                    tensors.append(None)
            elif ref["type"] in ("checkpoint", "merged"):
                if key in ref["state_dict"]:
                    tensors.append(ref["state_dict"][key].to(device))
                else:
                    tensors.append(None)
            else:
                raise ValueError(f"Cannot stream from type: {ref['type']}")
        
        # Fill None entries with zeros matching the first real tensor
        real_tensors = [t for t in tensors if t is not None]
        if not real_tensors:
            continue  # No model has this key at all, skip
        
        ref_tensor = real_tensors[0]
        tensors = [t if t is not None else torch.zeros_like(ref_tensor) for t in tensors]
        
        effective_params = dict(algo_params)
        if use_mbw:
            block_id = tensor_io.get_block_id_for_key(key)
            if block_id in mbw_weights:
                effective_params["alpha"] = mbw_weights[block_id]
        
        merged_tensor = algorithms.merge_tensors(algorithm_name, tensors, effective_params)
        result[key] = merged_tensor.cpu()
        
        # Free GPU memory immediately
        del tensors, merged_tensor
        if device == "cuda" and ki % 100 == 0:
            torch.cuda.empty_cache()
    
    state[node_id] = {
        "type": "merged",
        "state_dict": result,
        "metadata": {},
    }
    progress.log(f"  Streaming merge complete")


def _execute_apply_lora(state, lora_cache, node_id, params, device, low_vram, progress):
    """Apply LoRA to a model."""
    model_src = params["model_source"]
    lora_src = params["lora_source"]
    strength = params.get("strength", 1.0)
    
    src_state = state.get(model_src)
    lora_state = state.get(lora_src)
    
    if not src_state or not lora_state:
        raise ValueError("Missing model or LoRA source for apply_lora")
    
    progress.log(f"Applying LoRA (strength={strength})")
    
    if src_state["type"] in ("checkpoint", "merged"):
        # Full mode: apply directly to state dict
        lora_data = lora_state["data"]
        model_dict = src_state["state_dict"]
        
        applied = 0
        total = len(lora_data)
        for ki, (model_key, layer_data) in enumerate(lora_data.items()):
            if ki % 20 == 0:
                progress.update_sub(ki / max(total, 1), f"Applying LoRA layer {ki}/{total}")
            
            target_key = model_key
            if target_key not in model_dict:
                target_key = model_key + ".weight"
            if target_key not in model_dict:
                continue
            
            delta = lora_utils.compute_lora_delta(
                layer_data["up"].to(device),
                layer_data["down"].to(device),
                layer_data["alpha"],
                layer_data["rank"],
            )
            
            target = model_dict[target_key].to(device).float()
            if delta.shape == target.shape:
                model_dict[target_key] = (target + delta * strength).to(model_dict[target_key].dtype)
                applied += 1
            else:
                try:
                    delta = delta.reshape(target.shape)
                    model_dict[target_key] = (target + delta * strength).to(model_dict[target_key].dtype)
                    applied += 1
                except RuntimeError:
                    pass
        
        state[node_id] = {
            "type": "merged",
            "state_dict": model_dict,
            "metadata": src_state.get("metadata", {}),
        }
        progress.log(f"  Applied {applied}/{total} LoRA layers")
    
    elif src_state["type"] == "checkpoint_ref":
        # Low-VRAM: load model fully but apply LoRA on the fly
        progress.log("  Loading model for LoRA application...")
        model_dict = tensor_io.load_model_full(src_state["path"], device=device)
        lora_data = lora_state["data"]
        
        applied = 0
        for model_key, layer_data in lora_data.items():
            target_key = model_key
            if target_key not in model_dict:
                target_key = model_key + ".weight"
            if target_key not in model_dict:
                continue
            
            delta = lora_utils.compute_lora_delta(
                layer_data["up"].to(device),
                layer_data["down"].to(device),
                layer_data["alpha"],
                layer_data["rank"],
            )
            
            target = model_dict[target_key].float()
            if delta.shape == target.shape:
                model_dict[target_key] = (target + delta * strength).to(model_dict[target_key].dtype)
                applied += 1
        
        state[node_id] = {
            "type": "merged",
            "state_dict": model_dict,
            "metadata": src_state.get("metadata", {}),
        }
        progress.log(f"  Applied {applied} LoRA layers")


def _execute_replace_vae(state, node_id, params, device, progress):
    """Replace VAE in a model."""
    model_src = params["model_source"]
    vae_src = params["vae_source"]
    
    src_state = state.get(model_src)
    vae_state = state.get(vae_src)
    
    if not src_state or not vae_state:
        raise ValueError("Missing model or VAE source for replace_vae")
    
    progress.log("Replacing VAE...")
    
    if src_state["type"] in ("checkpoint", "merged"):
        model_dict = src_state["state_dict"]
        
        if vae_state["type"] in ("vae", "checkpoint", "merged"):
            vae_dict = vae_state["state_dict"]
        elif vae_state["type"] == "vae_ref":
            vae_dict = tensor_io.load_model_full(vae_state["path"], device=device)
        else:
            raise ValueError(f"Invalid VAE source type: {vae_state['type']}")
        
        result = vae_utils.replace_vae(model_dict, vae_dict)
        
        state[node_id] = {
            "type": "merged",
            "state_dict": result,
            "metadata": src_state.get("metadata", {}),
        }
        progress.log("  VAE replaced successfully")
    else:
        raise ValueError("Cannot replace VAE in streaming mode without loading model first")


def _execute_edit_metadata(state, node_id, params, progress):
    """Edit metadata on a model."""
    model_src = params["model_source"]
    new_metadata = params.get("metadata", {})
    
    src_state = state.get(model_src)
    if not src_state:
        raise ValueError("Missing model source for metadata_editor")
    
    # Pass through the model data with updated metadata
    existing_md = dict(src_state.get("metadata", {}))
    existing_md.update(new_metadata)
    
    state[node_id] = dict(src_state)
    state[node_id]["metadata"] = existing_md
    progress.log(f"  Metadata updated ({len(new_metadata)} fields)")


def _execute_save(state, node_id, params, output_dir, progress):
    """Save the final model to disk."""
    model_src = params["model_source"]
    filename = params.get("filename", "merged_model.safetensors")
    dtype = params.get("dtype", "fp16")
    custom_metadata = params.get("metadata", {})
    
    src_state = state.get(model_src)
    if not src_state:
        raise ValueError("Missing model source for save")
    
    if not filename.endswith(".safetensors"):
        filename += ".safetensors"
    
    output_path = os.path.join(output_dir, filename) if output_dir else filename
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    progress.log(f"Saving to: {output_path} (dtype={dtype})")
    
    # Prepare metadata
    model_metadata = dict(src_state.get("metadata", {}))
    model_metadata.update(custom_metadata)
    model_metadata["output_dtype"] = dtype
    # Ensure all values are strings
    model_metadata = {k: str(v) for k, v in model_metadata.items()}
    
    if src_state["type"] in ("checkpoint", "merged"):
        state_dict = src_state["state_dict"]
        progress.log(f"  Saving {len(state_dict)} tensors...")
        tensor_io.save_model(state_dict, output_path, dtype=dtype, metadata=model_metadata)
    elif src_state["type"] == "checkpoint_ref":
        # Need to load and save
        progress.log("  Loading model for save...")
        state_dict = tensor_io.load_model_full(src_state["path"])
        tensor_io.save_model(state_dict, output_path, dtype=dtype, metadata=model_metadata)
        del state_dict
    
    file_size = os.path.getsize(output_path)
    size_gb = file_size / (1024**3)
    progress.log(f"  ✓ Saved: {size_gb:.2f} GB")
    
    return output_path
