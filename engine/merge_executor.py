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
import json
import struct

class FileRegistry:
    def __init__(self):
        self._handles = {}
        self._headers = {}
        
    def get_handle(self, filepath, device="cpu"):
        if filepath not in self._handles:
            self._handles[filepath] = tensor_io.safe_open(filepath, framework="pt", device=device)
        return self._handles[filepath]
        
    def get_tensor(self, filepath, key, device="cpu"):
        h = self.get_handle(filepath, device=device)
        return h.get_tensor(key)
        
    def get_shape_and_dtype(self, filepath, key):
        if filepath not in self._headers:
            with open(filepath, "rb") as f:
                header_size = struct.unpack("<Q", f.read(8))[0]
                header_json = f.read(header_size).decode("utf-8")
                self._headers[filepath] = json.loads(header_json)
        info = self._headers[filepath].get(key)
        if info:
            return info["shape"], info["dtype"]
        return None, None
        
    def close_all(self):
        self._handles.clear()
        self._headers.clear()

def _get_lazy_shape(node_id, key, state, registry):
    node = state.get(node_id)
    if not node:
        return [], "F32"
    ntype = node["type"]
    if ntype == "checkpoint_ref":
        if key in node.get("keys", []):
            return registry.get_shape_and_dtype(node["path"], key)
        return None, None
    elif ntype in ("checkpoint", "merged"):
        t = node["state_dict"].get(key)
        if t is not None:
            dtype_str = "FP32"
            if t.dtype == torch.float16: dtype_str = "F16"
            elif t.dtype == torch.bfloat16: dtype_str = "BF16"
            return list(t.shape), dtype_str
        return None, None
    elif ntype == "lazy_merged":
        for input_name in sorted(node["model_inputs"].keys()):
            src_id = node["model_inputs"][input_name]
            shape, dt = _get_lazy_shape(src_id, key, state, registry)
            if shape is not None: return shape, dt
    elif ntype == "lazy_apply_lora":
        return _get_lazy_shape(node["model_source"], key, state, registry)
    elif ntype == "lazy_replace_vae" or ntype == "vae_ref":
        if key.startswith("first_stage_model.") and "vae_source" in node:
            shape, dt = _get_lazy_shape(node["vae_source"], key, state, registry)
            if shape is not None: return shape, dt
        elif ntype == "vae_ref":
            if key in node.get("keys", []):
                return registry.get_shape_and_dtype(node["path"], key)
        if "model_source" in node:
            return _get_lazy_shape(node["model_source"], key, state, registry)
    return None, None

def _evaluate_lazy_tensor(node_id, key, state, registry, device):
    node = state.get(node_id)
    if not node:
        return None
        
    ntype = node["type"]
    if ntype in ("checkpoint", "merged"):
        return node["state_dict"].get(key)
    elif ntype in ("checkpoint_ref", "vae_ref"):
        if key in node["keys"]:
            return registry.get_tensor(node["path"], key, device=device)
        return None
    elif ntype == "lazy_merged":
        all_keys = node["keys"]
        if key not in all_keys:
            return None
            
        tensors = []
        model_inputs = node["model_inputs"]
        for input_name in sorted(model_inputs.keys()):
            src_id = model_inputs[input_name]
            t = _evaluate_lazy_tensor(src_id, key, state, registry, device)
            tensors.append(t)
            
        real_tensors = [t for t in tensors if t is not None]
        if not real_tensors:
            return None
        ref_tensor = real_tensors[0]
        tensors = [t.to(device) if t is not None else torch.zeros_like(ref_tensor).to(device) for t in tensors]
        
        effective_params = dict(node["params"])
        if node.get("use_mbw"):
            block_id = tensor_io.get_block_id_for_key(key)
            mbw = node.get("mbw_weights", {})
            if block_id in mbw:
                effective_params["alpha"] = mbw[block_id]
                
        return algorithms.merge_tensors(node["algorithm"], tensors, effective_params)
        
    elif ntype == "lazy_apply_lora":
        t = _evaluate_lazy_tensor(node["model_source"], key, state, registry, device)
        if t is None:
            return None
        lora_data = node["lora_data"]
        layer_data = lora_data.get(key)
        if not layer_data and key.endswith(".weight"):
            layer_data = lora_data.get(key[:-7])
        if not layer_data:
            layer_data = lora_data.get(key + ".weight")
            
        t = t.to(device)
        if layer_data:
            delta = lora_utils.compute_lora_delta(
                layer_data["up"].to(device),
                layer_data["down"].to(device),
                layer_data["alpha"],
                layer_data["rank"]
            )
            target = t.float()
            if delta.shape == target.shape:
                t = (target + delta * node["strength"]).to(t.dtype)
            else:
                try:
                    delta = delta.reshape(target.shape)
                    t = (target + delta * node["strength"]).to(t.dtype)
                except RuntimeError:
                    pass
        return t
        
    elif ntype == "lazy_replace_vae":
        if key.startswith("first_stage_model."):
            t = _evaluate_lazy_tensor(node["vae_source"], key, state, registry, device)
            if t is not None:
                return t
        return _evaluate_lazy_tensor(node["model_source"], key, state, registry, device)
    
    return None


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
    
    registry = FileRegistry()
    state = {}
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
                if low_vram:
                    src_state = state.get(params["model_source"], {})
                    state[node_id] = {
                        "type": "lazy_apply_lora",
                        "model_source": params["model_source"],
                        "lora_data": lora_cache.get(params["lora_source"], {}),
                        "strength": params.get("strength", 1.0),
                        "keys": list(src_state.get("keys", src_state.get("state_dict", {}).keys())),
                        "metadata": src_state.get("metadata", {})
                    }
                    progress.log("  Prepared lazy LoRA apply")
                else:
                    _execute_apply_lora(state, lora_cache, node_id, params, device, low_vram, progress)
            
            elif step_type == "replace_vae":
                if low_vram:
                    m_state = state.get(params["model_source"], {})
                    v_state = state.get(params["vae_source"], {})
                    m_keys = set(m_state.get("keys", m_state.get("state_dict", {}).keys()))
                    v_keys = set(v_state.get("keys", v_state.get("state_dict", {}).keys()))
                    m_keys = {k for k in m_keys if not k.startswith("first_stage_model.")}
                    all_keys = m_keys | v_keys
                    state[node_id] = {
                        "type": "lazy_replace_vae", "model_source": params["model_source"],
                        "vae_source": params["vae_source"], "keys": sorted(all_keys),
                        "metadata": m_state.get("metadata", {})
                    }
                    progress.log("  Prepared lazy VAE replace")
                else:
                    _execute_replace_vae(state, node_id, params, device, progress)
            
            elif step_type == "edit_metadata":
                _execute_edit_metadata(
                    state, node_id, params, progress
                )
            
            elif step_type == "save":
                output_path = _execute_save(
                    state, node_id, params, output_dir, progress, registry, device, low_vram
                )
                results["output_files"].append(output_path)
            
            results["steps_completed"] = i + 1
            progress.log(f"✓ Step {i+1} completed: {step_type}")
            
            # ── Intelligent memory cleanup ──────────────────────────────
            # Free state entries that are no longer needed by remaining steps.
            # In low_vram mode, we skip this since state entries are just tiny graph node recipes
            # and lazy evaluation at the end needs the whole graph to be present.
            if not low_vram:
                remaining_steps = steps[i + 1:]
                referenced_ids = set()
                for future_step in remaining_steps:
                    fs = future_step.to_dict() if hasattr(future_step, 'to_dict') else future_step
                    fp = fs.get("params", {})
                    referenced_ids.add(fs.get("node_id", ""))
                    referenced_ids.add(fp.get("model_source", ""))
                    referenced_ids.add(fp.get("lora_source", ""))
                    referenced_ids.add(fp.get("vae_source", ""))
                    for v in fp.get("model_inputs", {}).values():
                        referenced_ids.add(v)
                referenced_ids.discard("")
                
                stale_ids = [sid for sid in state if sid not in referenced_ids]
                for sid in stale_ids:
                    entry = state.pop(sid)
                    if "state_dict" in entry:
                        del entry["state_dict"]
                    entry.clear()
                    progress.log(f"  🗑 Freed memory: node {sid}")
            
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
        registry.close_all()
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
        # Normal Mode: Load entirely into CPU RAM (not VRAM) to prevent OOM
        state_dict = tensor_io.load_model_full(filepath, device="cpu")
        md = meta_utils.read_safetensors_metadata(filepath)
        state[node_id] = {
            "type": "checkpoint",
            "state_dict": state_dict,
            "metadata": md,
        }
        progress.log(f"  Loaded {len(state_dict)} tensors to RAM (cpu)")


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
        # Normal Mode: Load VAE into CPU RAM (not VRAM)
        state_dict = tensor_io.load_model_full(filepath, device="cpu")
        state[node_id] = {
            "type": "vae",
            "state_dict": state_dict,
        }
    progress.log(f"  VAE loaded into RAM")


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
            merged_tensor = algorithms.merge_tensors(algorithm_name, tensors, effective_params)
            result[key] = merged_tensor.cpu()
        except Exception as e:
            # Fallback to simple interpolation on error
            progress.log(f"  ⚠ Algorithm error on {key}: {e}, using weighted_sum fallback")
            alpha = effective_params.get("alpha", 0.5)
            merged_tensor = torch.lerp(tensors[0].float(), tensors[1].float(), alpha).to(tensors[0].dtype)
            result[key] = merged_tensor.cpu()
    
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
    """Setup a lazy merge recipe instead of executing immediately."""
    algorithm_name = params["algorithm"]
    algo_params = params["params"]
    model_inputs = params["model_inputs"]
    
    progress.log(f"Preparing lazy merge with algorithm: {algorithm_name}")
    
    all_keys = set()
    for input_name in sorted(model_inputs.keys()):
        src_id = model_inputs[input_name]
        ref = state.get(src_id, {})
        if "keys" in ref:
            all_keys.update(ref["keys"])
        elif "state_dict" in ref:
            all_keys.update(ref["state_dict"].keys())
            
    state[node_id] = {
        "type": "lazy_merged",
        "algorithm": algorithm_name,
        "params": algo_params,
        "model_inputs": model_inputs,
        "use_mbw": params.get("use_mbw", False),
        "mbw_weights": params.get("mbw_weights", {}),
        "keys": sorted(all_keys),
        "metadata": {},
    }
    progress.log(f"  Lazy merge prepared ({len(all_keys)} keys registered)")


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
        # Full mode: apply to a new state dict to avoid mutating source node (prevents issues on branched graphs)
        lora_data = lora_state["data"]
        model_dict = src_state["state_dict"].copy()
        
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
                model_dict[target_key] = (target + delta * strength).to(model_dict[target_key].dtype).cpu()
                applied += 1
            else:
                try:
                    delta = delta.reshape(target.shape)
                    model_dict[target_key] = (target + delta * strength).to(model_dict[target_key].dtype).cpu()
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
            vae_dict = tensor_io.load_model_full(vae_state["path"], device="cpu")
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
    if "state_dict" in src_state:
        state[node_id]["state_dict"] = src_state["state_dict"].copy()
    progress.log(f"  Metadata updated ({len(new_metadata)} fields)")


def _execute_save(state, node_id, params, output_dir, progress, registry, device, low_vram):
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
    
    model_metadata = dict(src_state.get("metadata", {}))
    model_metadata.update(custom_metadata)
    model_metadata["output_dtype"] = dtype
    model_metadata = {k: str(v) for k, v in model_metadata.items()}
    
    if low_vram and src_state["type"] in ("lazy_merged", "lazy_apply_lora", "lazy_replace_vae", "checkpoint_ref"):
        progress.log(f"  Streaming save initialized...")
        all_keys = src_state.get("keys", [])
        
        def shape_func(key):
            return _get_lazy_shape(model_src, key, state, registry)
            
        def tensor_generator(key):
            t = _evaluate_lazy_tensor(model_src, key, state, registry, device)
            if t is None:
                raise RuntimeError(f"Missing tensor evaluated logic error: {key}")
            return t
            
        tensor_io.save_model_lazy_streaming(
            filepath=output_path,
            keys=all_keys,
            shape_func=shape_func,
            tensor_generator=tensor_generator,
            dtype=dtype,
            metadata=model_metadata,
            progress_callback=lambda ki, tot, key: progress.update_sub(ki / tot, f"Streaming {ki}/{tot}") if ki % 50 == 0 else None
        )
    elif src_state["type"] in ("checkpoint", "merged"):
        state_dict = src_state["state_dict"]
        progress.log(f"  Saving {len(state_dict)} tensors from RAM...")
        tensor_io.save_model(state_dict, output_path, dtype=dtype, metadata=model_metadata)
    
    file_size = os.path.getsize(output_path)
    size_gb = file_size / (1024**3)
    progress.log(f"  ✓ Saved: {size_gb:.2f} GB")
    
    return output_path
