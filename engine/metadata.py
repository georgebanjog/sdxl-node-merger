"""
Metadata management for safetensors model files.
Read, write, and merge metadata dictionaries.
"""

import json
import struct
import os
from datetime import datetime, timezone
from typing import Optional


def read_safetensors_metadata(filepath: str) -> dict[str, str]:
    """Read the __metadata__ field from a safetensors file header."""
    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            return header.get("__metadata__", {})
    except Exception:
        return {}


def create_merge_metadata(
    algorithm: str,
    source_models: list[str],
    params: dict,
    output_dtype: str,
    custom_metadata: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Create metadata for a merged model."""
    metadata = {
        "merger": "SDXL Node Merger v1.0",
        "merger_author": "George Kogan",
        "merge_date": datetime.now(timezone.utc).isoformat(),
        "merge_algorithm": algorithm,
        "merge_params": json.dumps(params),
        "source_models": json.dumps(source_models),
        "output_dtype": output_dtype,
    }
    
    if custom_metadata:
        metadata.update(custom_metadata)
    
    # Ensure all values are strings (safetensors requirement)
    return {k: str(v) for k, v in metadata.items()}


def merge_metadata(
    *metadata_dicts: dict[str, str],
    prefix_sources: bool = True,
) -> dict[str, str]:
    """Merge multiple metadata dictionaries."""
    result = {}
    
    for i, md in enumerate(metadata_dicts):
        for key, value in md.items():
            if prefix_sources and key in result:
                result[f"source_{i}_{key}"] = value
            else:
                result[key] = value
    
    return result


def format_metadata_for_display(metadata: dict[str, str]) -> list[dict]:
    """Format metadata for UI display."""
    items = []
    for key, value in sorted(metadata.items()):
        # Try to parse JSON values for better display
        display_value = value
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (list, dict)):
                display_value = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, TypeError):
            pass
        
        items.append({
            "key": key,
            "value": display_value,
            "raw_value": value,
            "editable": True,
        })
    
    return items
