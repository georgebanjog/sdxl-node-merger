"""
Graph Compiler — converts a visual node graph into an ordered execution plan.

The compiler performs:
1. Graph validation (type checking, cycle detection)
2. Topological sorting
3. Generation of execution steps
"""

from typing import Optional


# ─── Node Type Definitions ───────────────────────────────────────────────────────

NODE_TYPES = {
    "checkpoint_loader": {
        "category": "source",
        "inputs": [],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": ["file"],
    },
    "lora_loader": {
        "category": "source",
        "inputs": [],
        "outputs": [{"name": "LORA", "type": "LORA"}],
        "data_fields": ["file", "strength"],
    },
    "vae_loader": {
        "category": "source",
        "inputs": [],
        "outputs": [{"name": "VAE", "type": "VAE"}],
        "data_fields": ["file"],
    },
    "merge_models": {
        "category": "processing",
        "inputs": [
            {"name": "MODEL_A", "type": "MODEL"},
            {"name": "MODEL_B", "type": "MODEL"},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": ["algorithm", "params", "use_mbw", "mbw_weights"],
        "dynamic_inputs": True,  # Can have MODEL_C for 3-model algorithms
    },
    "apply_lora": {
        "category": "processing",
        "inputs": [
            {"name": "MODEL", "type": "MODEL"},
            {"name": "LORA", "type": "LORA"},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": ["strength"],
    },
    "replace_vae": {
        "category": "processing",
        "inputs": [
            {"name": "MODEL", "type": "MODEL"},
            {"name": "VAE", "type": "VAE"},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": [],
    },
    "save_checkpoint": {
        "category": "output",
        "inputs": [{"name": "MODEL", "type": "MODEL"}],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": ["filename", "dtype", "metadata"],
    },
    "metadata_editor": {
        "category": "utility",
        "inputs": [{"name": "MODEL", "type": "MODEL"}],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
        "data_fields": ["metadata"],
    },
    "note": {
        "category": "utility",
        "inputs": [],
        "outputs": [],
        "data_fields": ["text"],
    },
}

# Type compatibility for connections
TYPE_COLORS = {
    "MODEL": "#4A9EFF",
    "LORA": "#FF6B9E",
    "VAE": "#9B59B6",
}


class CompilationError(Exception):
    """Error during graph compilation."""
    def __init__(self, message: str, node_id: Optional[str] = None):
        self.node_id = node_id
        super().__init__(message)


class ExecutionStep:
    """A single step in the execution plan."""
    def __init__(self, step_type: str, node_id: str, params: dict):
        self.step_type = step_type
        self.node_id = node_id
        self.params = params
    
    def to_dict(self) -> dict:
        return {
            "step_type": self.step_type,
            "node_id": self.node_id,
            "params": self.params,
        }


def validate_graph(graph: dict) -> list[str]:
    """
    Validate a graph structure. Returns a list of error messages (empty = valid).
    
    graph format:
    {
        "nodes": [{"id": str, "type": str, "data": dict}, ...],
        "connections": [{"from": {"node": str, "output": str}, "to": {"node": str, "input": str}}, ...]
    }
    """
    errors = []
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    connections = graph.get("connections", [])
    
    # Check all node types are valid
    for node in graph.get("nodes", []):
        if node["type"] not in NODE_TYPES:
            errors.append(f"Unknown node type: {node['type']} (node {node['id']})")
    
    # Check connections reference valid nodes and ports
    for conn in connections:
        from_node = conn["from"]["node"]
        to_node = conn["to"]["node"]
        
        if from_node not in nodes:
            errors.append(f"Connection references non-existent source node: {from_node}")
            continue
        if to_node not in nodes:
            errors.append(f"Connection references non-existent target node: {to_node}")
            continue
        
        from_type = nodes[from_node]["type"]
        to_type = nodes[to_node]["type"]
        
        if from_type not in NODE_TYPES or to_type not in NODE_TYPES:
            continue
        
        # Validate output exists
        from_def = NODE_TYPES[from_type]
        output_names = [o["name"] for o in from_def["outputs"]]
        if conn["from"]["output"] not in output_names:
            errors.append(
                f"Node {from_node} ({from_type}) has no output '{conn['from']['output']}'"
            )
        
        # Validate input exists (allow dynamic inputs for merge nodes)
        to_def = NODE_TYPES[to_type]
        input_names = [i["name"] for i in to_def["inputs"]]
        if conn["to"]["input"] not in input_names:
            if not to_def.get("dynamic_inputs"):
                errors.append(
                    f"Node {to_node} ({to_type}) has no input '{conn['to']['input']}'"
                )
        
        # Validate type compatibility
        from_output = next((o for o in from_def["outputs"] if o["name"] == conn["from"]["output"]), None)
        to_input = next((i for i in to_def["inputs"] if i["name"] == conn["to"]["input"]), None)
        
        if from_output and to_input and from_output["type"] != to_input["type"]:
            errors.append(
                f"Type mismatch: {from_output['type']} → {to_input['type']} "
                f"({from_node}.{conn['from']['output']} → {to_node}.{conn['to']['input']})"
            )
    
    # Check for cycles
    if not errors:
        if _has_cycle(nodes, connections):
            errors.append("Graph contains a cycle — cannot compile")
    
    # Check that save nodes have inputs connected
    for node in graph.get("nodes", []):
        if node["type"] == "save_checkpoint":
            has_input = any(c["to"]["node"] == node["id"] for c in connections)
            if not has_input:
                errors.append(f"Save node {node['id']} has no model input connected")
    
    return errors


def _has_cycle(nodes: dict, connections: list) -> bool:
    """Detect cycles using DFS."""
    # Build adjacency list
    adj = {nid: [] for nid in nodes}
    for conn in connections:
        from_id = conn["from"]["node"]
        to_id = conn["to"]["node"]
        if from_id in adj:
            adj[from_id].append(to_id)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in nodes}
    
    def dfs(node):
        color[node] = GRAY
        for neighbor in adj[node]:
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False
    
    for nid in nodes:
        if color[nid] == WHITE:
            if dfs(nid):
                return True
    return False


def topological_sort(nodes: dict, connections: list) -> list[str]:
    """Topological sort of node IDs using Kahn's algorithm."""
    # Build adjacency and in-degree
    adj = {nid: [] for nid in nodes}
    in_degree = {nid: 0 for nid in nodes}
    
    for conn in connections:
        from_id = conn["from"]["node"]
        to_id = conn["to"]["node"]
        if from_id in adj and to_id in in_degree:
            adj[from_id].append(to_id)
            in_degree[to_id] += 1
    
    # Start with nodes that have no inputs
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []
    
    while queue:
        # Sort for deterministic order
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result


def compile_graph(graph: dict) -> list[ExecutionStep]:
    """
    Compile a node graph into an ordered list of execution steps.
    
    Returns a list of ExecutionStep objects in execution order.
    """
    # Validate first
    errors = validate_graph(graph)
    if errors:
        raise CompilationError("Graph validation failed:\n" + "\n".join(errors))
    
    nodes = {n["id"]: n for n in graph["nodes"]}
    connections = graph.get("connections", [])
    
    # Filter out note nodes (they don't produce steps)
    active_nodes = {nid: n for nid, n in nodes.items() if n["type"] != "note"}
    
    # Topological sort
    sorted_ids = topological_sort(active_nodes, connections)
    
    # Build input map: for each node, what are its inputs connected to?
    input_map = {}  # node_id -> {input_name: (source_node_id, source_output_name)}
    for conn in connections:
        to_id = conn["to"]["node"]
        to_input = conn["to"]["input"]
        from_id = conn["from"]["node"]
        from_output = conn["from"]["output"]
        
        if to_id not in input_map:
            input_map[to_id] = {}
        input_map[to_id][to_input] = (from_id, from_output)
    
    # Generate execution steps
    steps = []
    
    for node_id in sorted_ids:
        if node_id not in active_nodes:
            continue
            
        node = active_nodes[node_id]
        node_type = node["type"]
        node_data = node.get("data", {})
        inputs = input_map.get(node_id, {})
        
        if node_type == "checkpoint_loader":
            steps.append(ExecutionStep(
                step_type="load_checkpoint",
                node_id=node_id,
                params={
                    "file": node_data.get("file", ""),
                },
            ))
        
        elif node_type == "lora_loader":
            steps.append(ExecutionStep(
                step_type="load_lora",
                node_id=node_id,
                params={
                    "file": node_data.get("file", ""),
                    "strength": node_data.get("strength", 1.0),
                },
            ))
        
        elif node_type == "vae_loader":
            steps.append(ExecutionStep(
                step_type="load_vae",
                node_id=node_id,
                params={
                    "file": node_data.get("file", ""),
                },
            ))
        
        elif node_type == "merge_models":
            model_inputs = {}
            for input_name, (src_id, src_output) in inputs.items():
                model_inputs[input_name] = src_id
            
            steps.append(ExecutionStep(
                step_type="merge",
                node_id=node_id,
                params={
                    "algorithm": node_data.get("algorithm", "weighted_sum"),
                    "params": node_data.get("params", {"alpha": 0.5}),
                    "model_inputs": model_inputs,
                    "use_mbw": node_data.get("use_mbw", False),
                    "mbw_weights": node_data.get("mbw_weights", {}),
                },
            ))
        
        elif node_type == "apply_lora":
            model_src = inputs.get("MODEL", (None, None))[0]
            lora_src = inputs.get("LORA", (None, None))[0]
            
            steps.append(ExecutionStep(
                step_type="apply_lora",
                node_id=node_id,
                params={
                    "model_source": model_src,
                    "lora_source": lora_src,
                    "strength": node_data.get("strength", 1.0),
                },
            ))
        
        elif node_type == "replace_vae":
            model_src = inputs.get("MODEL", (None, None))[0]
            vae_src = inputs.get("VAE", (None, None))[0]
            
            steps.append(ExecutionStep(
                step_type="replace_vae",
                node_id=node_id,
                params={
                    "model_source": model_src,
                    "vae_source": vae_src,
                },
            ))
        
        elif node_type == "metadata_editor":
            model_src = inputs.get("MODEL", (None, None))[0]
            
            steps.append(ExecutionStep(
                step_type="edit_metadata",
                node_id=node_id,
                params={
                    "model_source": model_src,
                    "metadata": node_data.get("metadata", {}),
                },
            ))
        
        elif node_type == "save_checkpoint":
            model_src = inputs.get("MODEL", (None, None))[0]
            
            steps.append(ExecutionStep(
                step_type="save",
                node_id=node_id,
                params={
                    "model_source": model_src,
                    "filename": node_data.get("filename", "merged_model.safetensors"),
                    "dtype": node_data.get("dtype", "fp16"),
                    "metadata": node_data.get("metadata", {}),
                },
            ))
    
    return steps


def get_node_types_info() -> dict:
    """Return node type definitions for the frontend."""
    result = {}
    for name, definition in NODE_TYPES.items():
        result[name] = {
            "category": definition["category"],
            "inputs": definition["inputs"],
            "outputs": definition["outputs"],
            "data_fields": definition["data_fields"],
            "dynamic_inputs": definition.get("dynamic_inputs", False),
        }
    return result
