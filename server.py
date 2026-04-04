"""
SDXL Node Merger — Main Server
HTTP server for static files + REST API
WebSocket server for real-time progress updates

Copyright 2026 George Kogan
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import json
import asyncio
import threading
import webbrowser
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import websockets

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from engine import graph_compiler, merge_executor, tensor_io, algorithms, metadata
from engine import lora_utils


# ─── Configuration ───────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")
PROJECTS_DIR = os.path.join(PROJECT_ROOT, "projects")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")


def load_config() -> dict:
    """Load application configuration."""
    default = {
        "directories": {"checkpoints": "", "lora": "", "vae": "", "output": ""},
        "server": {"host": "127.0.0.1", "http_port": 8765, "ws_port": 8766},
        "language": "en",
        "theme": "midnight",
        "merge": {"low_vram": False, "default_dtype": "fp16", "device": "auto"},
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Deep merge: saved values override defaults
            _deep_merge(default, saved)
            print(f"  [CONFIG] Loaded from: {CONFIG_PATH}")
            if default["directories"].get("checkpoints"):
                print(f"  [CONFIG] Checkpoints: {default['directories']['checkpoints']}")
        except Exception as e:
            print(f"  [CONFIG] Error loading config: {e}")
    else:
        print(f"  [CONFIG] No config found, using defaults")
    return default


def _deep_merge(base: dict, override: dict):
    """Deep merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def save_config(config: dict):
    """Save application configuration."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [CONFIG] Saved to: {CONFIG_PATH}")


config = load_config()


# ─── WebSocket Clients ──────────────────────────────────────────────────────────

ws_clients = set()
MAIN_LOOP = None


async def ws_broadcast(message: dict):
    """Broadcast a message to all WebSocket clients."""
    global ws_clients
    if ws_clients:
        data = json.dumps(message, ensure_ascii=False)
        disconnected = set()
        for ws in ws_clients:
            try:
                await ws.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(ws)
        ws_clients -= disconnected


def ws_broadcast_sync(message: dict):
    """Broadcast from a synchronous context to the main event loop."""
    global MAIN_LOOP
    if MAIN_LOOP and MAIN_LOOP.is_running():
        try:
            asyncio.run_coroutine_threadsafe(ws_broadcast(message), MAIN_LOOP)
        except Exception:
            pass


# ─── HTTP Request Handler ───────────────────────────────────────────────────────

class MergerHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler for static files and REST API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self._handle_api_get(path, parse_qs(parsed.query))
        else:
            # Serve static files
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b""
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                data = {}
            self._handle_api_post(path, data)
        else:
            self.send_error(404)

    def _send_json(self, data: dict, status: int = 200):
        """Send a JSON response."""
        response = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(response)

    def _handle_api_get(self, path: str, params: dict):
        """Handle GET API requests."""
        global config

        if path == "/api/config":
            self._send_json(config)

        elif path == "/api/scan-models":
            dir_type = params.get("type", ["checkpoints"])[0]
            directory = config["directories"].get(dir_type, "")
            models = tensor_io.scan_directory(directory)
            self._send_json({"files": models, "directory": directory})

        elif path == "/api/model-info":
            filepath = params.get("path", [""])[0]
            if not filepath or not os.path.exists(filepath):
                self._send_json({"error": "File not found"}, 404)
                return
            info = tensor_io.read_metadata(filepath)
            type_info = tensor_io.get_model_type_info(filepath)
            info.update(type_info)
            self._send_json(info)

        elif path == "/api/algorithms":
            self._send_json({"algorithms": algorithms.get_algorithm_info()})

        elif path == "/api/node-types":
            self._send_json({"types": graph_compiler.get_node_types_info()})

        elif path == "/api/projects":
            projects = self._list_projects()
            self._send_json({"projects": projects})

        elif path == "/api/load-project":
            name = params.get("name", [""])[0]
            project = self._load_project(name)
            if project:
                self._send_json(project)
            else:
                self._send_json({"error": "Project not found"}, 404)

        elif path == "/api/languages":
            langs = self._list_languages()
            self._send_json({"languages": langs})

        elif path == "/api/language":
            lang = params.get("lang", [config.get("language", "en")])[0]
            lang_data = self._load_language(lang)
            self._send_json(lang_data)

        elif path == "/api/themes":
            themes = self._list_themes()
            self._send_json({"themes": themes})

        elif path == "/api/theme":
            theme = params.get("name", [config.get("theme", "midnight")])[0]
            theme_data = self._load_theme(theme)
            self._send_json(theme_data)

        elif path == "/api/mbw-blocks":
            self._send_json({"blocks": tensor_io.MBW_BLOCK_IDS})

        else:
            self._send_json({"error": "Unknown API endpoint"}, 404)

    def _handle_api_post(self, path: str, data: dict):
        """Handle POST API requests."""
        global config

        if path == "/api/config":
            _deep_merge(config, data)
            save_config(config)
            self._send_json({"status": "ok"})

        elif path == "/api/save-project":
            name = data.get("name", "untitled")
            project_data = data.get("project", {})
            self._save_project(name, project_data)
            self._send_json({"status": "ok", "name": name})

        elif path == "/api/delete-project":
            name = data.get("name", "")
            self._delete_project(name)
            self._send_json({"status": "ok"})

        elif path == "/api/validate-graph":
            errors = graph_compiler.validate_graph(data)
            self._send_json({"valid": len(errors) == 0, "errors": errors})

        elif path == "/api/execute":
            # Start merge execution in a background thread
            threading.Thread(
                target=self._execute_merge,
                args=(data,),
                daemon=True,
            ).start()
            self._send_json({"status": "started"})

        elif path == "/api/browse-directory":
            # Return directory listing for directory picker
            dir_path = data.get("path", "")
            result = self._browse_directory(dir_path)
            self._send_json(result)

        else:
            self._send_json({"error": "Unknown API endpoint"}, 404)

    # ─── Project Management ──────────────────────────────────────────────────

    def _list_projects(self) -> list[dict]:
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        projects = []
        for f in sorted(os.listdir(PROJECTS_DIR)):
            if f.endswith(".json"):
                full = os.path.join(PROJECTS_DIR, f)
                try:
                    with open(full, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    projects.append({
                        "name": f[:-5],
                        "filename": f,
                        "modified": data.get("modified", ""),
                        "node_count": len(data.get("nodes", [])),
                    })
                except Exception:
                    continue
        return projects

    def _save_project(self, name: str, data: dict):
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        safe_name = "".join(c for c in name if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            safe_name = "untitled"
        filepath = os.path.join(PROJECTS_DIR, safe_name + ".json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_project(self, name: str) -> dict | None:
        filepath = os.path.join(PROJECTS_DIR, name + ".json")
        if not os.path.exists(filepath):
            # Try with the name as-is
            filepath2 = os.path.join(PROJECTS_DIR, name)
            if os.path.exists(filepath2):
                filepath = filepath2
            else:
                return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _delete_project(self, name: str):
        filepath = os.path.join(PROJECTS_DIR, name + ".json")
        if os.path.exists(filepath):
            os.remove(filepath)

    # ─── Language & Theme ────────────────────────────────────────────────────

    def _list_languages(self) -> list[dict]:
        lang_dir = os.path.join(WEB_DIR, "lang")
        langs = []
        if os.path.isdir(lang_dir):
            for f in sorted(os.listdir(lang_dir)):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(lang_dir, f), "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        langs.append({
                            "code": f[:-5],
                            "name": data.get("_name", f[:-5]),
                        })
                    except Exception:
                        continue
        return langs

    def _load_language(self, lang: str) -> dict:
        filepath = os.path.join(WEB_DIR, "lang", lang + ".json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _list_themes(self) -> list[dict]:
        theme_dir = os.path.join(WEB_DIR, "themes")
        themes = []
        if os.path.isdir(theme_dir):
            for f in sorted(os.listdir(theme_dir)):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(theme_dir, f), "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        themes.append({
                            "id": f[:-5],
                            "name": data.get("name", f[:-5]),
                        })
                    except Exception:
                        continue
        return themes

    def _load_theme(self, name: str) -> dict:
        filepath = os.path.join(WEB_DIR, "themes", name + ".json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # ─── Directory Browser ───────────────────────────────────────────────────

    def _browse_directory(self, path: str) -> dict:
        if not path:
            # Return drives on Windows
            if sys.platform == "win32":
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:\\"
                    if os.path.exists(drive):
                        drives.append({"name": drive, "path": drive, "is_dir": True})
                return {"items": drives, "current": ""}
            else:
                path = "/"

        if not os.path.isdir(path):
            return {"items": [], "current": path, "error": "Not a directory"}

        items = []
        # Add parent directory
        parent = os.path.dirname(path.rstrip("/\\"))
        if parent != path:
            items.append({"name": "..", "path": parent, "is_dir": True})

        try:
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                try:
                    is_dir = os.path.isdir(full)
                    items.append({
                        "name": entry,
                        "path": full,
                        "is_dir": is_dir,
                    })
                except PermissionError:
                    continue
        except PermissionError:
            return {"items": items, "current": path, "error": "Permission denied"}

        return {"items": items, "current": path}

    # ─── Merge Execution ─────────────────────────────────────────────────────

    def _execute_merge(self, graph_data: dict):
        """Execute merge in background thread with WebSocket progress."""
        import traceback

        def progress_callback(progress_data):
            """Send progress updates via WebSocket and terminal."""
            msg_type = progress_data.get("type", "")
            if msg_type == "progress":
                pct = progress_data.get("overall_progress", 0) * 100
                step_info = progress_data.get("operation", "")
                step_num = progress_data.get("current_step", 0)
                total = progress_data.get("total_steps", 0)
                print(f"  [{step_num}/{total}] {pct:.0f}% — {step_info}")
            elif msg_type == "log":
                print(f"  [LOG] {progress_data.get('message', '')}")
            
            ws_broadcast_sync(progress_data)

        try:
            print()
            print("=" * 60)
            print("  MERGE EXECUTION STARTED")
            print("=" * 60)

            ws_broadcast_sync({
                "type": "execution_start",
                "message": "Merge execution started",
            })

            # Compile graph
            print("  [COMPILE] Compiling graph...")
            steps = graph_compiler.compile_graph(graph_data)
            print(f"  [COMPILE] Generated {len(steps)} execution steps")
            for i, step in enumerate(steps):
                print(f"    Step {i+1}: {step.step_type} (node: {step.node_id})")

            # Execute
            print("  [EXECUTE] Starting execution...")
            result = merge_executor.execute_plan(
                steps=steps,
                config=config,
                low_vram=config.get("merge", {}).get("low_vram", False),
                device=config.get("merge", {}).get("device", "auto"),
                progress_callback=progress_callback,
            )

            print()
            print("=" * 60)
            print("  MERGE COMPLETE!")
            print(f"  Result: {result}")
            print("=" * 60)
            print()

            ws_broadcast_sync({
                "type": "execution_complete",
                "result": result,
            })

        except Exception as e:
            print()
            print("!" * 60)
            print(f"  MERGE ERROR: {e}")
            print()
            traceback.print_exc()
            print("!" * 60)
            print()

            ws_broadcast_sync({
                "type": "execution_error",
                "error": str(e),
            })

    def log_message(self, format, *args):
        """Log all HTTP requests to terminal."""
        super().log_message(format, *args)


# ─── WebSocket Handler ───────────────────────────────────────────────────────────

async def ws_handler(websocket, path=None):
    """Handle WebSocket connections."""
    global ws_clients
    ws_clients.add(websocket)
    try:
        await websocket.send(json.dumps({
            "type": "connected",
            "message": "WebSocket connected",
        }))
        async for message in websocket:
            # Handle incoming WebSocket messages if needed
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        ws_clients.discard(websocket)


# ─── Main Entry Point ───────────────────────────────────────────────────────────

async def start_ws_server(host: str, port: int):
    """Start the WebSocket server."""
    server = await websockets.serve(ws_handler, host, port)
    print(f"  WebSocket server: ws://{host}:{port}")
    return server


def start_http_server(host: str, port: int):
    """Start the HTTP server in a thread."""
    httpd = HTTPServer((host, port), MergerHTTPHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"  HTTP server:      http://{host}:{port}")
    return httpd


def main():
    """Main entry point."""
    print()
    print("=" * 60)
    print("  SDXL Node Merger v1.0")
    print("  Created by George Kogan")
    print("  Licensed under Apache 2.0")
    print("=" * 60)
    print()

    # Ensure directories exist
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    os.makedirs(WEB_DIR, exist_ok=True)

    host = config["server"]["host"]
    http_port = config["server"]["http_port"]
    ws_port = config["server"]["ws_port"]

    # Start HTTP server
    httpd = start_http_server(host, http_port)
    print()
    print(f"  Open in browser: http://{host}:{http_port}")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    # Open browser
    webbrowser.open(f"http://{host}:{http_port}")

    # Start WebSocket server (async)
    global MAIN_LOOP
    MAIN_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(MAIN_LOOP)

    ws_server = None
    try:
        ws_server = MAIN_LOOP.run_until_complete(start_ws_server(host, ws_port))
        MAIN_LOOP.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop WebSocket server
        if ws_server:
            ws_server.close()
            MAIN_LOOP.run_until_complete(ws_server.wait_closed())
        
        # Stop HTTP server
        httpd.shutdown()
        
        # Cancel remaining async tasks
        pending = asyncio.all_tasks(MAIN_LOOP)
        for task in pending:
            task.cancel()
        if pending:
            MAIN_LOOP.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        MAIN_LOOP.close()
        print("  Server stopped.")
        
        # Force exit — daemon threads and sockets can linger on Windows
        os._exit(0)


if __name__ == "__main__":
    main()
