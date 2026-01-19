"""Cloud graph sync helper for real-time updates.

This module provides functions to sync memora data to Cloudflare D1
and notify connected WebSocket clients of updates.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

# Auto-detect sync script location (sibling memora-graph directory)
_THIS_DIR = Path(__file__).parent
_DEFAULT_SYNC_SCRIPT = _THIS_DIR.parent / "memora-graph" / "scripts" / "sync.sh"

# Configuration from environment
CLOUD_GRAPH_ENABLED = os.getenv("MEMORA_CLOUD_GRAPH_ENABLED", "").lower() in ("true", "1", "yes")
CLOUD_GRAPH_WORKER_URL = os.getenv(
    "MEMORA_CLOUD_GRAPH_WORKER_URL",
    "https://memora-graph-sync.cloudflare-strategic612.workers.dev"
)
CLOUD_GRAPH_SYNC_SCRIPT = os.getenv("MEMORA_CLOUD_GRAPH_SYNC_SCRIPT", "") or (
    str(_DEFAULT_SYNC_SCRIPT) if _DEFAULT_SYNC_SCRIPT.exists() else ""
)

# Debounce settings - batch rapid writes
_sync_timer: Optional[threading.Timer] = None
_sync_lock = threading.Lock()
SYNC_DEBOUNCE_SECONDS = float(os.getenv("MEMORA_CLOUD_GRAPH_DEBOUNCE", "2.0"))


def _do_sync() -> None:
    """Perform the actual sync operation."""
    global _sync_timer
    _sync_timer = None

    if not CLOUD_GRAPH_ENABLED:
        return

    try:
        # Run the sync script if configured
        if CLOUD_GRAPH_SYNC_SCRIPT:
            script_path = Path(CLOUD_GRAPH_SYNC_SCRIPT).expanduser()
            if script_path.exists():
                subprocess.run(
                    [str(script_path), "--remote"],
                    capture_output=True,
                    timeout=60,
                )

        # Always notify WebSocket clients
        _broadcast_update()

    except Exception:
        # Don't fail the main operation if sync fails
        pass


def _broadcast_update() -> None:
    """Notify connected WebSocket clients of an update."""
    try:
        url = f"{CLOUD_GRAPH_WORKER_URL}/broadcast"
        req = Request(
            url,
            data=json.dumps({}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read().decode())
            sent = result.get("sent", 0)
            if sent > 0:
                print(f"[cloud_sync] Broadcast sent to {sent} client(s)")
    except URLError as e:
        print(f"[cloud_sync] Warning: broadcast failed: {e}")
    except Exception as e:
        print(f"[cloud_sync] Warning: broadcast error: {e}")


def schedule_sync() -> None:
    """Schedule a sync operation with debouncing.

    Multiple rapid writes will be batched into a single sync
    after SYNC_DEBOUNCE_SECONDS of inactivity.
    """
    global _sync_timer

    if not CLOUD_GRAPH_ENABLED:
        return

    with _sync_lock:
        # Cancel any pending sync
        if _sync_timer is not None:
            _sync_timer.cancel()

        # Schedule new sync after debounce period
        _sync_timer = threading.Timer(SYNC_DEBOUNCE_SECONDS, _do_sync)
        _sync_timer.daemon = True
        _sync_timer.start()


def sync_now() -> None:
    """Perform sync immediately without debouncing."""
    global _sync_timer

    if not CLOUD_GRAPH_ENABLED:
        return

    with _sync_lock:
        # Cancel any pending sync
        if _sync_timer is not None:
            _sync_timer.cancel()
            _sync_timer = None

    # Run sync in background thread
    thread = threading.Thread(target=_do_sync, daemon=True)
    thread.start()
