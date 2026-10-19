#!/usr/bin/env python3
"""
AI Second-Hand Item Analyzer - High Performance Localhost Server Runner
Optimized for:
- Low-end CPUs and low-RAM devices (2GB-4GB RAM friendly)
- High concurrency: Handles multiple simultaneous users without crashing
- Automatic port resolution and browser launch
- Local network access (use from phone/laptop on same Wi-Fi)
"""

import os
import sys
import socket
import webbrowser
import threading
import time

def find_available_port(start_port=8000, max_attempts=50):
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', p)) != 0:
                return p
    return start_port

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def open_browser_delayed(url, delay=1.5):
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Set low-end CPU thread limits for PyTorch and BLAS
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"

    port = find_available_port(int(os.getenv("PORT", 8000)))
    local_ip = get_local_ip()

    print("=" * 64)
    print("  * AI Second-Hand Item Analyzer - Localhost Server")
    print("=" * 64)
    print(f"  * Local URL:      http://localhost:{port}")
    print(f"  * Network URL:    http://{local_ip}:{port}")
    print(f"  * Environment:    Local Hardware (CPU & RAM Optimized)")
    print(f"  * Concurrency:    Protected with threadpool & semaphore queue")
    print("=" * 64)
    print("  Press Ctrl+C at any time to shut down the server.\n")

    open_browser_delayed(f"http://localhost:{port}")

    try:
        import uvicorn
        from backend.app import app
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=False,
            timeout_keep_alive=30,
            limit_concurrency=100,
            backlog=128
        )
    except KeyboardInterrupt:
        print("\n[Server Shutdown] Server stopped gracefully. Goodbye!")
    except Exception as e:
        print(f"\n[Error launching server]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
