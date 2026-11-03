"""Open the local web app: Steam login page + the opening report.

    python -m src.app [--port 8765]

Starts tools/steam_gc/server.js (installing its Node packages on first run) and opens the
login page in your browser. Stop it with Ctrl+C.
"""

import argparse
import subprocess
import threading
import time
import webbrowser

from src.pipeline import STEAM_TOOL, ensure_node_tool


def main():
    ap = argparse.ArgumentParser(description="Steam login page + opening report in the browser")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    ensure_node_tool()
    url = f"http://127.0.0.1:{args.port}/login"
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    try:
        subprocess.run(["node", str(STEAM_TOOL / "server.js"), str(args.port)], check=True)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
