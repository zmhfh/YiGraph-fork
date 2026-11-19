import os
import sys

# Ensure CWD is the project root so that subprocesses
# (e.g. MCP servers launched via `python -m aag.xxx`)
# can find the `aag` package through Python's implicit
# CWD-on-sys.path behaviour.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Fix argv paths to absolute before chdir, otherwise Flask's reloader
# can't find this script on restart (Werkzeug reads sys.orig_argv[1]).
if not os.path.isabs(sys.argv[0]):
    sys.argv[0] = os.path.abspath(sys.argv[0])
if hasattr(sys, 'orig_argv') and len(sys.orig_argv) > 1:
    if not os.path.isabs(sys.orig_argv[1]):
        sys.orig_argv[1] = os.path.abspath(sys.orig_argv[1])

os.chdir(_PROJECT_ROOT)
# The script's own directory is still on sys.path[0], so
# `from route import ...` continues to work regardless of CWD.

import socket
from route import create_app

app = create_app()

def find_free_port(start_port=5089, max_tries=50):
    """Find an available port starting from start_port."""
    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Match Werkzeug's reuse behavior when a stopped instance leaves
            # connections in TIME_WAIT; do not mistake that for a live server.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return port  # This port is free
            except OSError:
                port += 1
    raise RuntimeError("No available ports found.")

if __name__ == "__main__":
    port = find_free_port()
    print(f"Using available port: {port}")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port,
        threaded=True,
    )
