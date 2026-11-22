"""Persistent local inference process; no TCP port or SSH tunnel."""
import atexit
import json
import os
import queue
import subprocess
import threading
import uuid
from pathlib import Path

_lock = threading.Lock()
_process = None
_responses = None
_command = None
_log = None


def close():
    global _process, _responses, _command, _log
    if _process is not None:
        if _process.poll() is None:
            _process.terminate()
            try:
                _process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _process.kill()
                _process.wait(timeout=5)
        for stream in (_process.stdin, _process.stdout):
            if stream:
                stream.close()
    if _log:
        _log.close()
    _process, _responses, _command, _log = None, None, None, None


def _read_lines(stream, responses):
    try:
        for line in stream:
            responses.put(line)
    finally:
        responses.put(None)


def request_local(request):
    global _process, _responses, _command, _log
    with _lock:
        command = json.loads(os.environ["YIGRAPH_BINDER_COMMAND"])
        if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
            raise ValueError("YIGRAPH_BINDER_COMMAND must be a JSON array of executable and arguments")
        if _process is None or _process.poll() is not None or _command != command:
            close()
            logfile = Path(os.environ["YIGRAPH_BINDER_LOG"]).expanduser()
            logfile.parent.mkdir(parents=True, exist_ok=True)
            _log = logfile.open("a", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = os.environ["YIGRAPH_BINDER_CUDA_VISIBLE_DEVICES"]
            env["PYTHONUNBUFFERED"] = "1"
            _process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=_log, text=True, encoding="utf-8", bufsize=1, env=env)
            _command, _responses = command, queue.Queue()
            threading.Thread(target=_read_lines, args=(_process.stdout, _responses), daemon=True).start()
        token = uuid.uuid4().hex
        try:
            _process.stdin.write(json.dumps({"request_id": token, "request": request}, ensure_ascii=False) + "\n")
            _process.stdin.flush()
            line = _responses.get(timeout=float(os.environ.get("YIGRAPH_BINDER_TIMEOUT", "180")))
            if line is None:
                raise RuntimeError("Local Binder exited; inspect YIGRAPH_BINDER_LOG")
            obj = json.loads(line)
            if obj.get("request_id") != token:
                raise RuntimeError("Unexpected worker output; inspect YIGRAPH_BINDER_LOG")
            if "error" in obj:
                raise RuntimeError("Local Binder: " + str(obj["error"]))
            return obj["result"]
        except Exception:
            close()
            raise


atexit.register(close)
