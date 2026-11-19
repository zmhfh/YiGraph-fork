"""
Chat streaming over plain HTTP (NDJSON lines).

POST /api/chat/stream  body: {message, model, dataset, ..., request_id, resume}

Response is a stream of JSON events, one per line (same envelope as the old
chat_response protocol). Disconnected clients can resume a stream by
re-POSTing with resume=true and the same request_id: buffered events are
replayed and the new connection joins the live stream.
"""

import asyncio
import json
import logging
import queue
import threading
from typing import Any, Dict, Optional

from flask import Blueprint, Response, jsonify, request

from aag.api.async_runtime import get_background_loop, get_chat_service
from aag.api.services.chat_service import CHAT_FRIENDLY_ERROR_MSG
from aag.engine.react_trace import ReactTraceStore

logger = logging.getLogger(__name__)

bp = Blueprint("chat_stream", __name__)


@bp.route("/api/chat/runs", methods=["GET"])
def list_runs():
    return jsonify({"success": True, "data": ReactTraceStore().recent()})


@bp.route("/api/chat/runs/<run_id>", methods=["GET"])
def get_run(run_id):
    try:
        return jsonify({"success": True, "data": ReactTraceStore().get(run_id)})
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Run not found"}), 404
    except ValueError:
        return jsonify({"success": False, "error": "Invalid run id"}), 400


@bp.route("/api/chat/runs/<run_id>/dag", methods=["GET"])
def get_run_dag(run_id):
    """Refresh an older display from saved evidence without executing its query."""
    from aag.models.graph_workflow_dag import GraphWorkflowDAG, dag_info_to_frontend_format
    try:
        record = ReactTraceStore().get(run_id)
        dag = GraphWorkflowDAG.from_react_trace(record.get("events", []))
        return jsonify({"success": True, "data": dag_info_to_frontend_format(dag.get_dag_info())})
    except FileNotFoundError:
        return jsonify({"success": False, "error": "Run not found"}), 404
    except ValueError:
        return jsonify({"success": False, "error": "Invalid run id"}), 400

# In-memory per-request event buffers for resume support.
# request_id -> {"events": [...], "queues": set, "finished": bool, "lock": Lock}
_streams: Dict[str, Dict[str, Any]] = {}
MAX_STREAMS = 50
KEEPALIVE_SECONDS = 15


def _register_stream(request_id: str) -> Dict[str, Any]:
    if len(_streams) >= MAX_STREAMS:
        _streams.pop(next(iter(_streams)))
    stream = {
        "events": [],
        "queues": set(),
        "finished": False,
        "lock": threading.Lock(),
    }
    _streams[request_id] = stream
    return stream


def _make_sender(stream: Dict[str, Any]):
    def send(chunk: Dict[str, Any]):
        if stream["finished"]:
            return
        with stream["lock"]:
            chunk = dict(chunk)
            chunk["seq"] = len(stream["events"])
            stream["events"].append(chunk)
            queues = list(stream["queues"])
        for q in queues:
            q.put(chunk)
        if chunk.get("type") == "stream_end":
            with stream["lock"]:
                stream["finished"] = True
                queues = list(stream["queues"])
            for q in queues:
                q.put(None)
    return send


async def _run_chat_stream(payload: Dict[str, Any], send) -> None:
    try:
        chat_service = get_chat_service()

        user_message = str(payload.get("message") or "").strip()
        model = str(payload.get("model") or "")
        dataset = str(payload.get("dataset") or "").strip()
        dataset_type = payload.get("dataset_type")
        dag_confirm = str(payload.get("dag_confirm") or "").strip()
        is_dag_modification = str(payload.get("is_dag_modification", "false")).lower() == "true"

        if dag_confirm or is_dag_modification:
            send({'error': 'ReAct runs directly. Submit a new question to continue the analysis.'})
            send({'type': 'stream_end'})
            return

        mode = "normal"
        logger.info(f"Stream chat: mode={mode}, model={model}, message={user_message[:50]}...")
        await chat_service.process_streaming_chat(
            message=user_message,
            model=model,
            dataset=dataset,
            dataset_type=dataset_type,
            mode=mode,
            expert_mode=False,
            callback=send,
        )
    except Exception as exc:
        logger.error(f"Background processing failed: {exc}", exc_info=True)
        send({'type': 'result', 'contentType': 'text', 'content': CHAT_FRIENDLY_ERROR_MSG})
        send({'type': 'stream_end'})


def _event_stream(stream: Dict[str, Any], q: "queue.Queue") -> Response:
    def generate():
        try:
            while True:
                try:
                    item = q.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    return
                yield json.dumps(item, ensure_ascii=False) + "\n"
        finally:
            with stream["lock"]:
                stream["queues"].discard(q)

    return Response(
        generate(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _attach_queue(stream: Dict[str, Any], q: "queue.Queue") -> None:
    with stream["lock"]:
        stream["queues"].add(q)


def _error_response(message: str) -> Response:
    return jsonify({"error": message}), 400


@bp.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    payload = request.get_json(silent=True) or {}
    request_id = str(payload.get("request_id") or "")
    resume = bool(payload.get("resume"))

    if resume:
        if not request_id:
            return _error_response("Invalid request format. Please check parameters.")
        stream = _streams.get(request_id)
        if not stream:
            # Buffer gone (backend restarted or evicted): fail the client stream.
            return _error_response("Resume buffer not found.")
        q: "queue.Queue" = queue.Queue()

        def replay():
            with stream["lock"]:
                snapshot = list(stream["events"])
                stream["queues"].add(q)
                finished = stream["finished"]
            for chunk in snapshot:
                q.put(chunk)
            if finished:
                q.put(None)

        # Replay on the background loop thread to preserve ordering with live sends.
        get_background_loop().call_soon_threadsafe(replay)
        return _event_stream(stream, q)

    user_message = str(payload.get("message") or "").strip()
    dag_confirm = str(payload.get("dag_confirm") or "").strip()
    is_dag_modification = str(payload.get("is_dag_modification", "false")).lower() == "true"
    dataset = str(payload.get("dataset") or "").strip()

    if dag_confirm or is_dag_modification:
        return _error_response("ReAct runs directly. Submit a new question to continue the analysis.")
    if not request_id:
        return _error_response("Invalid request format. Please check parameters.")
    if not user_message:
        return _error_response("Message content cannot be empty.")
    if not dataset:
        return _error_response("Dataset is empty. Please specify a dataset first.")

    stream = _register_stream(request_id)
    send = _make_sender(stream)
    q: "queue.Queue" = queue.Queue()
    _attach_queue(stream, q)

    loop = get_background_loop()
    asyncio.run_coroutine_threadsafe(_run_chat_stream(payload, send), loop)

    return _event_stream(stream, q)
