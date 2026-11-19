"""
Chat service: wraps AAG engine chat, handles user messages.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from .engine_service import EngineService
from aag.models.graph_workflow_dag import (
    WorkflowStep,
    dag_info_to_frontend_format,
    workflow_step_to_frontend,
)

logger = logging.getLogger(__name__)

# Unified error message for chat UI (returned as type: result so it shows in the bubble, not a popup)
CHAT_FRIENDLY_ERROR_MSG = "Request failed. Please try again later."


def _is_scheduler_error_text(text: str) -> bool:
    """Whether the text is a scheduler business error string (should be shown as friendly message)."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    return t.startswith(("⚠️", "❌", "错误：", "Error:"))


class ChatService:
    """
    Chat service: wraps AAG engine chat.
    Modes: normal / interact / expert.
    """

    def __init__(self):
        """Initialize chat service."""
        self.engine_service = EngineService.get_instance()
    
    async def process_message(
        self,
        message: str,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        mode: str = "normal",
        expert_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Process a user message (sync-style API).

        Args:
            message: User message.
            model: Model name (optional, for logging).
            dataset: Dataset name (optional).
            mode: "normal" | "interact" | "expert".
            expert_mode: Legacy frontend flag (True => use interact).

        Returns:
            Result dict.
        """
        try:
            mode = "normal"

            engine = self.engine_service.get_engine()

            if dataset:
                logger.info(f"Select dataset: {dataset}")
                engine.specific_dataset(dataset)

            logger.info(f"Process message: mode={mode}, model={model}, message={message[:50]}...")

            result = await engine.run(message, mode=mode)

            return {
                "success": True,
                "result": result,
                "mode": mode
            }
        except Exception as e:
            logger.error(f"Process message failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "mode": mode
            }
    
    async def process_dag_modification(
        self,
        modification_request: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Handle DAG modification request (expert mode).

        Args:
            modification_request: User modification request.
            callback: Optional callback for streaming events
                (subproblems after revision / algorithm_selected per step).

        Returns:
            Dict with updated DAG info.
        """
        try:
            engine = self.engine_service.get_engine()
            logger.info(f"Process DAG modification: {modification_request[:50]}...")

            result = await engine.expert_modify_dag(modification_request, callback=callback)

            return {
                "success": True,
                **result
            }
        except Exception as e:
            logger.error(f"DAG modification failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_expert_analysis(self) -> Dict[str, Any]:
        """
        Start expert-mode analysis.

        Returns:
            Analysis result dict.
        """
        try:
            engine = self.engine_service.get_engine()
            logger.info("Start expert-mode analysis")

            result = await engine.expert_start_analysis()

            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Expert analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_streaming_chat(
        self,
        message: str,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        dataset_type: Optional[str] = None,
        mode: str = "normal",
        expert_mode: bool = False,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Stream chat (for WebSocket).

        Args:
            message: User message.
            model: Model name (optional).
            dataset: Dataset name (optional).
            dataset_type: "text" | "graph" (optional).
            mode: "normal" | "interact" | "expert".
            expert_mode: Legacy flag (True => interact).
            callback: Callback for streaming chunks; signature callback(data: Dict[str, Any]).
        """
        try:
            mode = "normal"

            engine = self.engine_service.get_engine()

            if callback:
                callback({
                    "type": "thinking",
                    "contentType": "text",
                    "content": "Analyzing your question...",
                    "placeholder": True
                })

            # 包装回调：ReAct 会先流式下发动作/观测，执行结束后再发送
            # 由 trace 投影的 DAG；这里跟踪并避免重复发送。
            streamed_dag = {"value": False}

            def wrapped_callback(data_chunk):
                if isinstance(data_chunk, WorkflowStep):
                    data_chunk = workflow_step_to_frontend(data_chunk)
                if isinstance(data_chunk, dict) and data_chunk.get("contentType") == "dag":
                    streamed_dag["value"] = True
                callback(data_chunk)

            engine_callback = wrapped_callback if callback else None

            if dataset:
                logger.info(f"Select dataset: {dataset}, type: {dataset_type}")
                engine.specific_dataset(dataset, dataset_type)

            logger.info(f"Stream chat: mode={mode}, message={message[:50]}...")

            result = await engine.run(message, mode=mode, callback=engine_callback)

            if isinstance(result, dict):
                if callback and result.get("dag_info") and not streamed_dag["value"]:
                    callback({
                        "type": "result", "contentType": "dag",
                        "content": self._convert_dag_to_frontend_format(result),
                    })
                result_text = result.get("analysis_result") or result.get("error") or ""
            else:
                result_text = str(result) if result else "No result."
            if _is_scheduler_error_text(result_text):
                result_text = CHAT_FRIENDLY_ERROR_MSG
            if callback and result_text:
                # Preserve Markdown tables, lists and code blocks in the report.
                callback({"type": "result", "contentType": "text", "content": result_text})
            if callback:
                callback({
                    "type": "stream_end"
                })
                
        except Exception as e:
            logger.error(f"Stream chat failed: {str(e)}", exc_info=True)
            if callback:
                callback({
                    "type": "result",
                    "contentType": "text",
                    "content": CHAT_FRIENDLY_ERROR_MSG
                })
                callback({"type": "stream_end"})
            return
    
    def _convert_dag_to_frontend_format(self, dag_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert DAG result to frontend format (delegates to shared helper).

        Args:
            dag_result: DAG result from AAG engine.

        Returns:
            DAG data for frontend.
        """
        try:
            return dag_info_to_frontend_format(dag_result.get("dag_info", {}))
        except Exception as e:
            logger.error(f"Convert DAG format failed: {str(e)}")
            return {
                "nodes": [],
                "edges": []
            }
    
    def process_dag_confirmation(
        self,
        dag_confirm: str,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """Reject retired DAG controls without re-executing a trace."""
        return {"success": False, "error": "Only ReAct execution is supported. Submit a new analysis question."}
