"""Project recorded ReAct actions into analysis tasks for the result graph."""

from copy import deepcopy


def project_react_trace(trace):
    """Fold lifecycle events into tasks without planning or executing anything.

    Exact retries of an unfinished objective share a task. Explicit artifact
    references become data edges; otherwise a dashed sequence edge records
    execution order. Control/review events stay inside the report's details.
    """
    rounds, control = {}, []
    for index, event in enumerate(trace or []):
        data = event.get("data") or {}
        action = data.get("action") or {}
        round_id = data.get("step")
        if action.get("type") == "finish" or event.get("kind") in {
            "start", "stop", "finish", "completion_review", "report_review",
        }:
            control.append(event)
            continue
        if action.get("type") in {"algorithm", "graph_query", "numeric_analysis"} and round_id is not None:
            group = rounds.setdefault(round_id, {"round": round_id, "action": action, "events": [], "order": index})
            group["events"].append(event)
        elif round_id in rounds:
            rounds[round_id]["events"].append(event)
        else:
            control.append(event)

    tasks = []
    for group in rounds.values():
        action = group["action"]
        observations = [event["data"] for event in group["events"] if event.get("kind") == "observation"]
        observation = observations[-1] if observations else {}
        status = "success" if observation.get("success") is True else "failed" if observation else "running"
        objective = str(action.get("objective") or action.get("name") or action["type"]).strip()
        key = (action["type"], action.get("name"), " ".join(objective.split()))
        if tasks and tasks[-1]["retry_key"] == key and tasks[-1]["status"] != "success":
            task = tasks[-1]
        else:
            task = {"id": str(len(tasks) + 1), "question": objective, "action_type": action["type"],
                    "name": action.get("name"), "record_type": "analysis", "attempts": [],
                    "order": group["order"], "retry_key": key}
            tasks.append(task)
        task["attempts"].append(group)
        task.update(status=status, summary=observation.get("summary", ""), error=observation.get("error"))

    artifact_producers = {}
    for task in tasks:
        events = [event for attempt in task["attempts"] for event in attempt["events"]]
        artifact_ids = list(dict.fromkeys(str(event["data"]["artifact_id"]) for event in events
                                        if event.get("data", {}).get("artifact_id")))
        for artifact_id in artifact_ids:
            artifact_producers[artifact_id] = task
        task["record"] = {
            "action_type": task["action_type"], "tool": task["name"],
            "rounds": [attempt["round"] for attempt in task["attempts"]],
            "artifact_ids": artifact_ids, "summary": task["summary"],
            "attempts": [{"round": attempt["round"], "action": attempt["action"], "events": attempt["events"]}
                         for attempt in task["attempts"]],
        }

    edges = []
    for index, task in enumerate(tasks):
        parents = set()
        for attempt in task["attempts"]:
            for reference in attempt["action"].get("consumed") or []:
                artifact_id = reference if isinstance(reference, str) else reference.get("artifact_id", reference.get("id")) if isinstance(reference, dict) else None
                parent = artifact_producers.get(str(artifact_id))
                if parent and parent["order"] < task["order"]:
                    parents.add(parent["id"])
        for parent in sorted(parents, key=int):
            edges.append({"from": parent, "to": task["id"], "kind": "data"})
        if not parents and index:
            edges.append({"from": tasks[index - 1]["id"], "to": task["id"], "kind": "sequence"})

    finishes = [event for event in control if event.get("kind") == "finish"]
    if finishes:
        finish = finishes[-1]["data"]
        report_action = next((event["data"]["action"] for event in reversed(control)
                              if event.get("data", {}).get("action", {}).get("type") == "finish"), {})
        report_id = str(len(tasks) + 1)
        outgoing = {edge["from"] for edge in edges}
        edges.extend({"from": task["id"], "to": report_id, "kind": "summary"}
                     for task in tasks if task["id"] not in outgoing)
        tasks.append({
            "id": report_id, "question": report_action.get("objective") or "汇总分析结果并生成报告",
            "action_type": "report", "name": None, "record_type": "report",
            "status": "success" if finish.get("status") == "completed" else "failed",
            "summary": finish.get("report") or "",
            "error": None if finish.get("status") == "completed" else finish.get("termination_reason"),
            "record": {"outcome": finish.get("status"), "termination_reason": finish.get("termination_reason"),
                       "events": control, "summary": finish.get("report") or ""},
        })
    for task in tasks:
        for private in ("attempts", "order", "retry_key"):
            task.pop(private, None)
    return deepcopy({"nodes": tasks, "edges": edges})
