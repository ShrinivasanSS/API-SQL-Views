"""Runtime helpers for executing AI task and workflow definitions."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, Mapping

from .config import AITaskDefinition, AIWorkflowDefinition, AppConfig, TaskInputField, TaskStep


AGENT_MANAGER_EXTENSION_KEY = "ai_agent_manager"
_CONTEXT_CHAR_LIMIT = 100_000


class AgentExecutionError(RuntimeError):
    """Raised when an AI task or workflow cannot be executed."""


class _SafeFormatDict(dict):
    """Format mapping that returns an empty string for missing keys."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive fallback
        return ""


class AgentManager:
    """Coordinate execution of configured AI tasks and workflows."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._tasks = {task.name: task for task in config.ai_tasks}
        self._workflows = {workflow.name: workflow for workflow in config.ai_workflows}

    # --------------------------
    # Listing helpers
    # --------------------------
    def list_tasks(self) -> Iterable[AITaskDefinition]:
        return self._tasks.values()

    def list_workflows(self) -> Iterable[AIWorkflowDefinition]:
        return self._workflows.values()

    def get_task(self, name: str) -> AITaskDefinition:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise AgentExecutionError(f"Unknown AI task: {name}") from exc

    def get_workflow(self, name: str) -> AIWorkflowDefinition:
        try:
            return self._workflows[name]
        except KeyError as exc:
            raise AgentExecutionError(f"Unknown AI workflow: {name}") from exc

    # --------------------------
    # Execution helpers
    # --------------------------
    def execute_task(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        instructions_override: str | None = None,
    ) -> Dict[str, Any]:
        task = self.get_task(name)
        resolved_inputs = self._resolve_inputs(task.inputs, inputs)
        steps = self._run_task_steps(task, resolved_inputs)
        instructions = (instructions_override or task.instructions or "").strip()
        return {
            "task": task.to_metadata(),
            "inputs": resolved_inputs,
            "instructions": instructions,
            "steps": steps["steps"],
            "output": steps["output"],
        }

    def execute_workflow(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        instructions_override: str | None = None,
    ) -> Dict[str, Any]:
        workflow = self.get_workflow(name)
        resolved_inputs = self._resolve_inputs(workflow.inputs, inputs)
        context: Dict[str, Any] = _SafeFormatDict({**resolved_inputs})
        workflow_steps: list[Dict[str, Any]] = []

        for index, step in enumerate(workflow.steps, start=1):
            mapped_inputs = self._render_step_inputs(step.input_map, context)
            task_result = self._run_task_steps(self.get_task(step.task), mapped_inputs)
            workflow_steps.append(
                {
                    "task": step.task,
                    "step_index": index,
                    "inputs": mapped_inputs,
                    "steps": task_result["steps"],
                    "output": task_result["output"],
                }
            )

            context.update({f"{step.task}_rows": task_result["output"]})
            first_row = (
                task_result["output"][0]
                if task_result["output"] and isinstance(task_result["output"], list)
                and isinstance(task_result["output"][0], Mapping)
                else {}
            )
            context.update({
                f"{step.task}_first_row": first_row,
                "last_rows": task_result["output"],
                "last_first_row": first_row,
            })

        summary = self._summarise_workflow(workflow, workflow_steps)
        instructions = (instructions_override or workflow.instructions or "").strip()

        return {
            "workflow": workflow.to_metadata(),
            "inputs": resolved_inputs,
            "instructions": instructions,
            "steps": workflow_steps,
            "summary": summary,
        }

    # --------------------------
    # Internal helpers
    # --------------------------
    @staticmethod
    def _resolve_inputs(
        fields: Iterable[TaskInputField], inputs: Mapping[str, Any] | None
    ) -> Dict[str, Any]:
        provided = inputs or {}
        resolved: Dict[str, Any] = {}
        for field in fields:
            value = provided.get(field.name)
            if value is None or value == "":
                value = field.default
            resolved[field.name] = "" if value is None else str(value)
        return resolved

    def _run_task_steps(
        self, task: AITaskDefinition, inputs: Mapping[str, Any]
    ) -> Dict[str, Any]:
        step_results: list[Dict[str, Any]] = []
        last_output: Any = ["No output"]
        for step in task.steps:
            result = self._execute_sql_step(step, inputs)
            step_results.append(result)
            if result.get("rows"):
                last_output = result["rows"]
            elif result.get("error"):
                last_output = ["No output"]
        return {"steps": step_results, "output": last_output}

    def _execute_sql_step(
        self, step: TaskStep, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        try:
            with sqlite3.connect(self._config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(step.sql, params)
                rows = [dict(row) for row in cursor.fetchall()]
                columns = [description[0] for description in cursor.description or []]
        except sqlite3.Error as exc:
            return {
                "title": step.title,
                "sql": step.sql,
                "columns": [],
                "rows": ["No output"],
                "rowcount": 0,
                "error": str(exc),
                "preview": "No output",
            }

        preview_text = self._truncate_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        ) if rows else "No output"

        return {
            "title": step.title,
            "sql": step.sql,
            "columns": columns,
            "rows": rows,
            "rowcount": len(rows),
            "error": None,
            "preview": preview_text,
        }

    @staticmethod
    def _render_step_inputs(
        input_map: Mapping[str, str], context: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rendered: Dict[str, Any] = {}
        safe_context = _SafeFormatDict(context)
        for key, value in input_map.items():
            rendered[key] = str(value).format_map(safe_context)
        return rendered

    @staticmethod
    def _truncate_text(value: str) -> str:
        if len(value) <= _CONTEXT_CHAR_LIMIT:
            return value
        return value[:_CONTEXT_CHAR_LIMIT] + " …[truncated]"

    def _summarise_workflow(
        self, workflow: AIWorkflowDefinition, steps: Iterable[Mapping[str, Any]]
    ) -> str:
        summary_lines = [workflow.title or workflow.name]
        for step in steps:
            task_name = step.get("task", "")
            output = step.get("output")
            if isinstance(output, list) and output and isinstance(output[0], Mapping):
                rowcount = len(output)
                summary_lines.append(f"- {task_name}: {rowcount} row(s)")
            elif isinstance(output, list):
                summary_lines.append(f"- {task_name}: {output[0] if output else 'No output'}")
            else:
                summary_lines.append(f"- {task_name}: No output")
        summary_text = "\n".join(summary_lines)
        return self._truncate_text(summary_text)


__all__ = [
    "AGENT_MANAGER_EXTENSION_KEY",
    "AgentExecutionError",
    "AgentManager",
]
