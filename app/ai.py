"""Runtime helpers for executing AI task and workflow definitions."""

from __future__ import annotations

import json
import sqlite3
import re
from typing import Any, Dict, Iterable, Mapping

from .assistants import AssistantService
from .config import (
    AITaskDefinition,
    AIWorkflowDefinition,
    AppConfig,
    TaskInputField,
    TaskStep,
)


AGENT_MANAGER_EXTENSION_KEY = "ai_agent_manager"
_CONTEXT_CHAR_LIMIT = 100_000
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_TABLE_INPUTS = {
    "events_table": "events",
    "logs_table": "logs",
    "metrics_table": "metrics",
    "traces_table": "traces",
}


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
        mode: str | None = None,
        assistant_service: AssistantService | None = None,
    ) -> Dict[str, Any]:
        task = self.get_task(name)
        resolved_inputs = self._resolve_inputs(task.inputs, inputs)
        instructions = (instructions_override or task.instructions or "").strip()
        execution = self._execute_task_definition(
            task,
            resolved_inputs,
            instructions,
            mode,
            assistant_service,
        )
        return {
            "task": task.to_metadata(),
            "inputs": resolved_inputs,
            "instructions": instructions,
            **execution,
        }

    def execute_workflow(
        self,
        name: str,
        *,
        inputs: Mapping[str, Any] | None = None,
        instructions_override: str | None = None,
        mode: str | None = None,
        assistant_service: AssistantService | None = None,
    ) -> Dict[str, Any]:
        workflow = self.get_workflow(name)
        resolved_inputs = self._resolve_inputs(workflow.inputs, inputs)
        context: Dict[str, Any] = _SafeFormatDict({**resolved_inputs})
        self._inject_default_tables(context)
        workflow_steps: list[Dict[str, Any]] = []
        normalised_mode = self._normalise_mode(mode)

        for index, step in enumerate(workflow.steps, start=1):
            mapped_inputs = self._render_step_inputs(step.input_map, context)
            task_definition = self.get_task(step.task)
            task_result = self._execute_task_definition(
                task_definition,
                mapped_inputs,
                (task_definition.instructions or "").strip(),
                normalised_mode,
                assistant_service,
            )
            workflow_steps.append(
                {
                    "task": step.task,
                    "step_index": index,
                    "inputs": mapped_inputs,
                    "steps": task_result["steps"],
                    "output": task_result["output"],
                    "summary": task_result.get("summary"),
                    "mode": task_result.get("mode", normalised_mode),
                    "ai": task_result.get("ai"),
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
            self._apply_entity_context(step.task, task_result["output"], context)

        if normalised_mode == "ai":
            if assistant_service is None:
                raise AgentExecutionError("AI mode requires an assistant service")
            summary_payload = assistant_service.summarise_workflow(
                workflow_name=workflow.title or workflow.name,
                instructions=(instructions_override or workflow.instructions or "").strip(),
                steps=workflow_steps,
            )
            summary = summary_payload.get("summary", "")
            ai_metadata = {"summary_metadata": summary_payload.get("metadata"), "fallback_messages": self._collect_workflow_fallbacks(workflow_steps)}
        else:
            summary = self._summarise_workflow(workflow, workflow_steps)
            ai_metadata = None
        instructions = (instructions_override or workflow.instructions or "").strip()

        result: Dict[str, Any] = {
            "workflow": workflow.to_metadata(),
            "inputs": resolved_inputs,
            "instructions": instructions,
            "steps": workflow_steps,
            "summary": summary,
            "mode": normalised_mode,
        }
        if ai_metadata:
            result["ai"] = ai_metadata
        return result

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

    def _execute_task_definition(
        self,
        task: AITaskDefinition,
        inputs: Mapping[str, Any],
        instructions: str,
        mode: str | None,
        assistant_service: AssistantService | None,
    ) -> Dict[str, Any]:
        normalised_mode = self._normalise_mode(mode)
        if normalised_mode == "ai":
            if assistant_service is None:
                raise AgentExecutionError("AI mode requires an assistant service")
            return self._run_task_ai_mode(task, inputs, instructions, assistant_service)
        result = self._run_task_steps(task, inputs)
        result["mode"] = "query"
        return result

    @staticmethod
    def _normalise_mode(mode: str | None) -> str:
        value = (mode or "query").lower()
        if value not in {"query", "ai"}:
            raise AgentExecutionError(f"Unsupported execution mode: {mode}")
        return value

    def _run_task_steps(
        self, task: AITaskDefinition, inputs: Mapping[str, Any]
    ) -> Dict[str, Any]:
        step_results: list[Dict[str, Any]] = []
        last_output: Any = ["No output"]
        for step in task.steps:
            result = self._execute_sql_step(step, inputs)
            result["mode"] = "query"
            result["summary"] = self._summarise_rows(result.get("rows"))
            step_results.append(result)
            if result.get("rows"):
                last_output = result["rows"]
            elif result.get("error"):
                last_output = ["No output"]
        return {
            "steps": step_results,
            "output": last_output,
            "summary": self._summarise_rows(last_output),
        }

    def _run_task_ai_mode(
        self,
        task: AITaskDefinition,
        inputs: Mapping[str, Any],
        instructions: str,
        assistant_service: AssistantService,
    ) -> Dict[str, Any]:
        step_results: list[Dict[str, Any]] = []
        last_output: Any = ["No output"]
        fallback_messages: list[str] = []
        for step in task.steps:
            try:
                default_sql = self._render_sql(step.sql, inputs)
            except AgentExecutionError:
                default_sql = step.sql

            generation = assistant_service.generate_task_sql(
                task_name=task.title or task.name,
                step_title=step.title or step.sql[:30],
                instructions=instructions,
                inputs=dict(inputs),
                default_sql=default_sql,
            )
            generated_sql = (generation.get("sql") or "").strip()
            notes = generation.get("notes") or ""
            step_result: Dict[str, Any]
            fallback_reason: str | None = None

            if generated_sql:
                step_result = self._execute_sql_step(
                    step,
                    inputs,
                    sql_override=generated_sql,
                    assume_rendered=True,
                )
                executed_rows = step_result.get("rows") or []
                if step_result.get("error") or not executed_rows:
                    fallback_reason = step_result.get("error") or "AI query returned no rows"
                    fallback_messages.append(
                        f"{step.title or step.sql[:30]}: {fallback_reason}. Default query executed."
                    )
                    step_result = self._execute_sql_step(step, inputs)
                else:
                    step_result["sql"] = generated_sql
            else:
                fallback_reason = "Assistant did not provide SQL; default query executed"
                fallback_messages.append(f"{step.title or step.sql[:30]}: {fallback_reason}.")
                step_result = self._execute_sql_step(step, inputs)

            if step_result.get("rows"):
                last_output = step_result["rows"]
            elif step_result.get("error"):
                last_output = ["No output"]

            step_result["mode"] = "ai"
            step_result["summary"] = self._summarise_rows(step_result.get("rows"))
            step_result["generated_sql"] = generated_sql or None
            step_result["default_sql"] = default_sql
            step_result["ai"] = {
                "generated_sql": generated_sql or None,
                "notes": notes or None,
                "default_sql": default_sql,
                "used_fallback": bool(fallback_reason),
                "fallback_reason": fallback_reason,
            }
            step_results.append(step_result)

        rowcount = len(last_output) if isinstance(last_output, list) else 0
        summary_payload = assistant_service.summarise_task_result(
            task_name=task.title or task.name,
            instructions=instructions,
            inputs=dict(inputs),
            rows=last_output if isinstance(last_output, list) else [],
            rowcount=rowcount,
        )
        summary_text = summary_payload.get("summary", "")

        return {
            "steps": step_results,
            "output": last_output,
            "summary": summary_text,
            "mode": "ai",
            "ai": {
                "summary_metadata": summary_payload.get("metadata"),
                "fallback_messages": fallback_messages,
            },
        }

    def _execute_sql_step(
        self,
        step: TaskStep,
        params: Mapping[str, Any],
        *,
        sql_override: str | None = None,
        assume_rendered: bool = False,
    ) -> Dict[str, Any]:
        template_sql = step.sql
        try:
            if sql_override is not None:
                if assume_rendered:
                    sql = sql_override
                else:
                    sql = self._render_sql(sql_override, params)
            else:
                sql = self._render_sql(step.sql, params)
        except AgentExecutionError as exc:
            return {
                "title": step.title,
                "sql": template_sql,
                "template_sql": template_sql,
                "columns": [],
                "rows": ["No output"],
                "rowcount": 0,
                "error": str(exc),
                "preview": "No output",
            }
        try:
            with sqlite3.connect(self._config.database_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = [dict(row) for row in cursor.fetchall()]
                columns = [description[0] for description in cursor.description or []]
        except sqlite3.Error as exc:
            return {
                "title": step.title,
                "sql": sql,
                "template_sql": template_sql,
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
            "sql": sql,
            "template_sql": template_sql,
            "columns": columns,
            "rows": rows,
            "rowcount": len(rows),
            "error": None,
            "preview": preview_text,
        }

    @staticmethod
    def _summarise_rows(rows: Any) -> str:
        if isinstance(rows, list):
            if not rows:
                return "No rows returned"
            first = rows[0]
            if isinstance(first, Mapping):
                return f"{len(rows)} row(s)"
            return ", ".join(str(item) for item in rows[:3])
        return "No output"

    @staticmethod
    def _collect_workflow_fallbacks(steps: Iterable[Mapping[str, Any]]) -> list[str]:
        messages: list[str] = []
        for step in steps:
            ai_metadata = step.get("ai") if isinstance(step, Mapping) else None
            if not isinstance(ai_metadata, Mapping):
                continue
            if ai_metadata.get("used_fallback") and ai_metadata.get("fallback_reason"):
                task_name = step.get("task") if isinstance(step, Mapping) else None
                if task_name:
                    messages.append(f"{task_name}: {ai_metadata['fallback_reason']}")
                else:
                    messages.append(str(ai_metadata["fallback_reason"]))
        return messages

    @staticmethod
    def _render_step_inputs(
        input_map: Mapping[str, str], context: Mapping[str, Any]
    ) -> Dict[str, Any]:
        rendered: Dict[str, Any] = {}
        safe_context = _SafeFormatDict(context)
        for key, value in input_map.items():
            rendered[key] = str(value).format_map(safe_context)
        return rendered

    def _inject_default_tables(self, context: Dict[str, Any]) -> None:
        for key, value in _DEFAULT_TABLE_INPUTS.items():
            context.setdefault(key, value)

    def _apply_entity_context(
        self, task_name: str, output: Any, context: Dict[str, Any]
    ) -> None:
        if task_name not in {"findEntities", "fetchEntities"}:
            return
        if not isinstance(output, list) or not output:
            return
        first_row = output[0]
        if not isinstance(first_row, Mapping):
            return
        entity_type = str(first_row.get("EntityType") or "").strip()
        entity_id = str(first_row.get("EntityID") or "").strip()
        if entity_type:
            context["entity_type"] = entity_type
            context.update(self._resolve_blueprint_tables(entity_type))
        if entity_id and not context.get("entity_id"):
            context["entity_id"] = entity_id

    def _resolve_blueprint_tables(self, entity_type: str | None) -> Dict[str, str]:
        tables = dict(_DEFAULT_TABLE_INPUTS)
        if not entity_type:
            return tables
        blueprint = self._config.blueprint_registry.get(entity_type)
        if blueprint is None:
            return tables
        for table in blueprint.iter_tables():
            key = f"{table.type}_table"
            name = getattr(table, "table_name", "")
            if key in tables and name:
                tables[key] = str(name)
        return tables

    def _render_sql(self, template: str, params: Mapping[str, Any]) -> str:
        if "{" not in template or "}" not in template:
            return template
        placeholders = set(re.findall(r"{([A-Za-z_][A-Za-z0-9_]*)}", template))
        if not placeholders:
            return template
        values: Dict[str, str] = {}
        for key in placeholders:
            raw = params.get(key)
            if raw is None or str(raw).strip() == "":
                raise AgentExecutionError(
                    f"Task input '{key}' must be provided to render the SQL query"
                )
            values[key] = self._sanitize_identifier(raw)
        try:
            return template.format(**values)
        except KeyError as exc:  # pragma: no cover - defensive
            missing = exc.args[0] if exc.args else "unknown"
            raise AgentExecutionError(
                f"Missing template value for '{missing}'"
            ) from exc

    @staticmethod
    def _sanitize_identifier(value: Any) -> str:
        identifier = str(value).strip()
        if not identifier:
            raise AgentExecutionError("Table identifier cannot be empty")
        if not _IDENTIFIER_PATTERN.match(identifier):
            raise AgentExecutionError(
                "Table identifier must contain only letters, numbers, or underscores"
            )
        return identifier

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
