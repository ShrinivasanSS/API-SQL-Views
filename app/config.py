"""Configuration helpers for the Observability ETL demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from dotenv import load_dotenv
import os
import yaml

from .blueprints import BlueprintRegistry


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_storage_dir() -> Path:
    """Return the directory used to persist generated artefacts."""

    storage = PROJECT_ROOT / "storage"
    storage.mkdir(exist_ok=True)
    return storage


@dataclass(slots=True)
class PipelineRule:
    """Describe a single transformation from an input payload to a table."""

    name: str
    title: str
    input_path: Path
    rule_path: Path
    table_name: str
    table_type: str
    description: str
    input_params: Dict[str, Any] = field(default_factory=dict)
    load_mode: str = "replace"
    upsert_keys: tuple[str, ...] | None = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "input": str(self.input_path.relative_to(PROJECT_ROOT)),
            "rule": str(self.rule_path.relative_to(PROJECT_ROOT)),
            "table_name": self.table_name,
            "table_type": self.table_type,
            "description": self.description,
            "input_params": self.input_params,
            "load_mode": self.load_mode,
            "upsert_keys": list(self.upsert_keys) if self.upsert_keys else None,
        }


@dataclass(slots=True)
class TaskInputField:
    """Describe a user-editable input passed to an AI task."""

    name: str
    label: str
    description: str | None = None
    placeholder: str | None = None
    default: str | None = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "placeholder": self.placeholder,
            "default": self.default,
        }


@dataclass(slots=True)
class TaskStep:
    """Single SQL step executed as part of a task."""

    title: str
    sql: str

    def to_metadata(self) -> Dict[str, Any]:
        return {"title": self.title, "sql": self.sql}


@dataclass(slots=True)
class AITaskDefinition:
    """Runtime representation of a configured AI task."""

    name: str
    title: str
    description: str
    instructions: str
    inputs: tuple[TaskInputField, ...]
    steps: tuple[TaskStep, ...]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "instructions": self.instructions,
            "inputs": [field.to_metadata() for field in self.inputs],
            "steps": [step.to_metadata() for step in self.steps],
        }


@dataclass(slots=True)
class WorkflowStep:
    """Invocation of an AI task inside a workflow."""

    task: str
    input_map: Dict[str, str]

    def to_metadata(self) -> Dict[str, Any]:
        return {"task": self.task, "input_map": dict(self.input_map)}


@dataclass(slots=True)
class AIWorkflowDefinition:
    """Runtime representation of an AI workflow that chains tasks."""

    name: str
    title: str
    description: str
    instructions: str
    inputs: tuple[TaskInputField, ...]
    steps: tuple[WorkflowStep, ...]

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "instructions": self.instructions,
            "inputs": [field.to_metadata() for field in self.inputs],
            "steps": [step.to_metadata() for step in self.steps],
        }


@dataclass(slots=True)
class AzureOpenAIConfig:
    """Configuration required to access an Azure OpenAI deployment."""

    endpoint: str
    api_key: str
    deployment: str
    api_version: str
    assistant_id: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "api_key": self.api_key,
            "deployment": self.deployment,
            "api_version": self.api_version,
        }
        if self.assistant_id:
            payload["assistant_id"] = self.assistant_id
        return payload


@dataclass(slots=True)
class AppConfig:
    """Runtime configuration resolved from environment variables."""

    database_path: Path
    storage_dir: Path
    cache_dir: Path
    pipeline_rules: tuple[PipelineRule, ...]
    credentials: Dict[str, str]
    blueprint_registry: BlueprintRegistry
    azure_openai: AzureOpenAIConfig | None = None
    ai_tasks: tuple[AITaskDefinition, ...] = field(default_factory=tuple)
    ai_workflows: tuple[AIWorkflowDefinition, ...] = field(default_factory=tuple)

    @classmethod
    def load_from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        storage = _default_storage_dir()
        database_path = storage / "observability.db"
        cache_dir = (PROJECT_ROOT / ".cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Demo credentials – defaults are safe placeholders that can be
        # overridden locally without affecting version control.
        credentials = {
            "client_id": os.getenv("DEMO_CLIENT_ID", "demo-client"),
            "client_secret": os.getenv("DEMO_CLIENT_SECRET", "demo-secret"),
            "refresh_token": os.getenv("DEMO_REFRESH_TOKEN", "demo-refresh-token"),
            "api_domain": os.getenv("DEMO_API_DOMAIN", "https://api.example.com"),
        }

        rules = cls._load_default_rules()
        blueprints = BlueprintRegistry.from_csv(
            PROJECT_ROOT / "blueprints" / "blueprints.csv",
            project_root=PROJECT_ROOT,
        )

        tasks, workflows = cls._load_ai_definitions()

        azure_openai = None
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        assistant_id = os.getenv("AZURE_OPENAI_ASSISTANT_ID")
        if all([endpoint, api_key, deployment, api_version]):
            azure_openai = AzureOpenAIConfig(
                endpoint=endpoint,
                api_key=api_key,
                deployment=deployment,
                api_version=api_version,
                assistant_id=assistant_id or None,
            )

        return cls(
            database_path=database_path,
            storage_dir=storage,
            cache_dir=cache_dir,
            pipeline_rules=rules,
            credentials=credentials,
            blueprint_registry=blueprints,
            azure_openai=azure_openai,
            ai_tasks=tasks,
            ai_workflows=workflows,
        )

    def as_flask_config(self) -> Dict[str, Any]:
        base_rules = [rule.to_metadata() for rule in self.pipeline_rules]
        blueprint_rules = self.blueprint_registry.list_source_rules()
        config: Dict[str, Any] = {
            "DATABASE_PATH": str(self.database_path),
            "PIPELINE_RULES": [*base_rules, *blueprint_rules],
            "CREDENTIALS": self.credentials,
            "BLUEPRINTS": self.blueprint_registry.describe(),
            "CACHE_DIR": str(self.cache_dir),
        }
        if self.azure_openai:
            config["AZURE_OPENAI"] = self.azure_openai.as_dict()
        if self.ai_tasks:
            config["AI_TASKS"] = [task.to_metadata() for task in self.ai_tasks]
        if self.ai_workflows:
            config["AI_WORKFLOWS"] = [wf.to_metadata() for wf in self.ai_workflows]
        return config

    @staticmethod
    def _load_default_rules() -> tuple[PipelineRule, ...]:
        """Construct the default pipeline configuration."""

        return (
            PipelineRule(
                name="entities",
                title="Entities",
                input_path=PROJECT_ROOT
                / "sample_inputs"
                / "entities"
                / "api_current_status.json",
                rule_path=PROJECT_ROOT
                / "sample_transformation_rules"
                / "entitylist_transformation.rules",
                table_name="entities",
                table_type="preset",
                description="Current inventory of monitors extracted from the API",
            ),
        )

    @staticmethod
    def _load_ai_definitions() -> tuple[
        tuple[AITaskDefinition, ...], tuple[AIWorkflowDefinition, ...]
    ]:
        seed_path = PROJECT_ROOT / "seed.yaml"
        if not seed_path.exists():
            return tuple(), tuple()

        try:
            raw_seed = yaml.safe_load(seed_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            return tuple(), tuple()

        if not isinstance(raw_seed, Mapping):
            return tuple(), tuple()

        tasks = AppConfig._parse_ai_tasks(raw_seed.get("ai_tasks"))
        workflows = AppConfig._parse_ai_workflows(
            raw_seed.get("ai_workflows"), tasks
        )
        return tasks, workflows

    @staticmethod
    def _parse_ai_tasks(data: Any) -> tuple[AITaskDefinition, ...]:
        if not isinstance(data, list):
            return tuple()

        tasks: list[AITaskDefinition] = []
        for entry in data:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or "").strip()
            title = str(entry.get("title") or name or "").strip() or name
            description = str(entry.get("description") or "").strip()
            instructions = str(entry.get("instructions") or "").strip()
            raw_inputs = entry.get("inputs")
            raw_steps = entry.get("steps")
            if not name or not isinstance(raw_inputs, list) or not isinstance(
                raw_steps, list
            ):
                continue

            inputs: list[TaskInputField] = []
            for item in raw_inputs:
                if not isinstance(item, Mapping):
                    continue
                input_name = str(item.get("name") or "").strip()
                if not input_name:
                    continue
                label = str(item.get("label") or input_name).strip() or input_name
                placeholder = (
                    str(item.get("placeholder")).strip()
                    if item.get("placeholder") is not None
                    else None
                )
                description_value = (
                    str(item.get("description")).strip()
                    if item.get("description") is not None
                    else None
                )
                default_value = (
                    str(item.get("default")).strip()
                    if item.get("default") is not None
                    else None
                )
                inputs.append(
                    TaskInputField(
                        name=input_name,
                        label=label,
                        description=description_value or None,
                        placeholder=placeholder or None,
                        default=default_value or None,
                    )
                )

            steps: list[TaskStep] = []
            for step in raw_steps:
                if not isinstance(step, Mapping):
                    continue
                title_value = str(step.get("title") or "Step").strip() or "Step"
                sql_value = str(step.get("sql") or "").strip()
                if not sql_value:
                    continue
                steps.append(TaskStep(title=title_value, sql=sql_value))

            if not steps:
                continue

            tasks.append(
                AITaskDefinition(
                    name=name,
                    title=title,
                    description=description,
                    instructions=instructions,
                    inputs=tuple(inputs),
                    steps=tuple(steps),
                )
            )

        return tuple(tasks)

    @staticmethod
    def _parse_ai_workflows(
        data: Any, tasks: tuple[AITaskDefinition, ...]
    ) -> tuple[AIWorkflowDefinition, ...]:
        if not isinstance(data, list):
            return tuple()

        task_names = {task.name for task in tasks}
        workflows: list[AIWorkflowDefinition] = []
        for entry in data:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or "").strip()
            title = str(entry.get("title") or name or "").strip() or name
            description = str(entry.get("description") or "").strip()
            instructions = str(entry.get("instructions") or "").strip()
            raw_inputs = entry.get("inputs")
            raw_steps = entry.get("steps")
            if not name or not isinstance(raw_steps, list):
                continue

            inputs: list[TaskInputField] = []
            if isinstance(raw_inputs, list):
                for item in raw_inputs:
                    if not isinstance(item, Mapping):
                        continue
                    input_name = str(item.get("name") or "").strip()
                    if not input_name:
                        continue
                    label = str(item.get("label") or input_name).strip() or input_name
                    placeholder = (
                        str(item.get("placeholder")).strip()
                        if item.get("placeholder") is not None
                        else None
                    )
                    description_value = (
                        str(item.get("description")).strip()
                        if item.get("description") is not None
                        else None
                    )
                    default_value = (
                        str(item.get("default")).strip()
                        if item.get("default") is not None
                        else None
                    )
                    inputs.append(
                        TaskInputField(
                            name=input_name,
                            label=label,
                            description=description_value or None,
                            placeholder=placeholder or None,
                            default=default_value or None,
                        )
                    )

            steps: list[WorkflowStep] = []
            for step in raw_steps:
                if not isinstance(step, Mapping):
                    continue
                task_name = str(step.get("task") or "").strip()
                if not task_name or task_name not in task_names:
                    continue
                input_map_raw = step.get("input_map")
                input_map: Dict[str, str] = {}
                if isinstance(input_map_raw, Mapping):
                    for key, value in input_map_raw.items():
                        if not key:
                            continue
                        input_map[str(key)] = str(value) if value is not None else ""
                steps.append(WorkflowStep(task=task_name, input_map=input_map))

            if not steps:
                continue

            workflows.append(
                AIWorkflowDefinition(
                    name=name,
                    title=title,
                    description=description,
                    instructions=instructions,
                    inputs=tuple(inputs),
                    steps=tuple(steps),
                )
            )

        return tuple(workflows)


__all__ = [
    "AITaskDefinition",
    "AIWorkflowDefinition",
    "AppConfig",
    "AzureOpenAIConfig",
    "PipelineRule",
    "PROJECT_ROOT",
    "TaskInputField",
    "TaskStep",
    "WorkflowStep",
]
