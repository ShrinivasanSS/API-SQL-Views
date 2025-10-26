"""Configuration helpers for the Observability ETL demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import os

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
class AppConfig:
    """Runtime configuration resolved from environment variables."""

    database_path: Path
    storage_dir: Path
    pipeline_rules: tuple[PipelineRule, ...]
    credentials: Dict[str, str]
    blueprint_registry: BlueprintRegistry

    @classmethod
    def load_from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        storage = _default_storage_dir()
        database_path = storage / "observability.db"

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

        return cls(
            database_path=database_path,
            storage_dir=storage,
            pipeline_rules=rules,
            credentials=credentials,
            blueprint_registry=blueprints,
        )

    def as_flask_config(self) -> Dict[str, Any]:
        base_rules = [rule.to_metadata() for rule in self.pipeline_rules]
        blueprint_rules = self.blueprint_registry.list_source_rules()
        return {
            "DATABASE_PATH": str(self.database_path),
            "PIPELINE_RULES": [*base_rules, *blueprint_rules],
            "CREDENTIALS": self.credentials,
            "BLUEPRINTS": self.blueprint_registry.describe(),
        }

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


__all__ = ["AppConfig", "PipelineRule", "PROJECT_ROOT"]
