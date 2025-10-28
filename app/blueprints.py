"""Utilities for loading and expanding entity blueprints."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import yaml

from .extractors import ExtractorDefinition


@dataclass(slots=True)
class BlueprintTable:
    """Declarative definition for a blueprint-backed table."""

    kind: str
    type: str
    table_name: str
    source_kind: str
    source_config: Dict[str, Any]
    sample_input: Path | None
    extractors: tuple[ExtractorDefinition, ...]
    transformations: tuple[Path, ...]
    inputs: Dict[str, Any]
    upsert_keys: tuple[str, ...] | None
    description: str | None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self, project_root: Path, *, entity_type: str) -> Dict[str, Any]:
        """Summarise the table for UI consumption."""

        def _relative(path: Path | None) -> str:
            if path is None:
                return ""
            try:
                return str(path.relative_to(project_root))
            except ValueError:
                return str(path)

        return {
            "name": f"{entity_type}:{self.type}:{self.table_name}",
            "title": f"{entity_type} {self.type} · {self.table_name}",
            "input": _relative(self.sample_input),
            "rule": _relative(self.transformations[0]) if self.transformations else "",
            "table_name": self.table_name,
            "table_type": self.kind,
            "description": self.description
            or f"Blueprint table for {entity_type} {self.type}",
            "input_params": self.inputs,
            "load_mode": "upsert" if self.upsert_keys else "replace",
            "upsert_keys": list(self.upsert_keys) if self.upsert_keys else None,
        }


@dataclass(slots=True)
class Blueprint:
    """Parsed blueprint definition for an entity type."""

    entity_type: str
    entity_id_field: str
    tables: tuple[BlueprintTable, ...]
    views: List[Dict[str, Any]]
    defaults_upsert: Dict[str, tuple[str, ...]]
    metadata: Dict[str, Any]
    path: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    def iter_tables(self) -> Iterable[BlueprintTable]:
        yield from self.tables

    def default_upsert_keys(self, table_type: str) -> tuple[str, ...] | None:
        return self.defaults_upsert.get(table_type)


class BlueprintRegistry:
    """Load and cache blueprints declared in the registry file."""

    def __init__(self, project_root: Path, mapping: Dict[str, Path]):
        self.project_root = project_root
        self._mapping = mapping
        self._cache: Dict[str, Blueprint] = {}

    @classmethod
    def from_csv(cls, registry_path: Path, *, project_root: Path) -> "BlueprintRegistry":
        if not registry_path.exists():
            return cls(project_root, {})

        mapping: Dict[str, Path] = {}
        with registry_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                entity_type = (row.get("EntityType") or "").strip()
                blueprint_file = (row.get("BlueprintFile") or "").strip()
                if not entity_type or not blueprint_file:
                    continue
                path = (registry_path.parent / blueprint_file).resolve()
                mapping[entity_type] = path
        return cls(project_root, mapping)

    def is_empty(self) -> bool:
        return not self._mapping

    def describe(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        for entity_type, path in self._mapping.items():
            blueprint = self.get(entity_type)
            if blueprint is None:
                continue
            tables = [
                self._table_summary(table, entity_type=entity_type)
                for table in blueprint.iter_tables()
            ]
            summary[entity_type] = {
                "path": self._relative(path),
                "tables": tables,
                "views": blueprint.views,
                "metadata": blueprint.metadata,
            }
        return summary

    def describe_blueprint(self, entity_type: str) -> Dict[str, Any] | None:
        blueprint = self.get(entity_type)
        if blueprint is None:
            return None

        tables: List[Dict[str, Any]] = []
        for table in blueprint.iter_tables():
            tables.append(self._table_summary(table, entity_type=entity_type))

        return {
            "entity_type": blueprint.entity_type,
            "entity_id_field": blueprint.entity_id_field,
            "path": self._relative(blueprint.path),
            "tables": tables,
            "views": blueprint.views,
            "defaults": {
                key: list(values) for key, values in blueprint.defaults_upsert.items()
            },
            "metadata": blueprint.metadata,
            "yaml": blueprint.path.read_text(encoding="utf-8"),
        }

    def list_source_rules(self) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for entity_type, _ in self._mapping.items():
            blueprint = self.get(entity_type)
            if blueprint is None:
                continue
            for table in blueprint.iter_tables():
                if not table.transformations or table.sample_input is None:
                    continue
                metadata.append(table.to_metadata(self.project_root, entity_type=entity_type))
        return metadata

    def get(self, entity_type: str) -> Blueprint | None:
        key = (entity_type or "").strip()
        if not key:
            return None
        path = self._mapping.get(key)
        if path is None:
            return None
        if key not in self._cache:
            self._cache[key] = self._load_blueprint(path)
        return self._cache[key]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _table_summary(
        self, table: BlueprintTable, *, entity_type: str
    ) -> Dict[str, Any]:
        sample_input = self._relative(table.sample_input) if table.sample_input else ""
        return {
            "id": f"{entity_type}:{table.type}:{table.table_name}",
            "kind": table.kind,
            "type": table.type,
            "table_name": table.table_name,
            "sample_input": sample_input,
            "source": {
                "kind": table.source_kind,
                "config": table.source_config,
            },
            "extractors": [
                {
                    "name": extractor.name,
                    "config": extractor.config,
                    "output_row_name": extractor.output_row_name,
                    "merge_strategy": extractor.merge_strategy,
                }
                for extractor in table.extractors
            ],
            "transformations": [self._relative(path) for path in table.transformations],
            "upsert_keys": list(table.upsert_keys or ()),
            "description": table.description,
            "inputs": table.inputs,
            "metadata": table.metadata,
        }

    def _load_blueprint(self, path: Path) -> Blueprint:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = int(data.get("version", 1))
        if version < 2:
            raise ValueError(
                f"Blueprint schema version {version} is no longer supported; "
                "upgrade to version 2"
            )

        entity_section = data.get("entity") or {}
        if isinstance(entity_section, Mapping):
            entity_data = entity_section
        elif isinstance(entity_section, Sequence) and entity_section:
            entity_data = entity_section[0]
        else:
            entity_data = {}

        entity_type = (entity_data.get("type") or path.stem).strip()
        entity_id_field = (entity_data.get("identifier") or "EntityID").strip()

        defaults_section = entity_data.get("defaults", {}) or {}
        defaults_upsert: Dict[str, tuple[str, ...]] = {}
        for key, values in (defaults_section.get("upsert") or {}).items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                defaults_upsert[str(key)] = tuple(str(v) for v in values)

        views = entity_data.get("views") or []
        if not isinstance(views, list):
            views = []

        metadata = entity_data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        else:
            metadata = dict(metadata)

        tables_data = entity_data.get("tables") or []
        tables: List[BlueprintTable] = []
        for entry in tables_data:
            if not isinstance(entry, Mapping):
                continue
            table = self._parse_table(entry, base_dir=path.parent)
            if table is not None:
                tables.append(table)

        return Blueprint(
            entity_type=entity_type,
            entity_id_field=entity_id_field,
            tables=tuple(tables),
            views=views,
            defaults_upsert=defaults_upsert,
            metadata=metadata,
            path=path,
            raw=data,
        )

    def _parse_table(
        self, entry: Mapping[str, Any], *, base_dir: Path
    ) -> BlueprintTable | None:
        kind = str(entry.get("kind") or "custom").strip() or "custom"
        table_type = str(entry.get("type") or kind).strip() or kind
        table_name = str(entry.get("table_name") or table_type).strip()
        if not table_name:
            table_name = table_type

        source_section = entry.get("source") or {}
        if not isinstance(source_section, Mapping):
            source_section = {}
        source_kind = str(source_section.get("kind") or "api").strip() or "api"
        source_config = source_section.get("config", {}) or {}
        if not isinstance(source_config, Mapping):
            source_config = {}
        source_config = dict(source_config)

        sample_input = self._resolve_optional_path(
            source_config.get("sample_input"), base_dir
        )

        extractors_config = entry.get("extractors") or []
        extractors: List[ExtractorDefinition] = []
        for extractor_entry in extractors_config:
            if not isinstance(extractor_entry, Mapping):
                continue
            extractor_name = (extractor_entry.get("extractor") or "").strip()
            if not extractor_name:
                continue
            config = extractor_entry.get("config", {}) or {}
            if not isinstance(config, Mapping):
                config = {}
            extractors.append(
                ExtractorDefinition(
                    name=extractor_name,
                    config=dict(config),
                    output_row_name=extractor_entry.get("output_row_name"),
                    merge_strategy=extractor_entry.get("merge_strategy"),
                )
            )

        transformations = tuple(
            self._resolve_rule_path(path, base_dir)
            for path in entry.get("transformations", [])
            if isinstance(path, str) and path.strip()
        )

        inputs = entry.get("inputs", {}) or {}
        if not isinstance(inputs, Mapping):
            inputs = {}
        else:
            inputs = dict(inputs)

        metadata = entry.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        metadata = dict(metadata)

        upsert_keys = entry.get("upsert_keys")
        if upsert_keys is None:
            upsert_keys = metadata.get("upsert_keys")
        parsed_upsert: tuple[str, ...] | None = None
        if isinstance(upsert_keys, Sequence) and not isinstance(upsert_keys, (str, bytes, bytearray)):
            parsed_upsert = tuple(str(key) for key in upsert_keys)
            metadata.pop("upsert_keys", None)

        description = entry.get("description") or metadata.get("description")
        if isinstance(description, str):
            metadata.pop("description", None)
        else:
            description = None

        return BlueprintTable(
            kind=kind,
            type=table_type,
            table_name=table_name,
            source_kind=source_kind,
            source_config=source_config,
            sample_input=sample_input,
            extractors=tuple(extractors),
            transformations=transformations,
            inputs=inputs,
            upsert_keys=parsed_upsert,
            description=description,
            metadata=metadata,
        )

    def _resolve_rule_path(self, rule: str, base_dir: Path) -> Path:
        path = Path(rule)
        candidates = self._candidate_paths(path, base_dir, subdir="sample_transformation_rules")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Cannot resolve transform rule path: {rule}")

    def _resolve_optional_path(self, value: Any, base_dir: Path) -> Path | None:
        if not value:
            return None
        path = Path(str(value))
        candidates = self._candidate_paths(path, base_dir, subdir="sample_inputs")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _candidate_paths(self, path: Path, base_dir: Path, *, subdir: str) -> Iterable[Path]:
        if path.is_absolute():
            yield path
            return

        yield (base_dir / path).resolve()
        yield (self.project_root / path).resolve()
        yield (self.project_root / subdir / path).resolve()

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


__all__ = ["Blueprint", "BlueprintRegistry", "BlueprintTable"]

