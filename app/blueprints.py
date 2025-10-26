"""Utilities for loading and expanding entity blueprints."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml


@dataclass(slots=True)
class BlueprintSource:
    """Configuration for a single data source within a pillar."""

    pillar: str
    type: str
    api: Any
    transform_rules: tuple[Path, ...]
    load_table: str
    upsert_keys: tuple[str, ...] | None
    sample_input: Path | None
    inputs: Dict[str, Any] = field(default_factory=dict)
    pagination: Dict[str, Any] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    filter_expression: str | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self, project_root: Path, *, entity_type: str) -> Dict[str, Any]:
        """Summarise the source for UI consumption."""

        def _rel(path: Path | None) -> str:
            if path is None:
                return ""
            try:
                return str(path.relative_to(project_root))
            except ValueError:
                return str(path)

        return {
            "name": f"{entity_type}:{self.pillar}:{self.type}",
            "title": f"{entity_type} {self.pillar} · {self.type}",
            "input": _rel(self.sample_input),
            "rule": _rel(self.transform_rules[0]) if self.transform_rules else "",
            "table_name": self.load_table,
            "table_type": "preset",
            "description": f"Blueprint source for {entity_type} {self.pillar}",
            "input_params": self.inputs,
            "pillar": self.pillar,
            "blueprint": entity_type,
            "load_mode": "replace",
            "upsert_keys": None,
            "source_type": self.type,
            "transform_rules": [_rel(path) for path in self.transform_rules],
            "sample_input": _rel(self.sample_input),
            "upsert_override": list(self.upsert_keys) if self.upsert_keys else None,
            "api": self.api,
            "schedule": self.schedule,
            "pagination": self.pagination,
            "filter_expression": self.filter_expression,
            "extra": self.extra,
        }


@dataclass(slots=True)
class Blueprint:
    """Parsed blueprint definition for an entity type."""

    entity_type: str
    entity_id_field: str
    defaults_upsert: Dict[str, tuple[str, ...]]
    sources: Dict[str, List[BlueprintSource]]
    views: List[Dict[str, Any]]
    path: Path
    raw: Dict[str, Any] = field(default_factory=dict)

    def iter_sources(self) -> Iterable[BlueprintSource]:
        for pillar_sources in self.sources.values():
            yield from pillar_sources

    def default_upsert_keys(self, pillar: str) -> tuple[str, ...] | None:
        return self.defaults_upsert.get(pillar)


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
            sources = [
                source.to_metadata(self.project_root, entity_type=entity_type)
                for source in blueprint.iter_sources()
            ]
            summary[entity_type] = {
                "path": self._relative(path),
                "sources": sources,
                "views": blueprint.views,
            }
        return summary

    def describe_blueprint(self, entity_type: str) -> Dict[str, Any] | None:
        blueprint = self.get(entity_type)
        if blueprint is None:
            return None

        tables: List[Dict[str, Any]] = []
        for source in blueprint.iter_sources():
            default_keys = blueprint.default_upsert_keys(source.pillar)
            tables.append(
                {
                    "id": f"{entity_type}:{source.pillar}:{source.type}",
                    "pillar": source.pillar,
                    "type": source.type,
                    "table": source.load_table,
                    "upsert_keys": list(source.upsert_keys or default_keys or ()),
                    "transform_rules": [
                        self._relative(path) for path in source.transform_rules
                    ],
                    "sample_input": self._relative(source.sample_input)
                    if source.sample_input
                    else "",
                    "api": source.api,
                    "inputs": source.inputs,
                    "pagination": source.pagination,
                    "schedule": source.schedule,
                    "filter_expression": source.filter_expression,
                    "extra": source.extra,
                }
            )

        return {
            "entity_type": blueprint.entity_type,
            "path": self._relative(blueprint.path),
            "tables": tables,
            "views": blueprint.views,
            "defaults": {
                pillar: list(keys) for pillar, keys in blueprint.defaults_upsert.items()
            },
            "yaml": blueprint.path.read_text(encoding="utf-8"),
        }

    def list_source_rules(self) -> List[Dict[str, Any]]:
        metadata: List[Dict[str, Any]] = []
        for entity_type, path in self._mapping.items():
            blueprint = self.get(entity_type)
            if blueprint is None:
                continue
            for source in blueprint.iter_sources():
                if not source.sample_input or not source.transform_rules:
                    continue
                # Use the first rule as the primary reference in the UI.
                metadata.append(source.to_metadata(self.project_root, entity_type=entity_type))
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
    def _load_blueprint(self, path: Path) -> Blueprint:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        entity_meta = data.get("entity", {}) or {}
        entity_type = (entity_meta.get("entityType") or path.stem).strip()
        entity_id_field = entity_meta.get("entityIdField", "EntityID")

        defaults = data.get("defaults", {}) or {}
        default_upsert: Dict[str, tuple[str, ...]] = {}
        for pillar, keys in (defaults.get("upsert", {}) or {}).items():
            if isinstance(keys, Sequence):
                default_upsert[pillar] = tuple(str(key) for key in keys)

        sources: Dict[str, List[BlueprintSource]] = {}
        for pillar in ("metrics", "traces", "logs", "events"):
            items = data.get(pillar) or []
            parsed: List[BlueprintSource] = []
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                parsed.append(self._parse_source(pillar, item, base_dir=path.parent))
            if parsed:
                sources[pillar] = parsed

        views = data.get("views") or []
        if not isinstance(views, list):
            views = []

        return Blueprint(
            entity_type=entity_type,
            entity_id_field=entity_id_field,
            defaults_upsert=default_upsert,
            sources=sources,
            views=views,
            path=path,
            raw=data,
        )

    def _parse_source(
        self, pillar: str, entry: Mapping[str, Any], *, base_dir: Path
    ) -> BlueprintSource:
        type_name = (entry.get("type") or pillar).strip()
        api = entry.get("api")
        transform_rules = tuple(
            self._resolve_rule_path(rule, base_dir)
            for rule in entry.get("transform_rules", [])
            if isinstance(rule, str) and rule.strip()
        )
        load = entry.get("load", {}) or {}
        table = (load.get("table") or pillar).strip()
        if not table:
            table = pillar
        table = table.lower()

        upsert_override = load.get("upsert_keys")
        upsert_keys: tuple[str, ...] | None = None
        if isinstance(upsert_override, Sequence) and not isinstance(upsert_override, (str, bytes)):
            upsert_keys = tuple(str(key) for key in upsert_override)

        sample_input = self._resolve_optional_path(entry.get("sample_input"), base_dir)

        inputs = entry.get("inputs", {}) or {}
        if not isinstance(inputs, Mapping):
            inputs = {}

        pagination = entry.get("pagination", {}) or {}
        if not isinstance(pagination, Mapping):
            pagination = {}

        schedule = entry.get("schedule", {}) or {}
        if not isinstance(schedule, Mapping):
            schedule = {}

        filter_cfg = entry.get("filter", {}) or {}
        filter_expression: str | None = None
        if isinstance(filter_cfg, Mapping):
            filter_expression = filter_cfg.get("expression")

        extra = {
            key: value
            for key, value in entry.items()
            if key
            not in {
                "type",
                "api",
                "transform_rules",
                "load",
                "sample_input",
                "inputs",
                "pagination",
                "schedule",
                "filter",
            }
        }

        return BlueprintSource(
            pillar=pillar,
            type=type_name,
            api=api,
            transform_rules=transform_rules,
            load_table=table,
            upsert_keys=upsert_keys,
            sample_input=sample_input,
            inputs=dict(inputs),
            pagination=dict(pagination),
            schedule=dict(schedule),
            filter_expression=filter_expression,
            extra=extra,
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


__all__ = ["Blueprint", "BlueprintRegistry", "BlueprintSource"]

