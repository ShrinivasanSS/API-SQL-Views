"""ETL pipeline helpers for running transformations and loading SQLite."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import json

from .config import AppConfig, PipelineRule
from .extractors import apply_extractors


def load_json_payload(path: Path) -> pd.DataFrame:
    """Load an arbitrary JSON file into a single-row pandas DataFrame."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([data])


def execute_rule(
    df_input: pd.DataFrame,
    rule: PipelineRule,
    *,
    extra_globals: Dict[str, Any] | None = None,
    rule_override: str | None = None,
) -> pd.DataFrame:
    """Execute a transformation rule and return the resulting DataFrame."""

    env: Dict[str, Any] = {
        "pd": pd,
        "df_input": df_input,
        "INPUT_PARAMS": rule.input_params,
        "__builtins__": __builtins__,
    }
    if extra_globals:
        env.update(extra_globals)

    rule_source = (
        rule_override
        if rule_override is not None
        else rule.rule_path.read_text(encoding="utf-8")
    )
    exec(rule_source, env)

    if "df_output" not in env:
        raise RuntimeError(f"Rule {rule.rule_path.name} did not define df_output")

    df_output = env["df_output"]
    if not isinstance(df_output, pd.DataFrame):
        raise TypeError(
            f"Rule {rule.rule_path.name} produced {type(df_output)!r} instead of pandas.DataFrame"
        )

    return df_output


def load_dataframe_to_sqlite(
    df: pd.DataFrame,
    *,
    table_name: str,
    database_path: Path,
    load_mode: str = "replace",
    upsert_keys: tuple[str, ...] | None = None,
) -> None:
    """Load a DataFrame into SQLite honouring the requested semantics."""

    mode = (load_mode or "replace").lower()
    database_path.parent.mkdir(exist_ok=True)

    with sqlite3.connect(database_path) as conn:
        if mode == "replace":
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            return

        if mode == "append":
            df.to_sql(table_name, conn, if_exists="append", index=False)
            return

        if mode == "upsert":
            if upsert_keys is None:
                df.to_sql(table_name, conn, if_exists="append", index=False)
                return

            existing = None
            try:
                existing = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            except Exception:
                # Table does not exist yet – behave like replace
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                return

            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=list(upsert_keys), keep="last"
            )
            combined.to_sql(table_name, conn, if_exists="replace", index=False)
            return

        raise ValueError(f"Unsupported load mode: {load_mode}")


def run_pipeline_rule(
    rule: PipelineRule,
    database_path: Path,
    *,
    extra_globals: Dict[str, Any] | None = None,
    rule_override: str | None = None,
) -> Dict[str, Any]:
    """Execute a single rule end-to-end and store the result."""

    df_input = load_json_payload(rule.input_path)
    df_output = execute_rule(
        df_input, rule, extra_globals=extra_globals, rule_override=rule_override
    )
    load_dataframe_to_sqlite(
        df_output,
        table_name=rule.table_name,
        database_path=database_path,
        load_mode=rule.load_mode,
        upsert_keys=rule.upsert_keys,
    )

    csv_buffer = io.StringIO()
    df_output.to_csv(csv_buffer, index=False)

    return {
        "rule": rule.to_metadata(),
        "rowcount": len(df_output.index),
        "columns": list(df_output.columns),
        "preview_csv": csv_buffer.getvalue(),
    }


def run_full_pipeline(config: AppConfig) -> List[Dict[str, Any]]:
    """Execute all configured rules in sequence."""

    results = []
    for rule in config.pipeline_rules:
        results.append(run_pipeline_rule(rule, config.database_path))
    results.extend(run_blueprint_pipeline(config))
    return results


def run_blueprint_pipeline(config: AppConfig) -> List[Dict[str, Any]]:
    """Execute blueprint-driven sources for every known entity."""

    registry = config.blueprint_registry
    if registry.is_empty():
        return []

    try:
        with sqlite3.connect(config.database_path) as conn:
            entities_df = pd.read_sql_query("SELECT * FROM entities", conn)
    except Exception:
        return []

    if entities_df.empty:
        return []

    results: List[Dict[str, Any]] = []
    project_root = registry.project_root
    for entity in entities_df.to_dict(orient="records"):
        entity_type = (entity.get("EntityType") or "").strip()
        if not entity_type:
            continue

        blueprint = registry.get(entity_type)
        if blueprint is None:
            continue

        entity_id = entity.get("EntityID")
        entity_name = entity.get("EntityName", "")

        for table in blueprint.iter_tables():
            if table.sample_input is None:
                continue

            params = {**table.inputs}
            params.setdefault("entity_type", entity_type)
            params.setdefault("entity_id", entity_id)
            params.setdefault("entity_name", entity_name)
            params.setdefault("blueprint_entity_type", blueprint.entity_type)
            params.setdefault("blueprint_table_type", table.type)
            params.setdefault("blueprint_table_name", table.table_name)

            context_globals = {
                "ENTITY_ROW": entity,
                "BLUEPRINT_TABLE": {
                    "kind": table.kind,
                    "type": table.type,
                    "table_name": table.table_name,
                    "source_kind": table.source_kind,
                    "source_config": table.source_config,
                },
                "BLUEPRINT_SOURCE": {
                    "pillar": table.type,
                    "type": table.type,
                    "api": table.source_config.get("path"),
                    "pagination": table.source_config.get("pagination"),
                    "schedule": table.source_config.get("schedule"),
                    "filter_expression": table.source_config.get("filter"),
                },
            }

            upsert_keys = table.upsert_keys or blueprint.default_upsert_keys(table.type)
            load_mode = "upsert" if upsert_keys else "replace"

            if table.extractors:
                payload = json.loads(table.sample_input.read_text(encoding="utf-8"))
                extractor_context = {
                    "params": params,
                    "entity": entity,
                    "table": {
                        "kind": table.kind,
                        "type": table.type,
                        "table_name": table.table_name,
                    },
                }
                df_output = apply_extractors(
                    payload, table.extractors, context=extractor_context
                )

                if table.transformations:
                    for transform_path in table.transformations:
                        rule = PipelineRule(
                            name=f"{table.type}:{table.table_name}:{entity_id}:{transform_path.stem}",
                            title=f"{entity_type} {table.type} · {table.table_name}",
                            input_path=table.sample_input,
                            rule_path=transform_path,
                            table_name=table.table_name,
                            table_type=table.kind,
                            description=table.description
                            or f"Blueprint table {table.table_name}",
                            input_params=dict(params),
                            load_mode=load_mode,
                            upsert_keys=upsert_keys,
                        )
                        df_output = execute_rule(
                            df_output, rule, extra_globals=context_globals
                        )

                load_dataframe_to_sqlite(
                    df_output,
                    table_name=table.table_name,
                    database_path=config.database_path,
                    load_mode=load_mode,
                    upsert_keys=upsert_keys,
                )

                csv_buffer = io.StringIO()
                df_output.to_csv(csv_buffer, index=False)

                rule_metadata = table.to_metadata(project_root, entity_type=entity_type)
                extractor_path = project_root / "app" / "extractors.py"
                rule_metadata["rule"] = str(extractor_path.relative_to(project_root))
                rule_metadata["input_params"] = dict(params)
                rule_metadata["load_mode"] = load_mode
                rule_metadata["upsert_keys"] = (
                    list(upsert_keys) if upsert_keys else None
                )

                results.append(
                    {
                        "rule": rule_metadata,
                        "rowcount": len(df_output.index),
                        "columns": list(df_output.columns),
                        "preview_csv": csv_buffer.getvalue(),
                    }
                )
                continue

            if not table.transformations:
                continue

            for transform_path in table.transformations:
                rule = PipelineRule(
                    name=f"{table.type}:{table.table_name}:{entity_id}:{transform_path.stem}",
                    title=f"{entity_type} {table.type} · {table.table_name}",
                    input_path=table.sample_input,
                    rule_path=transform_path,
                    table_name=table.table_name,
                    table_type=table.kind,
                    description=table.description
                    or f"Blueprint table {table.table_name}",
                    input_params=dict(params),
                    load_mode=load_mode,
                    upsert_keys=upsert_keys,
                )

                results.append(
                    run_pipeline_rule(
                        rule,
                        config.database_path,
                        extra_globals=context_globals,
                    )
                )

    return results


def ensure_database_seeded(config: AppConfig) -> None:
    """Ensure the SQLite database contains freshly transformed tables."""

    run_full_pipeline(config)


def fetch_table_list(database_path: Path) -> List[Dict[str, Any]]:
    """Return table metadata for all user tables in the SQLite database."""

    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor]

    metadata = []
    for table_name in tables:
        with sqlite3.connect(database_path) as conn:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 50", conn)
        metadata.append(
            {
                "table_name": table_name,
                "rowcount": int(df.shape[0]),
                "columns": list(df.columns),
                "preview": df.to_dict(orient="records"),
            }
        )
    return metadata


def run_sql_query(database_path: Path, query: str) -> Dict[str, Any]:
    """Run a read-only SQL query against the observability database."""

    if not query.strip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are permitted")

    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(query, conn)

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "rowcount": int(df.shape[0]),
    }


__all__ = [
    "run_full_pipeline",
    "run_pipeline_rule",
    "run_blueprint_pipeline",
    "ensure_database_seeded",
    "fetch_table_list",
    "run_sql_query",
]
