"""ETL pipeline helpers for running transformations and loading SQLite."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

import json

from .config import AppConfig, PipelineRule


def load_json_payload(path: Path) -> pd.DataFrame:
    """Load an arbitrary JSON file into a single-row pandas DataFrame."""

    data = json.loads(path.read_text())
    return pd.DataFrame([data])


def execute_rule(
    df_input: pd.DataFrame,
    rule: PipelineRule,
    *,
    extra_globals: Dict[str, Any] | None = None,
    rule_override: str | None = None,
) -> pd.DataFrame:
    """Execute a transformation rule and return the resulting DataFrame."""

    env: Dict[str, Any] = {"pd": pd, "df_input": df_input, "INPUT_PARAMS": rule.input_params}
    if extra_globals:
        env.update(extra_globals)

    rule_source = rule_override if rule_override is not None else rule.rule_path.read_text()
    exec(rule_source, {}, env)

    if "df_output" not in env:
        raise RuntimeError(f"Rule {rule.rule_path.name} did not define df_output")

    df_output = env["df_output"]
    if not isinstance(df_output, pd.DataFrame):
        raise TypeError(
            f"Rule {rule.rule_path.name} produced {type(df_output)!r} instead of pandas.DataFrame"
        )

    return df_output


def load_dataframe_to_sqlite(df: pd.DataFrame, table_name: str, database_path: Path) -> None:
    """Replace a table with the given DataFrame contents in SQLite."""

    database_path.parent.mkdir(exist_ok=True)
    with sqlite3.connect(database_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def run_pipeline_rule(rule: PipelineRule, database_path: Path) -> Dict[str, Any]:
    """Execute a single rule end-to-end and store the result."""

    df_input = load_json_payload(rule.input_path)
    df_output = execute_rule(df_input, rule)
    load_dataframe_to_sqlite(df_output, rule.table_name, database_path)

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
    "ensure_database_seeded",
    "fetch_table_list",
    "run_sql_query",
]
