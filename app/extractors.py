"""Built-in extractor utilities for declarative blueprint tables."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd


def _normalise_value(value: Any) -> Any:
    """Convert nested structures to JSON strings for SQL compatibility."""

    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def _split_path(path: str) -> List[str]:
    """Split a simple JSON path (dot notation with optional indexes)."""

    path = path.strip()
    if not path or path == "$":
        return []
    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        path = path[1:]
    return [segment for segment in path.split(".") if segment]


def _descend(obj: Any, token: str) -> Any:
    """Navigate a JSON-like object following a single path token."""

    if not token:
        return obj

    # Support trailing [] which simply returns the collection.
    if token.endswith("[]"):
        token = token[:-2]
        if not token:
            return obj

    name = token
    index: int | None = None
    if "[" in token:
        name, index_part = token.split("[", 1)
        index = int(index_part.rstrip("]"))

    if name:
        if isinstance(obj, Mapping):
            obj = obj.get(name)
        else:
            return None

    if index is not None:
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            try:
                obj = obj[index]
            except IndexError:
                return None
        else:
            return None

    return obj


def resolve_json_path(data: Any, path: str) -> Any:
    """Resolve a limited JSONPath-like expression against *data*."""

    current = data
    for token in _split_path(path):
        current = _descend(current, token)
        if current is None:
            break
    return current


def _lookup_context(context: Mapping[str, Any] | None, path: str) -> Any:
    if not context:
        return None

    current: Any = context
    for token in [part for part in path.split(".") if part]:
        if isinstance(current, Mapping):
            current = current.get(token)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)) and token.isdigit():
            idx = int(token)
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
        else:
            return None
        if current is None:
            return None
    return current


def _extract_mapping_value(
    row: Any,
    expression: Any,
    *,
    root: Any,
    context: Mapping[str, Any] | None,
) -> Any:
    """Evaluate a mapping expression against the current row."""

    if isinstance(expression, str):
        expression = expression.strip()
        if not expression:
            return None
        if expression == "$":
            return _normalise_value(row)
        if expression.startswith("$"):
            return _normalise_value(resolve_json_path(row, expression))
        if expression.startswith("context."):
            path_expr = expression[len("context.") :]
            if isinstance(row, Mapping) and "{" in path_expr and "}" in path_expr:
                try:
                    path_expr = path_expr.format(**row)
                except Exception:  # pragma: no cover - defensive
                    pass
            return _normalise_value(_lookup_context(context, path_expr))
        if expression.startswith("@@"):
            # Explicit root reference using @@ prefix to avoid ambiguity.
            return _normalise_value(resolve_json_path(root, expression[1:]))
        if expression.startswith("@"):
            return _normalise_value(resolve_json_path(root, expression[1:]))
        if "." in expression or "[" in expression:
            return _normalise_value(resolve_json_path(row, expression))
        if isinstance(row, Mapping) and expression in row:
            return _normalise_value(row.get(expression))
        return _normalise_value(expression)

    if callable(expression):  # pragma: no cover - advanced use
        return _normalise_value(expression(row))

    # Assume literal/constant.
    return _normalise_value(expression)


def _rows_from_mapping(
    items: Iterable[Mapping[str, Any]],
    mapping: Mapping[str, Any],
    *,
    root: Any,
    context: Mapping[str, Any] | None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        row = {
            column: _extract_mapping_value(item, expr, root=root, context=context)
            for column, expr in mapping.items()
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_json_array_extractor(
    data: Any, config: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    path = config.get("path", "$") if isinstance(config, Mapping) else "$"
    target = resolve_json_path(data, str(path)) if path else data

    if target is None:
        return pd.DataFrame()

    if isinstance(target, Mapping):
        target_iterable: Iterable[Mapping[str, Any]] = [target]
    elif isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
        target_iterable = [item for item in target if isinstance(item, Mapping)]
    else:
        return pd.DataFrame()

    mapping = config.get("mapping", {}) if isinstance(config, Mapping) else {}
    if not isinstance(mapping, Mapping):
        mapping = {}

    if not mapping:
        # Return the raw objects if no mapping specified.
        return pd.DataFrame(list(target_iterable))

    return _rows_from_mapping(target_iterable, mapping, root=data, context=context)


def run_json_object_extractor(
    data: Any, config: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    path = config.get("path", "$") if isinstance(config, Mapping) else "$"
    target = resolve_json_path(data, str(path)) if path else data

    if not isinstance(target, Mapping):
        return pd.DataFrame()

    mapping = config.get("mapping", {}) if isinstance(config, Mapping) else {}
    if not isinstance(mapping, Mapping):
        mapping = {}

    if not mapping:
        return pd.DataFrame([target])

    row = {
        column: _extract_mapping_value(target, expr, root=data, context=context)
        for column, expr in mapping.items()
    }
    return pd.DataFrame([row])


def run_json_key_value_extractor(
    data: Any, config: Mapping[str, Any], *, context: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    path = config.get("path", "$") if isinstance(config, Mapping) else "$"
    target = resolve_json_path(data, str(path)) if path else data

    mapping = config.get("mapping", {}) if isinstance(config, Mapping) else {}
    if not isinstance(mapping, Mapping):
        mapping = {}

    rows: List[Mapping[str, Any]] = []
    ignore_keys = set()
    if isinstance(config, Mapping):
        raw_ignore = config.get("ignore_keys")
        if isinstance(raw_ignore, Sequence) and not isinstance(
            raw_ignore, (str, bytes, bytearray)
        ):
            ignore_keys = {str(key) for key in raw_ignore}

    if isinstance(target, Mapping):
        for key, value in target.items():
            if str(key) in ignore_keys:
                continue
            base_row: Dict[str, Any] = {
                "key": key,
                "value": value,
            }
            if isinstance(value, Mapping):
                base_row.update(value)
            rows.append(base_row)
    elif isinstance(target, Sequence) and not isinstance(target, (str, bytes, bytearray)):
        for item in target:
            if isinstance(item, Mapping):
                rows.append(dict(item))

    if not rows:
        return pd.DataFrame()

    if not mapping:
        return pd.DataFrame(rows)

    return _rows_from_mapping(rows, mapping, root=data, context=context)


EXTRACTOR_DISPATCH = {
    "jsonarrayextractor": run_json_array_extractor,
    "jsonobjectextractor": run_json_object_extractor,
    "jsonkeyvalueextractor": run_json_key_value_extractor,
}


@dataclass(slots=True)
class ExtractorDefinition:
    """Declarative configuration for a built-in extractor."""

    name: str
    config: Mapping[str, Any]
    output_row_name: str | None = None
    merge_strategy: str | None = None


def apply_extractors(
    payload: Any,
    extractors: Sequence[ExtractorDefinition],
    *,
    context: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a sequence of extractor definitions against *payload*."""

    combined: pd.DataFrame | None = None
    named_frames: Dict[str, pd.DataFrame] = {}

    for definition in extractors:
        func = EXTRACTOR_DISPATCH.get(definition.name.lower())
        if func is None:
            raise ValueError(f"Unsupported extractor: {definition.name}")

        df_part = func(payload, definition.config, context=context)
        df_part = df_part.copy()

        if definition.output_row_name:
            named_frames[definition.output_row_name] = df_part

        if combined is None:
            combined = df_part
            continue

        strategy = (definition.merge_strategy or "align").lower()
        if strategy == "broadcast":
            if df_part.empty:
                continue
            if df_part.shape[0] != 1:
                raise ValueError(
                    "Broadcast merge requires the extractor to return exactly one row"
                )
            combined = combined.assign(_merge_key=1).merge(
                df_part.assign(_merge_key=1), on="_merge_key"
            )
            combined = combined.drop(columns=["_merge_key"])
            continue

        if strategy == "align":
            combined = pd.concat(
                [combined.reset_index(drop=True), df_part.reset_index(drop=True)],
                axis=1,
            )
            continue

        if strategy.startswith("join:"):
            _, _, key_spec = strategy.partition(":")
            keys = [key.strip() for key in key_spec.split(",") if key.strip()]
            if not keys:
                raise ValueError("Join merge strategy requires key columns")
            combined = combined.merge(df_part, on=keys, how="left")
            continue

        raise ValueError(f"Unsupported merge strategy: {definition.merge_strategy}")

    if combined is None:
        return pd.DataFrame()

    return combined


__all__ = [
    "ExtractorDefinition",
    "apply_extractors",
    "resolve_json_path",
]

