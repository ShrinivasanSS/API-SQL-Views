"""Flask blueprints that expose both the HTML UI and API endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jq
from flask import Blueprint, Response, current_app, jsonify, render_template, request

from .assistants import ASSISTANT_EXTENSION_KEY, AssistantServiceError
from .config import PROJECT_ROOT, PipelineRule
from .pipeline import execute_rule, fetch_table_list, load_json_payload, run_full_pipeline
from .pipeline import run_sql_query
from .utils import load_json, preview_json, safe_resolve


SAMPLE_INPUTS = PROJECT_ROOT / "sample_inputs"
RULES_DIR = PROJECT_ROOT / "sample_transformation_rules"
USER_RULES_DIR = PROJECT_ROOT / "storage" / "user_rules"
USER_RULES_DIR.mkdir(parents=True, exist_ok=True)


def _rule_by_name(name: str) -> PipelineRule | None:
    for metadata in current_app.config["PIPELINE_RULES"]:
        if metadata["name"] == name:
            # Rehydrate PipelineRule for convenience.
            return PipelineRule(
                name=metadata["name"],
                title=metadata["title"],
                input_path=PROJECT_ROOT / metadata["input"],
                rule_path=PROJECT_ROOT / metadata["rule"],
                table_name=metadata["table_name"],
                table_type=metadata["table_type"],
                description=metadata["description"],
                input_params=metadata.get("input_params", {}),
                load_mode=metadata.get("load_mode", "replace"),
                upsert_keys=tuple(metadata["upsert_keys"])
                if metadata.get("upsert_keys")
                else None,
            )
    return None


def _list_api_payloads() -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(SAMPLE_INPUTS.rglob("*.json")):
        relative = path.relative_to(SAMPLE_INPUTS)
        payloads.append(
            {
                "id": str(relative),
                "name": path.stem,
                "category": relative.parts[0] if len(relative.parts) > 1 else "",
                "preview": preview_json(path),
            }
        )
    return payloads


ui_bp = Blueprint("ui", __name__)
api_bp = Blueprint("api", __name__)


@ui_bp.get("/")
def index() -> Response:
    return render_template("index.html")


@api_bp.get("/status")
def api_status() -> Response:
    tables = fetch_table_list(Path(current_app.config["DATABASE_PATH"]))
    return jsonify(
        {
            "credentials": current_app.config["CREDENTIALS"],
            "rules": current_app.config["PIPELINE_RULES"],
            "tables": tables,
            "blueprints": current_app.config.get("BLUEPRINTS", {}),
        }
    )


@api_bp.get("/credentials")
def get_credentials() -> Response:
    return jsonify(current_app.config["CREDENTIALS"])


@api_bp.get("/apis")
def list_apis() -> Response:
    return jsonify(_list_api_payloads())


@api_bp.get("/apis/<path:payload_id>")
def get_api_payload(payload_id: str) -> Response:
    path = safe_resolve(SAMPLE_INPUTS, payload_id)
    return jsonify(load_json(path))


@api_bp.post("/apis/<path:payload_id>/jq")
def run_jq_query(payload_id: str) -> Response:
    payload = load_json(safe_resolve(SAMPLE_INPUTS, payload_id))
    query = request.json.get("query", "") if request.is_json else ""
    if not query:
        return jsonify({"error": "Query body is required"}), 400

    try:
        program = jq.compile(query)
        result = list(program.input(payload).all())
    except Exception as exc:  # pragma: no cover - jq raises many types
        return jsonify({"error": str(exc)}), 400

    return jsonify({"result": result})


@api_bp.post("/assistants/apis")
def assist_jq_query() -> Response:
    service = current_app.extensions.get(ASSISTANT_EXTENSION_KEY)
    if service is None:
        return jsonify({"error": "Assistant service is not configured"}), 503

    payload = request.get_json(force=True)
    prompt = (payload.get("prompt") or "").strip()
    payload_id = (payload.get("payload_id") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if not payload_id:
        return jsonify({"error": "payload_id is required"}), 400

    data = load_json(safe_resolve(SAMPLE_INPUTS, payload_id))
    excerpt = _truncate_json(data)

    try:
        result = service.suggest_jq(prompt=prompt, payload_excerpt=excerpt)
    except AssistantServiceError as exc:  # pragma: no cover - relies on Azure OpenAI
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@api_bp.get("/rules")
def list_rules() -> Response:
    rules = []
    for metadata in current_app.config["PIPELINE_RULES"]:
        path = PROJECT_ROOT / metadata["rule"]
        rules.append({**metadata, "content": path.read_text(encoding="utf-8")})

    for path in sorted(USER_RULES_DIR.glob("*.rules")):
        rules.append(
            {
                "name": path.stem,
                "title": path.stem,
                "input": "",
                "rule": str(path.relative_to(PROJECT_ROOT)),
                "table_name": "",
                "table_type": "custom",
                "description": "User supplied rule",
                "content": path.read_text(encoding="utf-8"),
                "input_params": {},
            }
        )

    return jsonify(rules)


@api_bp.post("/rules")
def create_rule() -> Response:
    payload = request.get_json(force=True)
    name = payload.get("name")
    content = payload.get("content")
    if not name or not content:
        return jsonify({"error": "Both name and content are required"}), 400

    path = USER_RULES_DIR / f"{name}.rules"
    path.write_text(content, encoding="utf-8")

    return jsonify({"status": "created", "path": str(path.relative_to(PROJECT_ROOT))})


@api_bp.post("/rules/preview")
def preview_rule() -> Response:
    payload = request.get_json(force=True)
    rule_name = payload.get("rule_name")
    rule_content = payload.get("rule_content")
    input_path = payload.get("input_path")

    if not input_path and rule_name:
        rule = _rule_by_name(rule_name)
        if rule:
            input_path = str(rule.input_path.relative_to(PROJECT_ROOT))

    if not input_path:
        return jsonify({"error": "An input_path or known rule_name must be provided"}), 400

    if rule_name:
        rule = _rule_by_name(rule_name)
    else:
        rule = None

    if rule is None:
        rule = PipelineRule(
            name="preview",
            title="Preview",
            input_path=PROJECT_ROOT / input_path,
            rule_path=PROJECT_ROOT / input_path,  # dummy path, not used
            table_name="preview",
            table_type="custom",
            description="preview",
        )

    df_input = load_json_payload(PROJECT_ROOT / input_path)
    df_output = execute_rule(df_input, rule, rule_override=rule_content)

    return jsonify(
        {
            "columns": list(df_output.columns),
            "rows": df_output.to_dict(orient="records"),
            "rowcount": int(df_output.shape[0]),
        }
    )


@api_bp.post("/assistants/rules")
def assist_rule_generation() -> Response:
    service = current_app.extensions.get(ASSISTANT_EXTENSION_KEY)
    if service is None:
        return jsonify({"error": "Assistant service is not configured"}), 503

    payload = request.get_json(force=True)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    current_code = payload.get("current_code") or ""
    input_path = payload.get("input_path") or ""
    preview = ""
    if input_path:
        try:
            json_payload = load_json(safe_resolve(PROJECT_ROOT, input_path))
            preview = _truncate_json(json_payload)
        except FileNotFoundError:
            preview = ""

    try:
        result = service.suggest_rule(
            prompt=prompt,
            current_code=current_code,
            input_preview=preview,
        )
    except AssistantServiceError as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@api_bp.post("/pipeline/run")
def run_pipeline() -> Response:
    results = run_full_pipeline(current_app.config_obj)
    return jsonify(results)


@api_bp.get("/tables")
def list_tables() -> Response:
    tables = fetch_table_list(Path(current_app.config["DATABASE_PATH"]))
    return jsonify(tables)


@api_bp.post("/tables/query")
def query_table() -> Response:
    payload = request.get_json(force=True)
    query = payload.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        result = run_sql_query(Path(current_app.config["DATABASE_PATH"]), query)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 400

    return jsonify(result)


@api_bp.post("/assistants/tables")
def assist_sql_query() -> Response:
    service = current_app.extensions.get(ASSISTANT_EXTENSION_KEY)
    if service is None:
        return jsonify({"error": "Assistant service is not configured"}), 503

    payload = request.get_json(force=True)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    table_name = (payload.get("table_name") or "").strip() or None
    schema_description = ""
    sample_rows = ""
    if table_name:
        tables = fetch_table_list(Path(current_app.config["DATABASE_PATH"]))
        table_info = next((item for item in tables if item["table_name"] == table_name), None)
        if table_info:
            schema_description = ", ".join(table_info.get("columns", []))
            sample_rows = json.dumps(table_info.get("preview", [])[:3], indent=2)

    try:
        result = service.suggest_sql(
            prompt=prompt,
            table_name=table_name,
            schema_description=schema_description,
            sample_rows=sample_rows,
        )
    except AssistantServiceError as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 502

    return jsonify(result)


@api_bp.get("/blueprints")
def list_blueprints() -> Response:
    payload = current_app.config.get("BLUEPRINTS", {})
    items = [
        {
            "entity_type": entity_type,
            **metadata,
        }
        for entity_type, metadata in payload.items()
    ]
    items.sort(key=lambda item: item["entity_type"].lower())
    return jsonify(items)


@api_bp.get("/blueprints/<string:entity_type>")
def get_blueprint(entity_type: str) -> Response:
    registry = current_app.config_obj.blueprint_registry
    description = registry.describe_blueprint(entity_type)
    if description is None:
        return jsonify({"error": "Blueprint not found"}), 404
    return jsonify(description)


@api_bp.get("/views")
def get_views() -> Response:
    return jsonify([])


@api_bp.get("/workflows")
def get_workflows() -> Response:
    return jsonify({"status": "work in progress"})


@api_bp.get("/automations")
def get_automations() -> Response:
    return jsonify({"status": "work in progress"})


def _truncate_json(payload: Any, limit: int = 4000) -> str:
    """Render JSON payloads while keeping prompts below token limits."""

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."

