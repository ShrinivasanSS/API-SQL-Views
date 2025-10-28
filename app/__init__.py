"""Application factory for the Observability ETL demo."""

from __future__ import annotations

from flask import Flask

from .assistants import (
    ASSISTANT_EXTENSION_KEY,
    AssistantServiceError,
    build_assistant_service,
)
from .config import AppConfig
from .pipeline import ensure_database_seeded
from .routes import api_bp, ui_bp


def create_app(config: AppConfig | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__, static_folder="static", template_folder="templates")

    app_config = config or AppConfig.load_from_env()
    app.config.update(app_config.as_flask_config())
    app.config_obj = app_config

    assistant_service = None
    if app_config.azure_openai:
        try:
            assistant_service = build_assistant_service(app_config.azure_openai)
        except AssistantServiceError as exc:  # pragma: no cover - requires Azure OpenAI
            app.logger.warning("Assistant service disabled: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.exception("Failed to initialise assistant service")
    app.extensions[ASSISTANT_EXTENSION_KEY] = assistant_service

    # Seed the SQLite database on startup so the UI has data to display.
    ensure_database_seeded(app_config)

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


__all__ = ["create_app"]
