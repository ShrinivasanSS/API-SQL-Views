"""Application factory for the Observability ETL demo."""

from __future__ import annotations

from flask import Flask

from .config import AppConfig
from .pipeline import ensure_database_seeded
from .routes import api_bp, ui_bp


def create_app(config: AppConfig | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__, static_folder="static", template_folder="templates")

    app_config = config or AppConfig.load_from_env()
    app.config.update(app_config.as_flask_config())
    app.config_obj = app_config

    # Seed the SQLite database on startup so the UI has data to display.
    ensure_database_seeded(app_config)

    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app


__all__ = ["create_app"]
