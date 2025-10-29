# Observability ETL Workbench

This repository contains a proof-of-concept observability data workbench. The
goal of **Phase I** is to ingest cached API responses, apply transformation
rules, and materialise a set of preset observability tables in SQLite. A Flask
frontend allows browsing the cached payloads, fine-tuning transformation rules
and querying the resulting tables.

## Features implemented in Phase I

- **Configurable ETL pipeline** driven by transformation rule snippets written
  in pandas. Each rule converts a JSON payload into one of the preset
  observability tables (entities, metrics, logs, events and traces).
- **SQLite data store** that is rebuilt every time the ETL pipeline runs so the
  UI always works with fresh data.
- **Flask-based UI** that mimics the required three pane layout:
  - **Left pane** – navigation for credentials, APIs, transformation rules,
    tables, views, workflows and automations.
  - **Middle pane** – context-aware listing of resources (API payloads, rules,
    tables, etc.).
  - **Right pane** – interactive tools such as a JQ explorer for API payloads,
    rule preview editor and a SQLite query runner.
- **Rule authoring support** with live previews and the ability to persist
  custom rules in the local storage directory.

## Phase II additions

- **Blueprint-driven orchestration** converts YAML definitions into runnable
  extraction/transformation/load jobs per entity type so that new pipelines can
  be onboarded without touching Python code.
- **Events pillar uplift** introduces a normalised schema (`EntityType`,
  `EntityID`, `EventType`, `EventTime`, etc.) and blueprint-defined upsert keys
  to keep pagination idempotent.
- **Registry-backed discovery** exposes available blueprints (including their
  source rules and sample payloads) to the UI for previewing and debugging.

## Project structure

```
app/
  __init__.py            # Application factory
  config.py              # Runtime configuration & pipeline definitions
  pipeline.py            # ETL helpers (extract, transform, load)
  routes.py              # Flask blueprints for UI and REST API
  static/                # CSS + JavaScript for the three-pane UI
  templates/index.html   # Base layout
examples/
  inputs/               # Cached API responses (extraction stage)
  outputs/              # Example table snapshots for reference
  source_rules/         # Blueprint-managed rule bundles
  transformation_rules/ # pandas-based rule snippets (transformation stage)
storage/                 # Generated SQLite database and custom rules
```

The pipeline configuration lives in `app/config.py`. Each `PipelineRule` entry
references a JSON payload and a transformation rule. Adding new entity types is
as easy as dropping a new payload and rule file in the relevant folders under
`examples/` and extending the configuration tuple. Blueprint metadata is stored under
`blueprints/` with a `blueprints.csv` registry and one YAML file per entity
type.

## Getting started

### 1. Install dependencies

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure demo credentials

Copy `.env.example` to `.env` and update the placeholders if needed. The
credentials are only shown in the UI; no live API calls are performed in this
phase.

### 3. Run the Flask server

```bash
python app.py
```

Navigate to http://localhost:5000 to access the workbench. The ETL pipeline
runs during startup and can be re-triggered at any time via the “Run ETL
Pipeline” button in the left navigation pane. Tables can be inspected via the
SQL runner in the right pane of the “Tables” view.

## Development notes

- All transformed tables are stored in `storage/observability.db`. Delete this
  file if you want a completely clean rebuild.
- Custom transformation rules created from the UI are persisted under
  `storage/user_rules/` and automatically appear in the rule listing.
- The application uses the `jq` Python package to evaluate JQ expressions
  against cached API payloads so that analysts can explore responses directly
  from the browser.

## Next steps (beyond Phase I)

- Implement phase-specific extensions such as Saved Views, Workflows and
  Automations.
- Allow uploading new JSON payloads and associating them with custom
  transformation rules without touching the file system.
- Add authentication and fine-grained access controls for collaborative use.
