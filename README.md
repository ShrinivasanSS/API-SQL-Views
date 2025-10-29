# Observability ETL Workbench

The Observability ETL Workbench is a Phase IV prototype that demonstrates how
cached SaaS telemetry can be transformed into queryable tables, inspected
through a three-pane UI, and explored with Azure OpenAI–powered assistants. The
project started as a Phase I proof-of-concept ETL runner and has since expanded
to include blueprint-driven orchestration, saved AI tasks, and multi-step AI
workflows.

## Current capabilities

### Data ingestion and modelling (Phases I–II)

- **Configurable ETL pipeline** driven by pandas-based transformation snippets
  that convert cached JSON payloads into observability tables.
- **SQLite workspace** that is rebuilt on demand so the UI always reflects the
  latest run of the pipeline.
- **Three-pane web UI** with navigation, resource listings, and context-aware
  inspectors for credentials, APIs, transformation rules, tables, and
  blueprints.
- **Blueprint registry** that describes entities, related payloads, and the
  tables they populate, making it easy to reason about the demo domain.

### AI tasking and workflows (Phase IV)

- **AI Tasks inspector** with a “Query mode | AI mode” toggle. Query mode runs
  the YAML-defined SQL templates, while AI mode asks Azure OpenAI to generate a
  task-specific SQL statement, validates the result set, and falls back to the
  default query when needed. The inspector now shows the executed SQL, any AI
  suggestions, and the generated summary for each run.
- **AI Workflows** chain multiple tasks together. Each workflow inherits the
  Query/AI toggle, surfaces per-task fallbacks, and displays an aggregated AI
  summary when the run succeeds.
- **Table and workflow inputs as combo boxes** backed by the seeded table names
  so common parameters can be selected quickly while still allowing custom
  overrides.

## Project structure

```
app/
  __init__.py            # Application factory
  ai.py                  # AI task/workflow execution helpers
  assistants.py          # Azure OpenAI helper client
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

1. **Install dependencies**

   ```bash
   python -m venv .venv
   . .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure demo credentials** (optional for the UI). Copy `.env.example` to
   `.env` and adjust the placeholders if needed.

3. **Provide Azure OpenAI settings** if you want to enable AI mode. Populate the
   `AZURE_OPENAI_*` variables in your environment or `.env`; without them the UI
   automatically falls back to Query mode.

4. **Run the Flask server**

   ```bash
   python app.py
   ```

Navigate to http://localhost:5000 to access the workbench. The ETL pipeline runs
on startup and can be retriggered through the “Run ETL Pipeline” button in the
navigation pane. Inspect tables via the SQL workbench, explore API payloads with
the built-in JQ editor, and switch to the AI sections to execute configured
tasks and workflows.

## Working in the inspectors

- All transformed tables are stored in `storage/observability.db`. Delete this
  file if you want a completely clean rebuild.
- Custom transformation rules created from the UI are persisted under
  `storage/user_rules/` and automatically appear in the rule listing.
- The application uses the `jq` Python package to evaluate JQ expressions
  against cached API payloads so that analysts can explore responses directly
  from the browser.
- Cached API payloads created via the upload endpoint are stored in the
  project-root `.cache/` directory so they can be referenced by transformation
  rules without committing large fixtures to source control.
- Uploaded blueprint YAML files live under `.uploads/`; any file placed in this
  directory is merged into the runtime blueprint registry alongside the entries
  defined in `blueprints/blueprints.csv`.

## Uploading blueprints and cached payloads

The Blueprint Catalogue view now exposes an upload panel for YAML blueprints
and JSON payloads. Select one or more files and submit them to persist the
artifacts on disk:

- `.yaml`/`.yml` files are saved in `.uploads/` (nested folders are supported).
  The entity type is derived from the YAML content and automatically added to
  the in-memory registry unless a CSV-sourced blueprint already uses the same
  identifier. Updating a previously uploaded blueprint simply overwrites the
  existing file.
- `.json` files are copied into `.cache/` so transformation rules can reference
  the payloads without manual file system access.

After uploading new blueprints, rerun the ETL pipeline from the UI to materialise
the additional tables in SQLite.

## Roadmap

- Phase V explorations such as saved automations, rule authoring with
  change-tracking, and user-defined workflow builders.
- Uploading new JSON payloads directly from the UI and persisting them alongside
  generated rules.
- Authentication, authorisation, and collaboration tooling once the workbench
  evolves beyond a single-user demo.
