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
blueprints/              # Blueprint registry CSV + YAML definitions
sample_inputs/           # Cached API responses (extraction stage)
sample_transformation_rules/
                        # pandas-based rule snippets (transformation stage)
storage/                 # Generated SQLite database and custom rules
```

The AI task and workflow definitions live in `seed.yaml`. At start-up the file
is parsed into `AppConfig`, which initialises the ETL pipeline, exposes the AI
catalogue, and (when Azure OpenAI credentials are provided) configures the
assistant service that powers AI mode in the inspectors.

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

- **SQL Workbench (Tables view)**: continue to choose between raw SQL and
  natural-language prompting for ad hoc questions.
- **AI Tasks**: select Query mode to execute the saved SQL steps verbatim or
  switch to AI mode to let the assistant generate and run a bespoke statement.
  The UI records the AI suggestion, shows whether a fallback occurred, and
  renders the generated summary.
- **AI Workflows**: the top-level toggle applies to every step in the chain and
  exposes per-task summaries, SQL, and any fallback warnings.

## Roadmap

- Phase V explorations such as saved automations, rule authoring with
  change-tracking, and user-defined workflow builders.
- Uploading new JSON payloads directly from the UI and persisting them alongside
  generated rules.
- Authentication, authorisation, and collaboration tooling once the workbench
  evolves beyond a single-user demo.
