# Agent Notes

## Current Progress
- Phases I–II delivered the ETL core: cached APIs, transformation rules, seeded tables, and the blueprint registry.
- Phase III introduced the blueprint inspectors and navigation refinements across the three-pane UI.
- Phase IV added the AI Tasks and AI Workflows panes with Query/AI mode toggles, Azure OpenAI integration, SQL validation/fallback, and generated summaries surfaced in the inspectors.
- `seed.yaml` now powers both table defaults and the AI catalogue; cached payloads still live under `.cache` to minimise redundant fetches.

## Project Goals
- Simplify construction of tables and views from YAML configurations while keeping the system extensible for future phases.
- Maintain declarative blueprints that map entities to data pillars, transformation rules, and load targets.
- Provide interactive tooling (web UI + APIs) for inspecting payloads, running rules, previewing SQL, and now reviewing blueprint schemas.

## Upcoming Roadmap Ideas
- Phase V concepts: saved automations, workflow/automation designers, and richer blueprint editing & validation flows.
- Blueprint diffing/versioning plus automated validation checks that integrate with rule previews.
- Expand saved views/workflows and introduce collaboration features (auth, change tracking) once pipeline outputs stabilise.

## Navigating the Repository
- `app/` holds Flask app code: routes, blueprint registry utilities, AI orchestration, templates, and static assets.
  - `app/ai.py` runs AI tasks/workflows (query & AI modes with fallback reporting).
  - `app/assistants.py` wraps Azure OpenAI calls for SQL generation and summarisation.
  - `app/routes.py` serves HTML + API endpoints.
  - `app/static/` contains the UI (CSS/JS) and templates live in `app/templates/`.
- `blueprints/` stores registry CSV + YAML definitions.
- `examples/` hosts demo payloads, rules, and outputs under `inputs/`, `transformation_rules/`, `outputs/`, and `source_rules/`.
- `storage/` provides generated artefacts such as the SQLite database and user-authored rules.
- `agents/[DONE]-Phase-*.md` capture completed phase requirements; update this `AGENTS.md` as checkpoints evolve.
