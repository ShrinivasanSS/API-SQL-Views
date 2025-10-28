# Agent Notes

## Current Progress
- Phase I and Phase II foundations are complete: APIs, tables, and views are scaffolded and blueprints now drive pipeline execution.
- Blueprint registry (`blueprints/blueprints.csv`) resolves YAML blueprints (e.g., `blueprints/isp_blueprint.yaml`) that describe entity sources, rules, and views.
- UI exposes credentials, APIs, transformation rules, database tables, and now a blueprint catalogue with inspectors.
- Startup seeding now provisions default tables via `seed.yaml` and cached API payloads live under `.cache` to avoid redundant fetches.

## Project Goals
- Simplify construction of tables and views from YAML configurations while keeping the system extensible for future phases.
- Maintain declarative blueprints that map entities to data pillars, transformation rules, and load targets.
- Provide interactive tooling (web UI + APIs) for inspecting payloads, running rules, previewing SQL, and now reviewing blueprint schemas.

## Upcoming Roadmap Ideas
- Phase III/IV: extend blueprint coverage, add workflow/automation designers, integrate blueprint editing & validation, support multi-entity pipelines.
- Introduce blueprint diffing/versioning, validation checks, and integration with rule previews.
- Expand saved views/workflows sections once pipeline outputs stabilise.

## Navigating the Repository
- `app/` holds Flask app code: routes, blueprint registry utilities, pipeline orchestration, templates, and static assets.
  - `app/blueprints.py` parses YAML blueprints.
  - `app/routes.py` serves HTML + API endpoints.
  - `app/static/` contains the UI (CSS/JS) and templates live in `app/templates/`.
- `blueprints/` stores registry CSV + YAML definitions.
- `sample_inputs/`, `sample_transformation_rules/`, and `storage/` provide demo data, rules, and generated artefacts.
- `agents/[DONE]-Phase-*.md` capture completed phase requirements; update this `AGENTS.md` as checkpoints evolve.
