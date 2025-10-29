# TODO — Phase II: Blueprint-Driven Prefill of Tables & Views (with Events)

## Objective

Make the ETL fully **blueprint-driven**. For each `EntityType`, a YAML blueprint declares:

* which APIs to pull,
* which transformation rules to run,
* how to **load** into preset Observability tables (`Entities`, `Metrics`, `Traces`, `Logs`, **`Events`**),
* and which **Views** to register/materialize.

This phase adds **Events** as a first-class pillar. Unlike other pillars, all Events come from one shared API; only the **query** (filters + time range) varies by entity.

> **Implementation note:** For the offline prototype we attach `sample_input` hints to each blueprint source. These point to cached JSON payloads under `examples/inputs/` and keep the ETL runnable without live API access. Future phases can drop the hint in favour of real extractors.

---

## What’s New in Phase II

1. **Blueprint Registry** (`blueprints.csv`) → one YAML per `EntityType`.
2. **Typed arrays** per pillar: `metrics[]`, `traces[]`, `logs[]`, `events[]`.
3. **Events pillar** with special fetch semantics:

   * One shared endpoint:
     `/api/applog/search/{start_date_dd-mm-yyyy}%20{start_time_hh:mm:ss}/{end_date_dd-mm-yyyy}%20{end_time_hh:mm:ss}/{range}/desc?`
   * Example:
     `api/applog/search/25-10-2025%2006:17:26/26-10-2025%2006:17:25/1-100/desc?`
   * Query expression (e.g., `logtype="Infrastructure Events" and source="ISP"`) is **blueprint-defined** and passed as URL querystring (`q=`) or provider-specific param (see below).

---

## Blueprint Files

### 1) Registry

```csv
EntityType,BlueprintFile
ISP,isp_blueprint.yaml
```

### 2) Blueprint Schema (superset)

```yaml
version: 1

entity:
  entityType: ISP
  entityIdField: EntityID

defaults:
  method: GET
  # Upsert/merge keys per pillar; can be overridden at source level
  upsert:
    metrics:   [EntityType, EntityID, EntityInstance, MetricName]
    traces:    [TraceEntityType, TraceEntityID, TraceType, TraceInstance]
    logs:      [LogEntityType, LogEntityID, LogType, collection_time]
    events:    [EntityType, EntityID, EventType, EventTime, EventId]  # EventId optional if provider lacks it

metrics:
  - type: isp_tabular
    api: /app/api/isp/tabulardetails/{monitor_id}
    sample_input: examples/inputs/metrics/api_isp_tabular_details_15698000397185121.json
    transform_rules: [examples/transformation_rules/ISP/isp_tabulardata_metric_transformation.rules]
    inputs:
      metric_units:
        packet_loss: "%"
        latency: "ms"
        mtu: "bytes"
        jitter: "ms"
        asnumber_count: "count"
        hop_count: "count"
    load:
      table: Metrics

traces:
  - type: traceroute
    api: /api/isp/traceroute/{monitor_id}
    sample_input: examples/inputs/traces/api_isp_traceroute_15698000397185121.json
    transform_rules: [examples/transformation_rules/ISP/isp_trace_transformation.rules]
    load:
      table: Traces

logs:
  - type: availability_log
    api: /api/reports/log_reports/{monitor_id}?date={today}
    sample_input: examples/inputs/logs/api_isp_15698000397185121_logreport.json
    transform_rules: [examples/transformation_rules/ISP/isp_logreport_transformation.rules]
    load:
      table: Logs

# ─────────────────────────────
# NEW: Events pillar blueprint
# ─────────────────────────────
events:
  - type: Infrastructure
    # Logical filter the user wants (kept declarative)
    filter:
      expression: 'logtype="Infrastructure Events" and source="ISP"'
    # Shared events API – only the time window and range vary
    api:
      path: /api/applog/search/{start_date}%20{start_time}/{end_date}%20{end_time}/{range}/desc?
      # range format: "{start_index}-{end_index}" e.g., "1-100"
      # Provider-specific query parameter name for the filter:
      query_param: q
    sample_input: examples/inputs/events/api_isp_infrastructure_events.json
    pagination:
      page_size: 100                # chunk size for range window
      start_index: 1                # first index (inclusive)
      max_pages: 50                 # safety cap; stop earlier if page returns empty
    schedule:
      lookback: "24h"               # default rolling window when not specified by workflow
      align_to: "now"               # or "hour", "day"
    transform_rules: [examples/transformation_rules/eventlogs_transformation.rules]
    inputs:
      event_type: Infrastructure
    load:
      table: Events
      # Optional override for upsert keys if provider supplies an id
      upsert_keys: [EntityType, EntityID, EventType, EventId]
      # Map common event fields (the rule should output these columns)
      expected_columns:
        - EntityType
        - EntityID
        - EntityName
        - EventId
        - EventTime
        - EventType
        - Severity
        - Source
        - Message
        - Raw
```

> You’ll create one such YAML per `EntityType`. The `events[]` array lets you add more event filters later (e.g., `Security`, `Performance`) pointing to the same API.

---

## ETL Engine Changes

### A) Entity Loop

* For each row in `Entities`, pick its `EntityType` and load the matching blueprint.

### B) Placeholder Resolution

* Built-in variables: `{entity_id}`, `{monitor_id}` (alias for `entity_id` if the entityIdField is `monitor_id`), `{today}`, `{start_date}`, `{start_time}`, `{end_date}`, `{end_time}`, `{range}`.
* Time macros:

  * If a workflow specifies an explicit window, use it.
  * Else, derive `(start, end)` based on `events[].schedule.lookback` and `align_to`.
  * Format into `dd-mm-yyyy` and `hh:mm:ss` when injecting into the events API.

### C) Fetch Mechanics — Events

1. **Build filter**: take `events[i].filter.expression` and set it on `events[i].api.query_param` (e.g., `q=`).
2. **Build windowed path**:

   * `start_date`, `start_time`, `end_date`, `end_time` → format to `dd-mm-yyyy` and `hh:mm:ss`.
   * `range` → `"startIndex-endIndex"`, beginning with `start_index` and stepping by `page_size`.
3. **Paging**:

   * Request `.../{start}-{end}/desc?{query_param}={url_encoded(expression)}`.
   * If the page returns **0 results**, stop paging.
   * Else, increment the range (e.g., `1-100`, `101-200`, `201-300`, …) until `max_pages` or empty page.
4. **Transform**: pipe each page’s payload through `eventlogs_transformation.rules` and accumulate rows.
5. **Load**: upsert into `Events` using configured keys.

> Note: If the provider later introduces a cursor token, we’ll add `pagination.cursor_path` and switch strategies without changing the blueprint shape.

### D) Fetch Mechanics — Other Pillars

* Continue to use the blueprint’s `api` URI (and query params if present).
* Run the named `transform_rules`.
* Load into the designated table with pillar-level or source-level `upsert_keys`.

---

## Transformation Rules Contract

Each rule should output a **DataFrame** with the preset table’s columns. For `Events`, the rule should map/derive at least:

| Column     | Description                                                                           |
| ---------- | ------------------------------------------------------------------------------------- |
| EntityType | E.g., `ISP` (inject from blueprint/entity during transform or load)                   |
| EntityID   | The current entity’s id                                                               |
| EntityName | Optional helper sourced from the entities table                                       |
| EventType  | E.g., `Infrastructure` (from blueprint source `type`)                                 |
| EventTime  | Parsed timestamp (UTC recommended)                                                    |
| Severity   | Normalized string or numeric level                                                    |
| Source     | E.g., `ISP`                                                                           |
| Message    | Human-readable summary                                                                |
| EventId    | Provider event id if available; else build a hash of `(EntityID, EventTime, Message)` |
| Raw        | JSON string of the raw event (optional but recommended)                               |

(For `Metrics`, `Traces`, `Logs`, keep the Phase-1 tall-table shapes you already use.)

---

## Loader / Upsert Guidance

* **Idempotency**: always merge, never blindly append. Use the default keys or source-level overrides.
* **Time precision**: store UTC; if you must store local, also store `EventTimeUTC`.
* **Conflicts**: if two pages return the same event, prefer the later (descending) one, but the upsert key should prevent duplication anyway.

---

## UI Integration (Playground)

1. **APIs page**

   * Keep “Copy Response”.
   * Add **“Run via Blueprint”** to exercise a single events source over a small window (e.g., last 15 minutes) for quick previews.

2. **Transformation Rules**

   * Allow selecting a rule from a blueprint source and **Preview** with the cached payload (for events: preview with a one-page fetch).

3. **Tables**

   * Add “Load via Blueprint” buttons per entity or entity type.
   * For `Events`, show the window and range used in the run summary.

4. **Views**

   * Support blueprint-declared `views[]`. Let the user **Register** (logical view/template) or **Materialize** (write result into a view table).
   * Parameter binding (`:entity_type`, `:entity_id`, `:start`, `:end`) via UI controls.

---

## Example: ISP Blueprint (with Events)

```yaml
version: 1
entity:
  entityType: ISP
  entityIdField: monitor_id

metrics:
  - type: isp_tabular
    api: /app/api/isp/tabulardetails/{monitor_id}
    transform_rules: [isp_tabular_metric_transformation.rules]
    load: { table: Metrics }

traces:
  - type: traceroute
    api: /api/isp/traceroute/{monitor_id}
    transform_rules: [isp_trace_transformation.rules]
    load: { table: Traces }
  - type: mtr
    api: /api/isp/traceroute/{monitor_id}
    transform_rules: [isp_mtr_transformation.rules]
    load: { table: Traces }

logs:
  - type: availability_log
    api: /api/reports/log_reports/{monitor_id}?date={today}
    transform_rules: [isp_logreport_transformation.rules]
    load: { table: Logs_availability }

events:
  - type: Infrastructure
    filter:
      expression: 'logtype="Infrastructure Events" and source="ISP"'
    api:
      path: /api/applog/search/{start_date}%20{start_time}/{end_date}%20{end_time}/{range}/desc?
      query_param: query
    pagination:
      page_size: 100
      start_index: 1
      max_pages: 20
    schedule:
      lookback: "24h"
      align_to: "now"
    transform_rules: [eventlogs_transformation.rules]
    load:
      table: Events
      upsert_keys: [EntityType, EntityID, EventType, EventId]

views:
  - name: isp_latest_health
    sql_template: |
      SELECT e.EntityID, e.EntityName, m.MetricName, m.CurrentValue
      FROM Entities e
      JOIN Metrics m ON m.EntityID = e.EntityID
      WHERE e.EntityType = :entity_type
      ORDER BY m.updated_at DESC
    params:
      entity_type: ISP
```

---

## Orchestration & Scheduling

* **Workflows** can pass explicit `start`/`end` to override the blueprint’s `lookback`.
* For heavy backfills, split windows (e.g., day-by-day) to avoid deep pagination. Re-run safely thanks to upsert keys.

---

## Validation

* Add a JSON-Schema for blueprints:

  * Required: `entity.entityType`, `entity.entityIdField`
  * Each source requires: `type`, `transform_rules`, `load.table`
  * Events require: `api.path` + `filter.expression`
* Lint for unknown fields and warn on missing `upsert` definitions.

---

## Edge Cases

* **Empty event pages** → stop pagination early.
* **Clock skew** → add ±1 min pad to `start` and `end` if provider timestamps are imprecise.
* **Large messages** → truncate `Message` but keep full `Raw`.
* **Rate limits** → allow per-source `throttle.ms` and retry with backoff.

---

## Deliverables Checklist

* [ ] `blueprints.csv` + per-entity YAMLs (incl. `events[]`)
* [ ] Blueprint schema validator
* [ ] Events fetcher:

  * [ ] time-window formatter (dd-mm-yyyy + hh:mm:ss)
  * [ ] range paginator (`start-end`)
  * [ ] query builder (`q=` param with URL encoding)
* [ ] Rule runner integration (reuse existing)
* [ ] Loader upsert semantics for `Events`
* [ ] UI hooks: preview/run via blueprint; view registration/materialization
* [ ] Tests:

  * [ ] paging stops on empty page
  * [ ] upsert dedupes correctly
  * [ ] lookback/schedule overrides work

