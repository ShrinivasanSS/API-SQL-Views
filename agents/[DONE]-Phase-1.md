# Project Specifications 

The project tries to create an observability database that can be used to construct Automation workflows. 

To do that the project is split into 4 phases. 
Phase I - API data to Observability Tables.
Phase II - Observability Views.
Phase III - Observability workflows. 
Phase IV - Automation webhooks. 

## Technical requirements
This is a PoC project, so the tech stack will be 
- Python 
- Flask for UI and 
- SQLite for backend

## Backend
The project is built on Python, with backend as an SQLite database, which will have the following Observability tables. The data in the tables are transient and will be cleared regularly through scripts. 

### Observability Tables
There are 2 types of tables, 
* Preseet Tables
* Custom Extension Tables. 

Preset Tables are the ones which are crucial for mapping all the data from different APIs to the observability product. 

The following are considered crucial for the Preset Tables. 
- Entities
- Metrics
- Events
- Logs
- Traces
- Configurations

Apart from that we will have Custom tables. 

### Observability Concepts and Table mapping

You can find the API data in the examples/inputs/ folder and the respective table needed in examples/outputs/ folder.

These are limited, focused tables - for Phase 1 of the project. Assume that the examples/transformation_rules/ folder contains the rules for transformation and will be updated regularly to match the requirements for new entity types.

The Transformed data will be loaded on the SQLite db, flushing any existing data and be readily available for querying. 

In the project's Phase 2 - we will add custom extensions, to map custom table structure to be available for querying. 


## Frontend 

You need to create a python flask based application that has the following view. 

(assume minimum viewport of 1280x720 )

Left Pane  (20% screen) |      Middle Pane (60% screen)      | Right Pane (20% screen)

Left Pane options 
- Credentials
- APIs 
- Transformation Rules
- Tables
- Views (Saved queries)
- Workflows
- Automations 


Middle pane should adjust the views based on the selection of the left pane. 

- Credentials View
(shows the OAuth credentials - client_id, client_secret, refresh_token, api_domain. By default a demo credential is there loaded from the .env file)

- APIs view
(loads the API data from cached har files with option to refresh data. Uses Basic OAuth credentials, refreshes the Access token once per hour to hit the APIs and refresh the cache)
A list pane, showing all the API urls, response code, and partial view of the response json in the subsequent line with block quote and a "Copy Response" to copy the full content of the response. 

The corresponding right pane has a text box and a result box, where the selected API could be queried using JQ. Any valid JQ query run against the API will get a result that can be viewed there.

- Transformation Rules View
(loads the rules available and allows to create new rules to transform an API to a table/csv)
A list pane of the existing rules. An edit/delete option. And finally a Preview option by running the rule against an API, and clicking Save will create a new table of one of the pre-defined types.

The corresponding right pane has a text box and a result box, where the user can do arbitary pandas dataframe query to view what is the output for the loaded API. None of the changes are persisted.

- Tables View
(list of tables created after transformation, grouped by table type.)
A list pane, of all the tables. A view or delete option. 

The corresponding right pane has a text box and a result box, where the user can run arbitary SQLite queries against the selected table. Only SELECT queries are allowed. 

- Views (Saved queries)

The tables created in the above step are used to create the view, by combining multiple tables. And the result can be viewed here. With an option to save, the full result is saved. 

- Workflows and Automations 

Can be shown work in progress for now. 