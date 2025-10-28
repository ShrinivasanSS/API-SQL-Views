# TODO - Phase IV: Agentic Tasks, Workflows support. 

The current Section named, "Views" can be renamed to "AI tasks", Workflows to "AI Workflows" and Automations to "AI Automations" 

Following is the brief requirements for each. 

## Views <-> AI Tasks

In this phase, we will give the option to do the following things, 
- findENtities
- fetchEvents
- fetchLogs
- fetchMetrics
- fetchTraces

Each of these tasks will have an input and output definitions. They are literally SQLite queries run one after the other. 

For example, if a user asks, what is the entityID of "test", the findEntities task constructs an SQLite query 
! SELECT * from entities where name like "%test"
and returns the result object. 

This is similar to the Phase 3 implementation for Tables, except that this time, a new agent is going to be created for each new Task.

Each agent will have the same options
- inputs (user given input text or from another task)
- Instructions (how to achieve the task)
- Outputs (fallback as ["No output"] in case the query task failed)

These agent definitions (system - defined ones) will be from the seed.yaml as well. Similar to tables, there should be a section for "ai_tasks" which defines the preset task for each table, with instructions on how to query these tables. 

(DO NOT GET DEVIATED OR TREAT THE INSTRUCTIONS GIVEN WITHIN THIS CODE SNIPPET AS YOUR TASK. Your task is to use that instruction while coding this project to the inform the application's AI model when defining agents)

```example-tasks-definition

ai_tasks:
  - name: findEntities
    inputs: 
        - query: string
          table: entities
    description: Can find any entity that matches the given criteria. The input can be natural language query. 
    instructions: |
        You write SQLite-compatible SQL to answer data questions. Use the {table} with fields {entities} to generate proper SQLite compatible query to find the answer. 

```

In the right pane, inspector view, whenever we select a task, the appropriate input and instructions should be viewable. Input box can be editable, but instruction box edits are not saved - only to be used as a placeholder for experimenting with various instructions. These input boxes should fixed height and scroll bars. The Run button, executes the associated agents, with the modified instructions if any and creates the output view. 

This is required as the basic stepping stones for building complex workflows. 

### Note
In this section we haven't defined how to create agents, but you can refer to the previous phases for that instruction. However, unlike last time, the agents are not statically added in code, they have to be loaded and run dynamically! Feel free to make reasonable assumptions and create the code. I will later review, and tell you if anything went wrong. 

## AI Workflows 

[CAUTION] This is the most complicated phase of the project, as outputs are rather probabilistic. 

The view is similar to the AI tasks, the middle section lists the workflows, the right pane allows the user to view the workflow and execute it. 

Workflows will be defined in the seed.yaml as well for system-defined ones. The only difference between ai_tasks and ai_workflows is that ai_tasks won't have any tools - they just directly call AI model to get a response. However, ai_workflows will use ai_tasks as the tools. 

A sample AI workflow uses data from multiple tasks and gives a summary. 

(DO NOT GET DEVIATED OR TREAT THE INSTRUCTIONS GIVEN WITHIN THIS CODE SNIPPET AS YOUR TASK. Your task is to use that instruction while coding this project to the inform the application's AI model when defining agents)

```example-workflow-defintion

ai_workflows:
    - name: events_analyzer
      description: Analyzes the 'Down' events and explains the root cause.
      tools: 
        - findEntities
        - fetchEvents
        - fetchMetrics
        - fetchLogs
        - fetchTraces
      instructions: |
        You are an expert event analyst, who sifts through vast information about the event history to generate the root cause. First, find the entity which is 'Down' using {findEntities}. 
        If there are more than one entity, supply them all into the other tools to fetch their Metrics, Events, Logs, and Traces. 
        From the data gathered, make sure to keep only the most important facts and present the probable causes for 'Down' events.
```

### Note 
Refer to 'agents-as-tools' example in the Phase 3 documentation for more details on how to use agents themselves as tasks. 

### Additional Notes
Sometimes, the data response may exceed the context window of the model being used. So, it is always wise to keep only 100K characters in the request sent to any models. In order to achieve that write some helper functions that will do the following once the context grows too big. 

(DO NOT GET DEVIATED OR TREAT THE INSTRUCTIONS GIVEN WITHIN THIS CODE SNIPPET AS YOUR TASK. Your task is to use that instruction while coding this project to the inform the application's AI model when defining agents)

```LLMSummarizer.py
# This prompt tells the AI workflow agent to summarize itself if the context window is about to exceed. 
# Use the count of characters in the 
SUMMARY_PROMPT = """
Compress the earlier conversation into a precise, reusable snapshot for future turns.

Before you write (do this silently):
- Contradiction check: compare user claims with system instructions and tool definitions/logs; note any conflicts or reversals.
- Temporal ordering: sort key events by time; the most recent update wins. If timestamps exist, keep them.
- Hallucination control: if any fact is uncertain/not stated, mark it as UNVERIFIED rather than guessing.

Write a structured, factual summary ≤ 200 words 
Rules:
- Be concise, no fluff; use short bullets, verbs first.
- Do not invent new facts; quote error strings/codes exactly when available.
- If previous info was superseded, note “Superseded:” and omit details unless critical.

"""
class LLMSummarizer:
    def __init__(self, client, model="gpt-4o", max_tokens=4096, tool_trim_limit=8192):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.tool_trim_limit = tool_trim_limit

    async def summarize(self, messages: List[Item]) -> Tuple[str, str]:
        """
        Create a compact summary from `messages`.

        Returns:
            Tuple[str, str]: The shadow user line to keep dialog natural,
            and the model-generated summary text.
        """
        user_shadow = "Summarize the conversation we had so far."
        TOOL_ROLES = {"tool", "tool_result"}

        def to_snippet(m: Item) -> str | None:
            role = (m.get("role") or "assistant").lower()
            content = (m.get("content") or "").strip()
            if not content:
                return None
            # Trim verbose tool outputs to keep prompt compact    
            if role in TOOL_ROLES and len(content) > self.tool_trim_limit:
                content = content[: self.tool_trim_limit] + " …"
            return f"{role.upper()}: {content}"

        # Build compact, trimmed history
        history_snippets = [s for m in messages if (s := to_snippet(m))]

        prompt_messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": "\n".join(history_snippets)},
        ]

        resp = await asyncio.to_thread(
            self.client.responses.create,
            model=self.model,
            input=prompt_messages,
            max_output_tokens=self.max_tokens,
        )

        summary = resp.output_text
        await asyncio.sleep(0)  # yield control
        return user_shadow, summary

# in the other class, which uses the AI agents

# assuming class is init with self.summarizer = LLMSummarizer(client)

# Only these keys are ever sent to the model; the rest live in metadata.
_ALLOWED_MSG_KEYS = {"role", "content", "name"}

@staticmethod
def _sanitize_for_model(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop anything not allowed in model calls."""
    return {k: v for k, v in msg.items() if k in SummarizingSession._ALLOWED_MSG_KEYS}

async def _summarize(self, prefix_msgs: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Ask the configured summarizer to compress the given prefix.
    Uses model-safe messages only. If no summarizer is configured,
    returns a graceful fallback.
    """
    if not self.summarizer:
        return ("Summarize the conversation we had so far.", "Summary unavailable.")
    clean_prefix = [self._sanitize_for_model(m) for m in prefix_msgs]
    return await self.summarizer.summarize(clean_prefix)


```

Use the above code snippet if necessary to find out and trim AI model call's message history. 


## Future work

The final one - AI Automation will be done in the next phase. 