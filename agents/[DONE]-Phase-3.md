# TODO - Phase III: Azure OpenAI Support. 

## Summary
We are implementing an enhancement in Phase 3 to support AI assisted code / query generation. Later in the next iterations we will expand to use the AI agent to build Views, Workflows and Automations. 

## Implementation details
Here is the brief. In each section, whether it is tables, or Blueprints or Views, we ask user to write a proper JQ query or SQL query. Add some agents for each page which will be assiting the user's natural language query to proper structured JQ query. 

For example, fetch all entity names in table view is now given as ".data.monitors[].name"

Instead, the user can simply ask "entity names" after selecting the entitites table, and an agent automatically calls the AI backend to fetch the right command for this and gives it. 

Add a new toggle button "Code | Natural Language" in the query box in the Inspector view of each page which supports a query with a submit button. (there are 3 now, In the APIs page its - "Run JQ Query", in the Transformation Rules page, its "Preview Transformation", and in the Tables page its "Run SQLite Query" which have to be supported by the respective Agents - lets call them assistants) 

Add any suitable error handling, and other supporting functionalities for robustness. 

We will be implementing workflows and Automations in the next iteration. For now, give this option. 

## Agent SDK examples
Here is the documentation you require to add that support. Assume all config is loaded from .env file so .env.example has to be updated as well. 

Assume that OpenAI SDK (Agents SDK as well) is going to be included with support for consuming Azure resources. 

This is a sample agent sdk implementation. 

```agents-azure-openai-sdk-basic-deterministic.py
import asyncio

from pydantic import BaseModel

from agents import Agent, Runner, trace, set_default_openai_client

"""
This example demonstrates a deterministic flow, where each step is performed by an agent.
1. The first agent generates a story outline
2. We feed the outline into the second agent
3. The second agent checks if the outline is good quality and if it is a scifi story
4. If the outline is not good quality or not a scifi story, we stop here
5. If the outline is good quality and a scifi story, we feed the outline into the third agent
6. The third agent writes the story
"""

load_dotenv()

azure_openai_client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

openai_client = azure_openai_client
set_default_openai_client(openai_client)


story_outline_agent = Agent(
    name="story_outline_agent",
    instructions="Generate a very short story outline based on the user's input.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)


class OutlineCheckerOutput(BaseModel):
    good_quality: bool
    is_scifi: bool


outline_checker_agent = Agent(
    name="outline_checker_agent",
    instructions="Read the given story outline, and judge the quality. Also, determine if it is a scifi story.",
    output_type=OutlineCheckerOutput,
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)

story_agent = Agent(
    name="story_agent",
    instructions="Write a short story based on the given outline.",
    output_type=str,
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)


async def main():
    input_prompt = input("What kind of story do you want? ")

    # Ensure the entire workflow is a single trace
    with trace("Deterministic story flow"):
        # 1. Generate an outline
        outline_result = await Runner.run(
            story_outline_agent,
            input_prompt,
        )
        print("Outline generated")

        # 2. Check the outline
        outline_checker_result = await Runner.run(
            outline_checker_agent,
            outline_result.final_output,
        )

        # 3. Add a gate to stop if the outline is not good quality or not a scifi story
        assert isinstance(outline_checker_result.final_output, OutlineCheckerOutput)
        if not outline_checker_result.final_output.good_quality:
            print("Outline is not good quality, so we stop here.")
            exit(0)

        if not outline_checker_result.final_output.is_scifi:
            print("Outline is not a scifi story, so we stop here.")
            exit(0)

        print("Outline is good quality and a scifi story, so we continue to write the story.")

        # 4. Write the story
        story_result = await Runner.run(
            story_agent,
            outline_result.final_output,
        )
        print(f"Story: {story_result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
```

And another example of how agents are used 

```agents-with-tool-calls-banking-demo.py

@function_tool
def check_account_balance(account_id: str) -> float:
    """Check the balance of a bank account."""
    # This is a mock function - in a real application, this would query a database
    balances = {
        "1234": 5432.10,
        "5678": 10245.33,
        "9012": 750.25,
        "default": 1000.00
    }
    return balances.get(account_id, balances["default"])

# Banking-themed agents
general_agent = Agent(
    name="Banking Assistant",
    instructions="You are a helpful banking assistant. Be concise and professional.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
    tools=[check_account_balance],
)

```

You can also use agents-as-tools. 

```agents-as-tools.py



# Create specialized agents
spanish_agent = Agent(
    name="spanish_agent",
    instructions="You respond in Spanish. Always reply to the user's question in Spanish.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)

french_agent = Agent(
    name="french_agent",
    instructions="You respond in French. Always reply to the user's question in French.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)

italian_agent = Agent(
    name="italian_agent",
    instructions="You respond in Italian. Always reply to the user's question in Italian.",
    model=OpenAIChatCompletionsModel(
        model="gpt-4o",
        openai_client=openai_client
    ),
)

# Create orchestrator with conditional tools
orchestrator = Agent(
    name="orchestrator",
    instructions=(
        "You are a multilingual assistant. You use the tools given to you to respond to users. "
        "You must call ALL available tools to provide responses in different languages. "
        "You never respond in languages yourself, you always use the provided tools."
    ),
    tools=[
        spanish_agent.as_tool(
            tool_name="respond_spanish",
            tool_description="Respond to the user's question in Spanish",
            is_enabled=True,  # Always enabled
        ),
        french_agent.as_tool(
            tool_name="respond_french",
            tool_description="Respond to the user's question in French",
            is_enabled=french_spanish_enabled,
        ),
        italian_agent.as_tool(
            tool_name="respond_italian",
            tool_description="Respond to the user's question in Italian",
            is_enabled=european_enabled,
        ),
    ],
)
```

## Notes

Make reasonable assumptions, and generate the code for functions, agents and where to integrate them. The architecture should be extensible for future developments. I will verify by running locally and giving feedback once you have done your implementation. 

Once you have done, mark this file as [DONE] and update the AGENTS.md file about the progress. 