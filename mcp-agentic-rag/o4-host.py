# Import necessary libraries
import os, time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import McpTool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the MCP server configuration from environment variables
mcp_server_url = os.getenv("MCP_SERVER_URL", "https://gitmcp.io/Azure/azure-rest-api-specs")
mcp_server_label = os.getenv("MCP_SERVER_LABEL", "github")
project_endpoint = os.getenv("PROJECT_ENDPOINT")

project_client = AIProjectClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
         exclude_environment_credential=True,
         exclude_managed_identity_credential=True
     ),
)

# setting up the tool/resources/prompt list
mcp_tool = McpTool(
    server_label=mcp_server_label,
    server_url=mcp_server_url,
    allowed_tools=["python_faq_retrieval_tool", "firecrawl_web_search_tool"]
)

# Creating an agent
agents_client = project_client.agents
agent = agents_client.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="covid-specialist-agent",
        instructions="""
            You are a covid specialist agent who provides accurate information regarding covid queries.
            Additionally, you have the capacity to asnwer precise covid related questions with the help of MCP tools for Covid FAQ.
            Lastly, if you don't know the answer to questions unrelated to covid, you have the MCP tool to crawl the web and get answers.
        """,
        tools=mcp_tool.definitions,
    )

# agent = agents_client.get_agent("asst_g9bOuyDvGaSlEDlAhLsg7GWl")
        

# Starting the message initiation on a new thread
# Create a thread for communication
thread = agents_client.threads.create()
print(f"Created thread, ID: {thread.id}")

# Create a message for the thread
message = agents_client.messages.create(
    thread_id=thread.id,
    role="user",
    content="Please summarize the Azure REST API specifications Readme",
)
print(f"Created message, ID: {message.id}")


if True:
    # Create and automatically process the run, handling tool calls internally
    run = project_client.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent.id)
    print(f"Run finished with status: {run.status}")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Retrieve the steps taken during the run for analysis
    run_steps = project_client.agents.run_steps.list(thread_id=thread.id, run_id=run.id)

    # # Loop through each step to display information
    # for step in run_steps:
    #     print(f"Step {step['id']} status: {step['status']}")

    #     tool_calls = step.get("step_details", {}).get("tool_calls", [])
    #     for call in tool_calls:
    #         print(f"  Tool Call ID: {call.get('id')}")
    #         print(f"  Type: {call.get('type')}")
    #         function_details = call.get("function", {})
    #         if function_details:
    #             print(f"  Function name: {function_details.get('name')}")
    #             print(f" function output: {function_details.get('output')}")

    #     print()


    # # Delete the agent resource to clean up
    # project_client.agents.delete_agent(agent.id)
    # print("Deleted agent")

    # # Fetch and log all messages exchanged during the conversation thread
    # messages = project_client.agents.messages.list(thread_id=thread.id)
    # for msg in messages:
    #     print(f"Message ID: {msg.id}, Role: {msg.role}, Content: {msg.content}")