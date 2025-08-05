from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import json
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

load_dotenv()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
    env=None,
)


def convert_to_llm_tool(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"]
            }
        }
    }


def get_llm_client():
    token = os.getenv("MODEL_API_KEY")
    endpoint = os.getenv("MODEL_ENDPOINT")
    model_name = os.getenv("MODEL_DEPLOYMENT_NAME")

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token),
    )

    return client, model_name


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            functions = [convert_to_llm_tool(tool) for tool in tools.tools]

            client, model_name = get_llm_client()

            async def call_llm(prompt):

                print("Calling LLM with prompt and tool definitions...")
                response = client.complete(
                    messages=[
                        {
                            "role": "system",
                            "content": """
                                You are a covid specialist agent who provides accurate information regarding covid queries.
                                Additionally, you have the capacity to answer precise covid-related questions with the help of MCP tools for Covid FAQ.
                                Lastly, if you don't know the answer to questions unrelated to covid, you have the MCP tool to crawl the web and get answers.
                            """,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    model=model_name,
                    tools=functions,
                    temperature=1.0,
                    max_tokens=2500,
                    top_p=1
                )

                response_msg = response.choices[0].message
                functions_to_call = []

                if response_msg.tool_calls:
                    for tool_call in response_msg.tool_calls:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        functions_to_call.append({
                            "id": tool_call.id,
                            "name": name,
                            "args": args
                        })

                tool_results = []
                for f in functions_to_call:
                    print(f"Invoking tool: {f['name']} with args: {f['args']}")
                    result = await session.call_tool(f["name"], arguments=f["args"])
                    print("Tool result:", result.content[0].text[:100])
                    tool_results.append({
                        "tool_call_id": f["id"],
                        "output": result.content[0].text
                    })

                print("Sending tool output back to LLM for final response...")
                messages = messages=[
                        {
                            "role": "system",
                            "content": """
                                You are a covid specialist agent who provides accurate information regarding covid queries.
                                Additionally, you have the capacity to answer precise covid-related questions with the help of MCP tools for Covid FAQ.
                                Lastly, if you don't know the answer to questions unrelated to covid, you have the MCP tool to crawl the web and get answers.
                            """,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                        response_msg,
                        # Adding/injecting multiple llm-user conversations here, add all new dictionaries here
                        *[
                            {
                                "role": "tool",
                                "tool_call_id": r["tool_call_id"],
                                "content": r["output"]
                            }
                            for r in tool_results
                        ],
                    ]
                
                final_response = client.complete(
                    messages=messages,
                    model=model_name,
                    tools=functions,
                    temperature=1.0,
                    max_tokens=2500,
                    top_p=1
                )

                print("\nFinal Answer from LLM:")
                print(final_response.choices[0].message)

            # prompt = "When did covid emerge as an epidemic? I want the covid FAQ referred"
            prompt = "Who won the last FIFA world cup?"
            await call_llm(prompt)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
