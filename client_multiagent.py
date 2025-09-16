# python client_multiagent.py

import os
import uuid
import asyncio
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt
from langgraph.graph import StateGraph
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]

async def main():
    client = MultiServerMCPClient(
        {
            "MCP-RAG-app": {
                "url": os.getenv("MCP_SERVER_URL"),  # Run manually, since transport is http
                "transport": "streamable_http"
            }
        }
    )

    # Creating thread for the memory
    thread_id = uuid.uuid4()
    thread_list = [thread_id]
    config = {"configurable": {"thread_id": thread_id}}

    tools = await client.get_tools()
    # print(tools)

    model = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT"),
        api_version="2024-12-01-preview",
        temperature=0.7,
        model_name="gpt-4o",
        azure_endpoint=os.getenv("AZURE_OPENAI_SERVICE")
    )

    # # Binding tools to my LLM
    # model_with_tools = model.bind_tools(tools)

    # async def model_with_tools_node(input_state: State) -> State:
    #     res = await model_with_tools.ainvoke({"messages": input_state["messages"]})
    #     return res

    # # Building the graph 
    # builder = StateGraph(State)
    # builder.add_node("LLMWithTools", model_with_tools_node)
    # builder.add_node("tools", ToolNode(tools))

    # builder.add_edge(START, "LLMWithTools")
    # builder.add_conditional_edges("LLMWithTools", tools_condition)
    # builder.add_edge("tools", "LLMWithTools")
    # builder.add_edge("LLMWithTools", END)

    # # Short term memory
    # memory = MemorySaver()
    # graph = builder.compile(checkpointer=memory)

    # print(result)
    # result = await graph.ainvoke({"messages": [{"role": "user", "content": "Explain in detail why Formula One is so famous?"}]}, config=config)

    memory = MemorySaver()
    agent = create_react_agent(model, tools, checkpointer=memory) # Use this agent with other agents in a supervision mode
    while True:
        query = input("Write your question here: ")
        if query == "quit":
            break

        elif query == "new chat":
            thread_id = uuid.uuid4()
            thread_list.append(thread_id)
            config = {"configurable": {"thread_id": thread_id}}
            query = input("Write your new question here: ")

        result = await agent.ainvoke({"messages": [{"role": "user", "content": query}]}, config=config)

        print("\nFinal response:")
        for msg in result["messages"]:
            print(f"{msg.type.upper()}: {msg.content}\n")

    # # Optional: delete memory for this thread
    for ids in thread_list:
        memory.delete_thread(thread_id=ids)

if __name__ == "__main__":
    asyncio.run(main())