# python client.py new_server.py True

import asyncio
from typing import Optional, List
from contextlib import AsyncExitStack
import os
import json

from mcp.client.stdio import stdio_client
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv

load_dotenv()

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.token: str = os.getenv("MODEL_API_KEY")
        self.endpoint: str = os.getenv("MODEL_ENDPOINT")
        self.model_name: str = os.getenv("MODEL_DEPLOYMENT_NAME")
        self.context = []

        # API auth
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.token),
        )

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script
        """
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
    async def listing_tools(self) -> List[dict]:
        response = await self.session.list_tools()
        # tools -> name, description, inputSchema
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.inputSchema["properties"]
                }
            }
        } for tool in response.tools]


    async def process_query(self) -> str:
        tools = await self.listing_tools()

        response = await self.client.complete(
            messages=self.context,
            model=self.model_name,
            tools=tools,
            tool_choice="auto",
            temperature=1.0,
            max_tokens=1500,
            top_p=1
        )
        
        result = response.choices[0].message
        self.context.append(result)
        tool_msgs = []
        if result.tool_calls:
            for tc in (response.choices[0].message.tool_calls or []):
                args = json.loads(tc.function.arguments)
                result = await self.session.call_tool(tc.function.name, args)
                tool_msgs.append({
                    "role": "tool",
                    "content": result.content[0].text,
                    "tool_call_id": tc.id
                })

        # Continue conversation
        self.context.extend(tool_msgs)
        follow_up = await self.client.complete(
            messages=self.context,
            model=self.model_name,
            temperature=1.0,
            max_tokens=2500,
            top_p=1
        )

        result = follow_up.choices[0].message
        self.context.append(result)

        return result.content



    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        # try using prompt as a primitive from the server next time
        self.context = [
                        {
                            "role": "system",
                            "content": """
                                You are a COVID-specialist agent designed to provide accurate, clear, and concise information.

                                When a user asks a question that is clearly about COVID-19 — including topics such as symptoms, vaccines, transmission, treatments, guidelines, or even explicitly states "I want the Covid FAQ referred" — use your internal capabilities to retrieve the most relevant responses from the Covid FAQ collection.
                                This specialized knowledge base is designed to answer COVID-related queries using semantic understanding and retrieval from pre-indexed, trusted information.

                                If the user's question is not related to COVID-19, use your external browsing capability to search and summarize relevant results from the web.
                                This includes all other specific or factual questions outside the Covid domain, where you should attempt to fetch relevant and concise answers by crawling and analyzing web content.

                                If the question is unclear, assume Covid relevance only when there is a clear signal — such as direct mention of Covid or pandemic-related terms — and otherwise default to external search.

                                You must choose the most appropriate tool available to assist with the query, based on this logic. Do not guess or fabricate information when tools are available to retrieve accurate data.
                            """,
                        }
        ]
        try:
            while True:
                query = input("\nQuery: ").strip()
                
                if query.lower() == 'quit':
                    break
  
                self.context.append({
                            "role": "user",
                            "content": query,
                        })
                response = await self.process_query()
                print("\n Final result from the LLM: " + response)
                    
        except Exception as e:
            print(f"\nError in the chat_loop: {e}")
        finally:
            await self.cleanup()

    
    async def get_context(self) -> None:
        print(self.context)
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 3:
        print("Usage: python client.py <path_to_server_script> <bool_for_getting_history>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
        if sys.argv[2] and sys.argv[2]=="True":
            await client.get_context()
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    import sys
    asyncio.run(main())