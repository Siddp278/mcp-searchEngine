# python mcp_server.py

import os
from typing import List
import requests
from bs4 import BeautifulSoup
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import models, QdrantClient

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Configuration constants
QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = "covid-faq" 
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
url = os.getenv("FIRECRAWL_URL")
api_key = os.getenv('FIRECRAWL_API_KEY')

mcp_server = FastMCP("MCP-RAG-app",
                     host=HOST,
                     port=PORT)

# logger.debug(f"MCP Server instantiated on host: {HOST} and port: {PORT}")



@mcp_server.tool()
def covid_faq_retrieval_tool(query: str) -> str:
    """
    Retrieve the most relevant documents from the Covid FAQ collection. 
    Use this tool when the user asks about covid related questions. 
    OR
    Says `I want the covid FAQ referred`
    
    Args:
        query (str): The user query to retrieve the most relevant documents.
        
    Returns:
        str: The most relevant documents retrieved from the vector DB.
    """
    # logger.debug(f"Running the covid_faq_retrieval_tool with query: {query}")
    # logger.info(f"Running the covid_faq_retrieval_tool with query: {query}")
    if not isinstance(query, str):
        logger.error("argument to covid_faq_retrieval_tool() is not a string")
        raise TypeError("Query must be a string.")
    

    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        trust_remote_code=True
    )
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=True)
    query_embedding = embed_model.get_query_embedding(query)
    # logger.debug("Got the query embeddings")

    # Search Qdrant for the most similar vectors
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        with_payload=True,
        limit=3,
    ).points

    if not search_result:
        # logger.info("No embeddings matched, empty response from the covid tool")
        return "I couldn't find a relevant answer in my knowledge base."

    else:
        # logger.info("Embeddings matched, context response from the covid tool returned")
        return " ".join([hit.payload["context"] for hit in search_result])



def crawl_and_extract_text(target_url: str) -> str:
    # logger.info(f"Running the crawl_and_extract_text on URL: {target_url}")
    # logger.debug(f"Running the crawl_and_extract_text on URL: {target_url}")
    try:
        r = requests.get(target_url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text[:600] 
    except Exception as e:
        # logger.error(f"Exception occured while parsing URL in crawl_and_extract_text: {e}")
        return f"Error fetching {target_url}: {e}"


@mcp_server.tool()
def firecrawl_web_search_tool(query: str) -> List[str]:
    """
    Search for information on a given topic using Firecrawl.
    Use this tool when the user asks a specific question not related to the Covid.

    Args:
        query (str): The user query to search for information.

    Returns:
        List[str]: A list of the most relevant web search results.
    """
    # logger.debug(f"Running the firecrawl_web_search_tool with query: {query}")
    # logger.info(f"Running the firecrawl_web_search_tool with query: {query}")
    
    if not isinstance(query, str):
        # logger.debug("argument to firecrawl_web_search_tool() is not a string")
        # logger.error("argument to firecrawl_web_search_tool() is not a string")
        raise TypeError("Query must be a string.")

    payload = {"query": query, "timeout": 60000}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        # logger.debug(f"Running request on URL to get crawled data")
        # logger.info(f"Running request on URL to get crawled data")

        response.raise_for_status()
        results = response.json().get("data", [])
    except requests.exceptions.RequestException as e:
        # logger.debug(f"Error connecting to Firecrawl API: {e}")
        # logger.error(f"Error connecting to Firecrawl API: {e}")
        results = []
    finally:
        extracted_result = []
        for item in results:
            page_url = item.get("url")

            if not page_url:
                continue

            extracted_text = crawl_and_extract_text(page_url)
            extracted_result.append(f"{extracted_text}...")

        # logger.debug(f"Getting the final result: {extracted_result}")
        return extracted_result if extracted_text else ["I could not find any related information, please check from your own training data"]
    

if __name__ == "__main__":
    # logger.info("Starting the MCP Server")
    # mcp_server.run(transport="sse")
    # mcp_server.run(transport="stdio")
    mcp_server.run(transport="streamable-http")