from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
import os


llm_client = AzureChatOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    deployment_name=deployment,
    max_tokens=1500
)

embeddings = AzureOpenAIEmbeddings(
    model=azure_deployment,
    azure_endpoint=azure_endpoint,
    openai_api_key=azure_openai_api_key,
)

"""
# If I want to create a new vector store in Azure Search
vector_store = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    index_name="covid-index",
)

# Getting unstructured textual data from the website
loader = WebBaseLoader("https://www.dasa.org")
docs = loader.load()

# Chunking the data so that we can create embeddings for it
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
"""


retriever = AzureAISearchRetriever(
    content_key="content", top_k=3, index_name="covid-vector-index"
)

# returns a list of document objects with meta data and page_content
ans = retriever.invoke("does the president have a plan for covid-19?")
# print(ans)

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the context provided.

Context: {context}

Question: {question}"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Runnable pass through is LangChain’s way of saying “just pass this input through untouched, but still behave like a proper Runnable.”
chain = (
    # Since context is dynamically retrieved, we need to pass it in the chain, rahter than defining it while in invoke
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_client
    | StrOutputParser()
)

result = chain.invoke({"question": "does the president have a plan for covid-19?"})
print(result)