"""
For the Conversational retrieval chain, we have to get the retriever fetch documents relevant 
not only to the user input but also to the chat history. Therefore, the retriever needs to
 have a query based not only on the user input but also on the relevant documents from the 
 chat history. In order to do this, we provide the LLM with the chat history and user input 
 and ask it to derive a search query for the retriever to fetch the relevant data from the 
 vector store.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_openai import AzureChatOpenAI
from langchain_community.retrievers import AzureAISearchRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
import os



llm_client = AzureChatOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
    deployment_name=deployment,
    max_tokens=1500
)

retriever = AzureAISearchRetriever(
    content_key="content", top_k=3, index_name="covid-vector-index"
)


# The MessagesPlaceholder is a prompt template that assumes that the variable provided 
# to it as input is a list of messages
prompt_search_query = ChatPromptTemplate.from_messages([
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
("user","Given the above conversation, generate a search query to look up to get information relevant to the conversation")
])

# use chat_history+current_user_prompt -> new_prompt/query to send it to retriever
# Create a chain that takes conversation history and returns documents.
retriever_chain = create_history_aware_retriever(llm_client, retriever, prompt_search_query)

prompt_get_answer = ChatPromptTemplate.from_messages([
("system", "Answer the user's questions based on the below context:\\n\\n{context}"),
MessagesPlaceholder(variable_name="chat_history"),
("user","{input}"),
])

# Restructure and will send the prompt to the llm
document_chain=create_stuff_documents_chain(llm_client,prompt_get_answer)

# I have retriever_chain to retrieve al relevant context and document_chain to get results from llm - combining them now
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can covid-19 be deadly?"), AIMessage(content="Yes")]
response = retrieval_chain.invoke({
"chat_history":chat_history,
"input":"Why?"
})
print(response['answer'])