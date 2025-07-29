from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import models, QdrantClient


qdrant_url: str = "http://localhost:6333"
collection_name: str = "covid-faq"
embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5"

query = """
What is the evolution for covid virus as understood till now?
"""

embed_model = HuggingFaceEmbedding(
    model_name=embed_model_name,
    trust_remote_code=True
)
vector_dim = len(embed_model.get_text_embedding("test"))

client = QdrantClient(url=qdrant_url, prefer_grpc=True)

query_embedding = embed_model.get_query_embedding(query)

# Search Qdrant for the most similar vectors
search_result = client.search(
    collection_name=collection_name,
    query_vector=query_embedding,
    limit=5, #top_k
    score_threshold=0.5
)

if not search_result:
    print("I couldn't find a relevant answer in my knowledge base.")

relevant_contexts = [
    hit.payload["context"] for hit in search_result
]
print(relevant_contexts)