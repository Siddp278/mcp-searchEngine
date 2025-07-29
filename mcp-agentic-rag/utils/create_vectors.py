# Utilize llama-index and qdrant to develop vectors out of your data.
"""
qdrant setup with docker - 
shell command - docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
"""


from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import models, QdrantClient
from tqdm import tqdm
import uuid
import re
import unicodedata
import gc


# running qdrant in local mode suitable for experiments
qdrant_url: str = "http://localhost:6333"
collection_name: str = "covid-faq"
embed_model_name: str = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 16  # Tune this based on your memory/network


# Setup outside the loops
splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=30)

embed_model = HuggingFaceEmbedding(
    model_name=embed_model_name,
    trust_remote_code=True
)
vector_dim = len(embed_model.get_text_embedding("test"))

client = QdrantClient(url=qdrant_url, prefer_grpc=True)

def clean_text(text: str) -> str:
    text = re.sub(r'/uni[0-9A-Fa-f]+', ' ', text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'(\b\w{2,})\s+(\w{2,}\b)', r'\1\2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Prepare payload: list of dict with cleaned text and source
reader = SimpleDirectoryReader(
    input_dir="./../../covid_data",
    num_files_limit=2)
docs = reader.load_data()

payload = [{"document": clean_text(doc.text), "source": getattr(doc, "file_path", "No file path")} for doc in docs]

all_points = []

for entry in tqdm(payload, desc="Processing documents"):
    doc = Document(text=entry["document"], metadata={"source": entry["source"]})
    chunks = splitter.split_text(doc.text)

    try:
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists. Skipping creation.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dim,
                distance=models.Distance.DOT
            )
        )

    # Process chunks in small batches - memory constraints lol
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        embeddings = embed_model.get_text_embedding_batch(batch_chunks, show_progress_bar=False)

        all_points = []
        for chunk_text, embedding in zip(batch_chunks, embeddings):
            all_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"context": chunk_text, "source": entry["source"]},
                )
            )

        if len(all_points) >= BATCH_SIZE:
            client.upload_points(
                collection_name=collection_name,
                points=all_points,
                wait=True
            )
            all_points = []

        # Upload remaining points
        if all_points:
            client.upload_points(
                collection_name=collection_name,
                points=all_points,
                wait=True
            )
        
        # Now safe to clean up
        del batch_chunks, embeddings, all_points
        gc.collect()

# Optimize collection after upload
client.update_collection(
    collection_name=collection_name,
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
)


