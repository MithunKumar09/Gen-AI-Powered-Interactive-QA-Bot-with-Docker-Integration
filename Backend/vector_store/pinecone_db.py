#Backend (RAG Model)/vector_store/pinecone_db.py
import os
from pinecone import Pinecone, ServerlessSpec
import numpy as np

def create_index(api_key, env, index_name="sample-movies", dim=4096):
    pinecone_client = Pinecone(api_key=api_key)
    
    if index_name not in pinecone_client.list_indexes().names():
        pinecone_client.create_index(
            name=index_name,
            dimension=dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=env
            )
        )

def add_documents_to_index(index_name, documents, embeddings):
    pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pinecone_client.Index(index_name)
    for i, emb in enumerate(embeddings):
        emb = np.array(emb).astype(np.float32).tolist()  # Ensure embedding is a list of floats
        index.upsert([(f"doc_{i}", emb, {"text": documents[i]})])

def query_index(index_name, query_embedding, top_k=5):
    pinecone_client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pinecone_client.Index(index_name)
    
    # Ensure the query_embedding is correctly formatted as a list
    query_embedding = np.array(query_embedding).astype(np.float32).tolist()
    
    response = index.query(queries=[query_embedding], top_k=top_k, include_metadata=True)
    return response['matches']
