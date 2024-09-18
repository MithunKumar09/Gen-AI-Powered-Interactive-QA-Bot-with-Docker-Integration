#Backend (RAG Model)/models/rag_model.py
import datetime
import os
import numpy as np
import cohere
from pinecone import Pinecone, ServerlessSpec

class RAGModel:
    def __init__(self, cohere_api_key, pinecone_api_key, pinecone_env):
        self.cohere_api_key = cohere_api_key
        self.cohere_client = cohere.Client(cohere_api_key)
        
        # Initialize Pinecone
        self.pinecone_client = Pinecone(api_key=pinecone_api_key)
        index_name = 'sample-movies'
        dimension = 4096  # Ensure this matches Cohere embedding dimension

        # Check if the index exists, if not create it
        if index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=pinecone_env
                )
            )
        self.index = self.pinecone_client.Index(index_name)

    def embed_documents(self, documents):
        embeddings = []
        for doc in documents:
            response = self.cohere_client.embed(texts=[doc])
            embedding = response.embeddings[0]
            if len(embedding) != 4096:
                raise ValueError(f"Embedding dimension {len(embedding)} does not match index dimension 4096.")
            embedding = self.normalize_embedding(embedding)
            embeddings.append(embedding)
        return embeddings

    def retrieve_relevant_docs(self, query_embedding):
        query_embedding = self.normalize_embedding(query_embedding).tolist()

        try:
            response = self.index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            return response['matches']
        except Exception as e:
            return {"error": str(e)}

    def generate_answer(self, context, question):
        combined_text = f"Context: {context}\nQuestion: {question}"
        response = self.cohere_client.generate(
            model='command-xlarge-nightly',
            prompt=combined_text,
            max_tokens=500,
        )
        answer = response.generations[0].text.strip()

        # Save the answer to a file
        self.save_answer_to_file(question, answer)
        
        return answer

    def query(self, question):
        query_embedding = self.cohere_client.embed(texts=[question]).embeddings[0]
        matches = self.retrieve_relevant_docs(query_embedding)
        return matches

    def normalize_embedding(self, embedding):
        embedding = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def save_answer_to_file(self, question, answer):
        # Define the path to save the file
        data_dir = 'data'
        
        # Create the /data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Create a unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"answer_{timestamp}.txt"
        filepath = os.path.join(data_dir, filename)

        # Write the answer to the file
        with open(filepath, 'w') as file:
            file.write(f"Question: {question}\n")
            file.write(f"Answer: {answer}\n")