from pinecone import Pinecone, ServerlessSpec

def delete_index(api_key, index_name):
    pinecone = Pinecone(api_key=api_key)
    if index_name in pinecone.list_indexes().names():
        pinecone.delete_index(name=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")

def create_index(api_key, env, index_name="sample-movies", dim=4096):
    pinecone = Pinecone(api_key=api_key)
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=env
            )
        )
        print(f"Index '{index_name}' created with dimension {dim}.")
    else:
        print(f"Index '{index_name}' already exists.")

if __name__ == "__main__":
    api_key = "acb4f1cc-ab8c-4a96-a782-c67735cac9ae"
    env = "us-east-1"
    index_name = "sample-movies"
    
    delete_index(api_key, index_name)
    create_index(api_key, env, index_name)
