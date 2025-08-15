from rag.embeddings import get_embedding_model
from rag.vector_store import create_faiss_vectorstore

if __name__ == "__main__":
    embedding_model = get_embedding_model()
    create_faiss_vectorstore(embedding_model)
