from rag.embeddings import get_embedding_model
from rag.vector_store import create_hybrid_vectorstore
from shared.log import setup_logging


if __name__ == "__main__":
    setup_logging()
    create_hybrid_vectorstore(get_embedding_model())
