from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    return HuggingFaceEmbeddings(model_name=model_name)
