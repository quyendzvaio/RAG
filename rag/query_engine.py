def search_query(question: str, vectorstore, top_k: int = 3):
    print(f"Câu hỏi: {question}")
    try:
        docs = vectorstore.similarity_search(question, k=top_k)
    except Exception as e:
        print("Lỗi khi tìm kiếm tương đồng:", str(e))
        return []
    return docs
