import os
from langchain_community.vectorstores import FAISS
from rag.data_loader import load_html_documents, split_documents

FAISS_DIR = "faiss_index"
DATA_DIR = "data_raw"

def create_faiss_vectorstore(embedding_model):
    print("Đang đọc các file HTML từ thư mục:", DATA_DIR)
    documents = load_html_documents(DATA_DIR)

    if not documents:
        print("Không tìm thấy file HTML hoặc tất cả file rỗng!")
        return

    print(f"Đã nạp {len(documents)} tài liệu. Đang chia chunk và tính embedding...")
    chunks = split_documents(documents)
    print(f" Đã tạo {len(chunks)} đoạn văn bản sau khi chia chunk.")

    vectorstore = FAISS.from_documents(chunks, embedding_model)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"Đã lưu FAISS vectorstore tại: {FAISS_DIR}")

def load_faiss_vectorstore(embedding_model):
    if not os.path.exists(FAISS_DIR):
        print("Vectorstore chưa tồn tại. Hãy chạy tạo FAISS trước.")
        return None
    print(" Đang tải FAISS vectorstore...")
    return FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)
