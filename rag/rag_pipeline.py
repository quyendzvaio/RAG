import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from rag.embeddings import get_embedding_model
from rag.vector_store import create_faiss_vectorstore, load_faiss_vectorstore
from rag.query_engine import search_query
import gradio as gr

def build_gradio_interface():
    embedding_model = get_embedding_model()
    vectorstore = load_faiss_vectorstore(embedding_model)

    if vectorstore is None:
        raise ValueError("Vectorstore chưa tồn tại. Vui lòng chạy tạo FAISS trước.")

    def query_interface(user_query):
        if not user_query.strip():
            return "Bạn chưa nhập câu hỏi."
        full_query = f"Theo quy định pháp luật Việt Nam, {user_query.strip()}"
        docs = search_query(full_query, vectorstore)
        if not docs:
            return "Không tìm thấy kết quả phù hợp."

        context = "\n\n".join(doc.page_content for doc in docs)
        final_answer = query_with_openai(context, user_query)
        return final_answer

    iface = gr.Interface(
        fn=query_interface,
        inputs=gr.Textbox(label="Nhập câu hỏi pháp lý"),
        outputs=gr.Textbox(label="Trả lời từ GPT-3.5"),
        title="Hỏi Đáp Luật Việt Nam bằng",
        description="Hệ thống tìm kiếm tài liệu pháp luật và sinh câu trả lời tự động bằng GPT-3.5"
    )
    return iface
def query_with_openai(context: str, user_query: str) -> str:
    prompt = f"""
Bạn là chuyên gia pháp luật Việt Nam. Dưới đây là thông tin tham khảo từ tài liệu pháp luật:

{context}

Dựa trên thông tin này, hãy trả lời câu hỏi sau một cách rõ ràng, ngắn gọn, chính xác:

Câu hỏi: {user_query}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Bạn là một luật sư am hiểu pháp luật Việt Nam."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi gọi OpenAI API: {str(e)}"
    


def main():
    iface = build_gradio_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()
