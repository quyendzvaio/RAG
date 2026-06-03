import os

import gradio as gr

from rag.embeddings import get_embedding_model
from rag.query_engine import hybrid_search
from rag.settings import deepseek_settings
from rag.vector_store import load_hybrid_vectorstore


def _format_result(index, ranked_doc):
    doc = ranked_doc.document
    metadata = doc.metadata
    urls = metadata.get("urls") or []
    url_text = "\n".join(f"URL: {url}" for url in urls[:2])
    legal_header = " | ".join(
        value
        for value in (
            metadata.get("article_code", ""),
            metadata.get("doc_type", ""),
            metadata.get("effective_date", ""),
            metadata.get("source", ""),
        )
        if value
    )
    ranks = []
    if ranked_doc.vector_rank:
        ranks.append(f"vector #{ranked_doc.vector_rank}")
    if ranked_doc.bm25_rank:
        ranks.append(f"bm25 #{ranked_doc.bm25_rank}")
    rank_text = ", ".join(ranks)

    return (
        f"[{index}] {legal_header}\n"
        f"Chuong: {metadata.get('chapter', '')} {metadata.get('chapter_title', '')}\n"
        f"Ghi chu: {metadata.get('legal_note', '')}\n"
        f"Diem: {ranked_doc.score:.4f}"
        + (f" ({rank_text})" if rank_text else "")
        + "\n"
        + (f"{url_text}\n" if url_text else "")
        + f"Noi dung:\n{doc.page_content[:1200]}\n"
        "---\n"
    )


def build_gradio_interface():
    llm_settings = deepseek_settings()
    embedding_model = get_embedding_model()
    store = load_hybrid_vectorstore(embedding_model)

    if store is None:
        raise ValueError("Hybrid index chua ton tai. Vui long chay: python create_faiss.py")

    def query_interface(user_query, top_k, candidate_k):
        if not user_query.strip():
            return "Ban chua nhap cau hoi."
        try:
            results = hybrid_search(user_query, store, top_k=int(top_k), candidate_k=int(candidate_k))
        except RuntimeError as exc:
            return f"Loi truy xuat: {exc}"
        if not results:
            return "Khong tim thay ket qua phu hop."
        return "".join(_format_result(index, item) for index, item in enumerate(results, 1))

    llm_status = (
        f"DeepSeek LLM da cau hinh: {llm_settings.model}"
        if llm_settings.configured
        else "DeepSeek LLM chua cau hinh API key; hien UI dang tra ket qua retrieval."
    )

    iface = gr.Interface(
        fn=query_interface,
        inputs=[
            gr.Textbox(label="Nhap cau hoi phap ly"),
            gr.Slider(1, 10, value=5, step=1, label="So ket qua hien thi"),
            gr.Slider(10, 100, value=40, step=5, label="So ung vien truy xuat truoc khi rerank"),
        ],
        outputs=gr.Textbox(label="Ket qua Hybrid RAG + Reranking", lines=24),
        title="Hoi Dap Luat Viet Nam - Structure-aware Hybrid RAG",
        description=f"Truy xuat theo cau truc Dieu/Chuong, ket hop BM25 + Vector Search + Reranking. {llm_status}",
    )
    return iface


def main():
    iface = build_gradio_interface()
    share = os.getenv("GRADIO_SHARE", "false").strip().lower() in {"1", "true", "yes"}
    iface.launch(server_name="0.0.0.0", server_port=7860, share=share)


if __name__ == "__main__":
    main()
